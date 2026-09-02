from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from grounded_vqa.cli.train_lora import resolve_data_files, save_checkpoint, seed_everything
from grounded_vqa.data.collator import VQAGenerativeCollator, move_batch_to_device
from grounded_vqa.data.complementary import (
    build_complementary_pairs,
    load_complementary_pair_ids,
)
from grounded_vqa.data.selection import select_samples
from grounded_vqa.data.vqav2 import VQASample, load_vqav2
from grounded_vqa.models.loading import load_model, model_input_device
from grounded_vqa.models.lora import load_trainable_lora, trainable_parameter_summary
from grounded_vqa.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue LoRA training on VQAv2 complementary hard pairs"
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-kind", choices=["blip2", "instructblip"], required=True)
    parser.add_argument("--adapter-init", type=Path, required=True)
    parser.add_argument("--pairs-file", type=Path)
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="4bit")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1, help="Number of pairs per batch")
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=250)
    return parser.parse_args()


class ComplementaryPairDataset(Dataset[tuple[VQASample, VQASample]]):
    def __init__(self, pairs: list[tuple[VQASample, VQASample]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[VQASample, VQASample]:
        return self.pairs[index]


@dataclass
class ComplementaryPairCollator:
    base: VQAGenerativeCollator

    def __call__(
        self,
        pairs: list[tuple[VQASample, VQASample]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        positives: list[VQASample] = []
        negatives: list[VQASample] = []
        for first, second in pairs:
            positives.extend((first, second))
            negatives.extend(
                (
                    replace(first, image_id=second.image_id, image_path=second.image_path),
                    replace(second, image_id=first.image_id, image_path=first.image_path),
                )
            )
        return self.base(positives), self.base(negatives)


def sequence_token_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return one mean teacher-forced token NLL per sequence."""

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("Expected logits [batch, tokens, vocab] aligned with labels")
    token_losses = functional.cross_entropy(
        logits.float().transpose(1, 2),
        labels,
        ignore_index=-100,
        reduction="none",
    )
    mask = labels.ne(-100)
    if not bool(mask.any(dim=1).all()):
        raise ValueError("Every sequence must contain at least one target token")
    return (token_losses * mask).sum(dim=1) / mask.sum(dim=1)


def pairwise_logistic_loss(
    positive_nll: torch.Tensor,
    negative_nll: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Smoothly rank the correct image above the swapped complementary image."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if positive_nll.shape != negative_nll.shape:
        raise ValueError("Positive and negative NLL tensors must have equal shapes")
    return temperature * functional.softplus(
        (positive_nll - negative_nll) / temperature
    ).mean()


def main() -> None:
    args = parse_args()
    if args.contrastive_weight < 0:
        raise ValueError("contrastive-weight must be non-negative")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if args.max_pairs <= 0:
        raise ValueError("max-pairs must be positive")
    seed_everything(args.seed)

    paths = ProjectPaths.from_environment()
    paths.ensure()
    output_dir = paths.output_root / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_path = args.pairs_file or (
        paths.data_root
        / "complementary_pairs"
        / "v2_mscoco_train2014_complementary_pairs.json"
    )

    questions, annotations, images_root = resolve_data_files(paths.data_root, "train")
    samples = load_vqav2(questions, annotations, images_root, "train")
    pair_ids = load_complementary_pair_ids(pair_path)
    pairs, audit = build_complementary_pairs(samples, pair_ids)
    pairs = select_samples(pairs, args.max_pairs, args.seed)
    selected_question_ids = [sample.question_id for pair in pairs for sample in pair]
    (output_dir / "selected_question_ids.json").write_text(
        json.dumps(selected_question_ids, indent=2) + "\n",
        encoding="utf-8",
    )

    loaded = load_model(args.model_id, args.model_kind, args.quantization)
    model = load_trainable_lora(loaded.model, str(args.adapter_init))
    model.train()
    parameter_report = trainable_parameter_summary(model)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "adapter_init": str(args.adapter_init),
                "pairs_file": str(pair_path),
                "pair_audit": asdict(audit),
                "selected_pairs": len(pairs),
                "selected_positive_examples": 2 * len(pairs),
                "parameters": parameter_report,
                "objective": (
                    "mean(positive token NLL) + contrastive_weight * "
                    "temperature * softplus((positive NLL - swapped-image NLL) / "
                    "temperature)"
                ),
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = ComplementaryPairDataset(pairs)
    collator = ComplementaryPairCollator(
        VQAGenerativeCollator(loaded.processor, args.model_kind)
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    device = model_input_device(model)
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    optimizer_step = 0
    started = time.monotonic()
    log_path = output_dir / "train.jsonl"
    for epoch in range(args.epochs):
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for positive_batch, negative_batch in progress:
            global_step += 1
            positive_batch, _ = move_batch_to_device(positive_batch, device)
            positive_outputs = model(**positive_batch)
            positive_nll = sequence_token_nll(
                positive_outputs.logits,
                positive_batch["labels"],
            )
            vqa_loss = positive_nll.mean()

            if args.contrastive_weight > 0:
                negative_batch, _ = move_batch_to_device(negative_batch, device)
                negative_outputs = model(**negative_batch)
                negative_nll = sequence_token_nll(
                    negative_outputs.logits,
                    negative_batch["labels"],
                )
                contrastive_loss = pairwise_logistic_loss(
                    positive_nll,
                    negative_nll,
                    args.temperature,
                )
                preference_accuracy = positive_nll.lt(negative_nll).float().mean()
            else:
                negative_nll = None
                contrastive_loss = vqa_loss.detach().new_zeros(())
                preference_accuracy = vqa_loss.detach().new_zeros(())

            combined_loss = vqa_loss + args.contrastive_weight * contrastive_loss
            (combined_loss / args.gradient_accumulation).backward()
            if global_step % args.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            values = {
                "positive_nll": float(vqa_loss.detach()),
                "negative_nll": (
                    float(negative_nll.mean().detach()) if negative_nll is not None else None
                ),
                "contrastive_loss": float(contrastive_loss.detach()),
                "preference_accuracy": float(preference_accuracy.detach()),
                "combined_loss": float(combined_loss.detach()),
            }
            progress.set_postfix(loss=f"{values['combined_loss']:.4f}")
            if global_step % args.log_every == 0:
                record = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "optimizer_step": optimizer_step,
                    **values,
                    "elapsed_seconds": time.monotonic() - started,
                    "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
            if global_step % args.save_every == 0:
                save_checkpoint(model, output_dir, global_step)

    if global_step % args.gradient_accumulation:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    model.save_pretrained(output_dir / "final-adapter")
    loaded.processor.save_pretrained(output_dir / "processor")


if __name__ == "__main__":
    main()
