from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from grounded_vqa.cli.train_lora import resolve_data_files, save_checkpoint, seed_everything
from grounded_vqa.data.collator import VQAGenerativeCollator, move_batch_to_device
from grounded_vqa.data.selection import build_mismatched_indices, select_samples
from grounded_vqa.data.vqav2 import VQASample, load_vqav2
from grounded_vqa.models.loading import load_model, model_input_device
from grounded_vqa.models.lora import load_trainable_lora, trainable_parameter_summary
from grounded_vqa.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continue LoRA training with mismatch loss")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-kind", choices=["blip2", "instructblip"], required=True)
    parser.add_argument("--adapter-init", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="4bit")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--mismatch-weight", type=float, default=0.1)
    parser.add_argument("--mismatch-margin", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=250)
    return parser.parse_args()


class MismatchPairDataset(Dataset[tuple[VQASample, VQASample]]):
    def __init__(self, samples: list[VQASample], mismatch_indices: list[int]) -> None:
        if len(samples) != len(mismatch_indices):
            raise ValueError("Samples and mismatch permutation must have equal length")
        self.samples = samples
        self.mismatch_indices = mismatch_indices

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[VQASample, VQASample]:
        positive = self.samples[index]
        mismatched_image = self.samples[self.mismatch_indices[index]]
        negative = replace(
            positive,
            image_id=mismatched_image.image_id,
            image_path=mismatched_image.image_path,
        )
        return positive, negative


@dataclass
class MismatchPairCollator:
    base: VQAGenerativeCollator

    def __call__(
        self,
        pairs: list[tuple[VQASample, VQASample]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        positives = [positive for positive, _ in pairs]
        negatives = [negative for _, negative in pairs]
        return self.base(positives), self.base(negatives)


def mismatch_margin_loss(
    positive_nll: torch.Tensor,
    negative_nll: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    return functional.relu(margin + positive_nll - negative_nll)


def main() -> None:
    args = parse_args()
    if args.mismatch_weight < 0:
        raise ValueError("mismatch-weight must be non-negative")
    if args.mismatch_margin < 0:
        raise ValueError("mismatch-margin must be non-negative")
    seed_everything(args.seed)
    paths = ProjectPaths.from_environment()
    paths.ensure()
    output_dir = paths.output_root / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_model(args.model_id, args.model_kind, args.quantization)
    model = load_trainable_lora(loaded.model, str(args.adapter_init))
    model.train()
    parameter_report = trainable_parameter_summary(model)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "adapter_init": str(args.adapter_init),
                "parameters": parameter_report,
                "mismatch_policy": "seeded different-image permutation",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    questions, annotations, images_root = resolve_data_files(paths.data_root, args.split)
    samples = load_vqav2(questions, annotations, images_root, args.split)
    samples = select_samples(samples, args.max_samples, args.seed)
    mismatch_indices = build_mismatched_indices(
        [sample.image_id for sample in samples],
        args.seed,
    )
    dataset = MismatchPairDataset(samples, mismatch_indices)
    collator = MismatchPairCollator(VQAGenerativeCollator(loaded.processor, args.model_kind))
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
            positive_nll = model(**positive_batch).loss

            if args.mismatch_weight > 0:
                negative_batch, _ = move_batch_to_device(negative_batch, device)
                negative_nll = model(**negative_batch).loss
                alignment_loss = mismatch_margin_loss(
                    positive_nll,
                    negative_nll,
                    args.mismatch_margin,
                )
            else:
                negative_nll = None
                alignment_loss = positive_nll.detach().new_zeros(())

            combined_loss = positive_nll + args.mismatch_weight * alignment_loss
            (combined_loss / args.gradient_accumulation).backward()
            if global_step % args.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            values = {
                "positive_nll": float(positive_nll.detach()),
                "negative_nll": (
                    float(negative_nll.detach()) if negative_nll is not None else None
                ),
                "mismatch_loss": float(alignment_loss.detach()),
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
