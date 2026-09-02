from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from grounded_vqa.cli.train_complementary import (
    ComplementaryPairCollator,
    ComplementaryPairDataset,
    pairwise_logistic_loss,
    sequence_token_nll,
)
from grounded_vqa.cli.train_lora import resolve_data_files, seed_everything
from grounded_vqa.data.collator import VQAGenerativeCollator, move_batch_to_device
from grounded_vqa.data.complementary import (
    build_complementary_pairs,
    load_complementary_pair_ids,
)
from grounded_vqa.data.selection import select_samples
from grounded_vqa.data.vqav2 import VQASample, load_vqav2
from grounded_vqa.models.loading import load_model, model_input_device
from grounded_vqa.models.lora import (
    add_trainable_qformer_to_frozen_adapter,
    save_adapter_stack,
    trainable_parameter_summary,
)
from grounded_vqa.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mixed VQAv2 and hard-pair training with Q-Former LoRA"
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-kind", choices=["blip2", "instructblip"], required=True)
    parser.add_argument("--adapter-init", type=Path, required=True)
    parser.add_argument("--hard-pairs-file", type=Path, required=True)
    parser.add_argument("--val-pairs-file", type=Path)
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="4bit")
    parser.add_argument("--qformer-rank", type=int, default=8)
    parser.add_argument("--qformer-alpha", type=int, default=16)
    parser.add_argument("--qformer-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--contrastive-weight", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--ordinary-examples", type=int, default=700)
    parser.add_argument("--hard-pairs", type=int, default=150)
    parser.add_argument("--validation-pairs", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1, help="Number of two-example units")
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--eval-every", type=int, default=50, help="Optimizer steps")
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=250)
    return parser.parse_args()


@dataclass(frozen=True)
class MixedUnit:
    first: VQASample
    second: VQASample
    is_hard_pair: bool


class MixedUnitDataset(Dataset[MixedUnit]):
    def __init__(self, units: list[MixedUnit]) -> None:
        self.units = units

    def __len__(self) -> int:
        return len(self.units)

    def __getitem__(self, index: int) -> MixedUnit:
        return self.units[index]


@dataclass
class MixedUnitCollator:
    base: VQAGenerativeCollator

    def __call__(
        self,
        units: list[MixedUnit],
    ) -> tuple[dict[str, Any], dict[str, Any] | None, list[int]]:
        positives: list[VQASample] = []
        negatives: list[VQASample] = []
        hard_positive_indices: list[int] = []
        for unit in units:
            first_index = len(positives)
            positives.extend((unit.first, unit.second))
            if unit.is_hard_pair:
                hard_positive_indices.extend((first_index, first_index + 1))
                negatives.extend(
                    (
                        replace(
                            unit.first,
                            image_id=unit.second.image_id,
                            image_path=unit.second.image_path,
                        ),
                        replace(
                            unit.second,
                            image_id=unit.first.image_id,
                            image_path=unit.first.image_path,
                        ),
                    )
                )
        negative_batch = self.base(negatives) if negatives else None
        return self.base(positives), negative_batch, hard_positive_indices


def build_mixed_units(
    ordinary_samples: list[VQASample],
    hard_pairs: list[tuple[VQASample, VQASample]],
) -> list[MixedUnit]:
    if len(ordinary_samples) % 2:
        raise ValueError("ordinary_samples must contain an even number of examples")
    units = [
        MixedUnit(ordinary_samples[index], ordinary_samples[index + 1], False)
        for index in range(0, len(ordinary_samples), 2)
    ]
    units.extend(MixedUnit(first, second, True) for first, second in hard_pairs)
    return units


def evaluate_pair_margin(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
) -> dict[str, float | int]:
    was_training = model.training
    model.eval()
    margins: list[torch.Tensor] = []
    with torch.inference_mode():
        for positive_batch, negative_batch in loader:
            positive_batch, _ = move_batch_to_device(positive_batch, device)
            negative_batch, _ = move_batch_to_device(negative_batch, device)
            positive_nll = sequence_token_nll(
                model(**positive_batch).logits,
                positive_batch["labels"],
            )
            negative_nll = sequence_token_nll(
                model(**negative_batch).logits,
                negative_batch["labels"],
            )
            margins.append((negative_nll - positive_nll).detach().cpu())
    if was_training:
        model.train()
    all_margins = torch.cat(margins)
    pair_margins = all_margins.reshape(-1, 2).mean(dim=1)
    return {
        "pairs": int(pair_margins.numel()),
        "mean_margin": float(pair_margins.mean()),
        "pair_preference_accuracy": float(100.0 * pair_margins.gt(0).float().mean()),
        "direction_preference_accuracy": float(100.0 * all_margins.gt(0).float().mean()),
    }


def main() -> None:
    args = parse_args()
    if args.ordinary_examples <= 0 or args.ordinary_examples % 2:
        raise ValueError("ordinary-examples must be a positive even number")
    if args.hard_pairs <= 0 or args.validation_pairs <= 0:
        raise ValueError("hard-pairs and validation-pairs must be positive")
    if args.contrastive_weight < 0 or args.temperature <= 0:
        raise ValueError("Invalid contrastive weight or temperature")
    seed_everything(args.seed)
    paths = ProjectPaths.from_environment()
    paths.ensure()
    output_dir = paths.output_root / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    val_pair_path = args.val_pairs_file or (
        paths.data_root
        / "complementary_pairs"
        / "v2_mscoco_val2014_complementary_pairs.json"
    )

    train_questions, train_annotations, images_root = resolve_data_files(
        paths.data_root, "train"
    )
    train_samples = load_vqav2(
        train_questions, train_annotations, images_root, "train"
    )
    ordinary_samples = select_samples(train_samples, args.ordinary_examples, args.seed)
    hard_pair_ids = load_complementary_pair_ids(args.hard_pairs_file)
    all_hard_pairs, hard_audit = build_complementary_pairs(train_samples, hard_pair_ids)
    hard_pairs = select_samples(all_hard_pairs, args.hard_pairs, args.seed)
    units = build_mixed_units(ordinary_samples, hard_pairs)

    val_questions, val_annotations, val_images_root = resolve_data_files(
        paths.data_root, "val"
    )
    val_samples = load_vqav2(val_questions, val_annotations, val_images_root, "val")
    val_pair_ids = load_complementary_pair_ids(val_pair_path)
    all_val_pairs, val_audit = build_complementary_pairs(val_samples, val_pair_ids)
    val_pairs = select_samples(all_val_pairs, args.validation_pairs, args.seed)

    loaded = load_model(args.model_id, args.model_kind, args.quantization)
    model = add_trainable_qformer_to_frozen_adapter(
        loaded.model,
        str(args.adapter_init),
        rank=args.qformer_rank,
        alpha=args.qformer_alpha,
        dropout=args.qformer_dropout,
    )
    model.train()
    parameter_report = trainable_parameter_summary(model)
    collator = VQAGenerativeCollator(loaded.processor, args.model_kind)
    loader = DataLoader(
        MixedUnitDataset(units),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=MixedUnitCollator(collator),
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        ComplementaryPairDataset(val_pairs),
        batch_size=args.validation_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ComplementaryPairCollator(collator),
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

    ordinary_positive_count = len(ordinary_samples)
    hard_positive_count = 2 * len(hard_pairs)
    config = {
        **vars(args),
        "adapter_init": str(args.adapter_init),
        "hard_pairs_file": str(args.hard_pairs_file),
        "val_pairs_file": str(val_pair_path),
        "hard_pair_audit": asdict(hard_audit),
        "validation_pair_audit": asdict(val_audit),
        "ordinary_positive_examples": ordinary_positive_count,
        "hard_positive_examples": hard_positive_count,
        "hard_positive_fraction": hard_positive_count
        / (ordinary_positive_count + hard_positive_count),
        "trainable_parameters": parameter_report,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (output_dir / "selected_question_ids.json").write_text(
        json.dumps(
            {
                "ordinary": [sample.question_id for sample in ordinary_samples],
                "hard_pairs": [
                    [first.question_id, second.question_id] for first, second in hard_pairs
                ],
                "validation_pairs": [
                    [first.question_id, second.question_id] for first, second in val_pairs
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    validation_log = output_dir / "validation.jsonl"
    baseline_validation = evaluate_pair_margin(model, val_loader, device)
    best_margin = float(baseline_validation["mean_margin"])
    best_optimizer_step = 0
    bad_validations = 0
    with validation_log.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"optimizer_step": 0, **baseline_validation, "is_best": True})
            + "\n"
        )
    save_adapter_stack(model, output_dir / "best-adapter")

    global_step = 0
    optimizer_step = 0
    stopped_early = False
    started = time.monotonic()
    train_log = output_dir / "train.jsonl"
    for epoch in range(args.epochs):
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for positive_batch, negative_batch, hard_indices in progress:
            global_step += 1
            positive_batch, _ = move_batch_to_device(positive_batch, device)
            positive_nll = sequence_token_nll(
                model(**positive_batch).logits,
                positive_batch["labels"],
            )
            vqa_loss = positive_nll.mean()
            if negative_batch is not None and args.contrastive_weight > 0:
                negative_batch, _ = move_batch_to_device(negative_batch, device)
                negative_nll = sequence_token_nll(
                    model(**negative_batch).logits,
                    negative_batch["labels"],
                )
                hard_positive_nll = positive_nll[
                    torch.as_tensor(hard_indices, device=positive_nll.device)
                ]
                contrastive_loss = pairwise_logistic_loss(
                    hard_positive_nll,
                    negative_nll,
                    args.temperature,
                )
                preference_accuracy = hard_positive_nll.lt(negative_nll).float().mean()
            else:
                negative_nll = None
                contrastive_loss = vqa_loss.detach().new_zeros(())
                preference_accuracy = vqa_loss.detach().new_zeros(())
            combined_loss = vqa_loss + args.contrastive_weight * contrastive_loss
            (combined_loss / args.gradient_accumulation).backward()

            performed_optimizer_step = global_step % args.gradient_accumulation == 0
            if performed_optimizer_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            values = {
                "vqa_loss": float(vqa_loss.detach()),
                "negative_nll": (
                    float(negative_nll.mean().detach()) if negative_nll is not None else None
                ),
                "contrastive_loss": float(contrastive_loss.detach()),
                "preference_accuracy": float(preference_accuracy.detach()),
                "combined_loss": float(combined_loss.detach()),
                "hard_unit": bool(hard_indices),
            }
            progress.set_postfix(loss=f"{values['combined_loss']:.4f}")
            if global_step % args.log_every == 0:
                with train_log.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "global_step": global_step,
                                "optimizer_step": optimizer_step,
                                **values,
                                "elapsed_seconds": time.monotonic() - started,
                                "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                            }
                        )
                        + "\n"
                    )

            should_validate = (
                performed_optimizer_step
                and args.eval_every > 0
                and optimizer_step % args.eval_every == 0
            )
            if should_validate:
                validation = evaluate_pair_margin(model, val_loader, device)
                current_margin = float(validation["mean_margin"])
                is_best = current_margin > best_margin + args.early_stop_min_delta
                if is_best:
                    best_margin = current_margin
                    best_optimizer_step = optimizer_step
                    bad_validations = 0
                    save_adapter_stack(model, output_dir / "best-adapter")
                else:
                    bad_validations += 1
                with validation_log.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "optimizer_step": optimizer_step,
                                **validation,
                                "is_best": is_best,
                                "bad_validations": bad_validations,
                            }
                        )
                        + "\n"
                    )
                if bad_validations >= args.early_stop_patience:
                    stopped_early = True
                    break
            if performed_optimizer_step and optimizer_step % args.save_every == 0:
                save_adapter_stack(model, output_dir / f"checkpoint-{optimizer_step:08d}")
        if stopped_early:
            break

    if global_step % args.gradient_accumulation and not stopped_early:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1
    save_adapter_stack(model, output_dir / "final-adapter")
    loaded.processor.save_pretrained(output_dir / "processor")
    (output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "global_step": global_step,
                "optimizer_step": optimizer_step,
                "stopped_early": stopped_early,
                "best_validation_margin": best_margin,
                "best_optimizer_step": best_optimizer_step,
                "baseline_validation": baseline_validation,
                "elapsed_seconds": time.monotonic() - started,
                "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
