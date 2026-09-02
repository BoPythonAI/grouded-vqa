from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from grounded_vqa.cli.train_complementary import sequence_token_nll
from grounded_vqa.cli.train_lora import resolve_data_files, seed_everything
from grounded_vqa.data.coco_grounding import load_grounding_samples
from grounded_vqa.data.collator import VQAGenerativeCollator, move_batch_to_device
from grounded_vqa.data.selection import select_samples
from grounded_vqa.data.vqav2 import load_vqav2
from grounded_vqa.models.loading import load_model, model_input_device
from grounded_vqa.models.lora import (
    add_trainable_qformer_to_frozen_adapter,
    save_adapter_stack,
    trainable_parameter_summary,
)
from grounded_vqa.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q-Former LoRA with COCO grounding data")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-kind", choices=["blip2", "instructblip"], required=True)
    parser.add_argument("--adapter-init", type=Path, required=True)
    parser.add_argument("--grounding-train", type=Path, required=True)
    parser.add_argument("--grounding-val", type=Path, required=True)
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="4bit")
    parser.add_argument("--qformer-rank", type=int, default=8)
    parser.add_argument("--qformer-alpha", type=int, default=16)
    parser.add_argument("--qformer-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--ordinary-pool-size", type=int, default=1000)
    parser.add_argument("--ordinary-examples", type=int, default=700)
    parser.add_argument("--grounding-examples", type=int, default=300)
    parser.add_argument("--validation-grounding-examples", type=int, default=128)
    parser.add_argument("--validation-vqa-examples", type=int, default=128)
    parser.add_argument("--vqa-nll-tolerance", type=float, default=0.005)
    parser.add_argument("--evaluation-exclusion-size", type=int, default=1000)
    parser.add_argument("--evaluation-exclusion-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument("--eval-every", type=int, default=50, help="Optimizer steps")
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=250)
    return parser.parse_args()


def evaluate_grounding_nll(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
) -> dict[str, float | int]:
    was_training = model.training
    model.eval()
    losses: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            batch, _ = move_batch_to_device(batch, device)
            losses.append(
                sequence_token_nll(model(**batch).logits, batch["labels"]).detach().cpu()
            )
    if was_training:
        model.train()
    all_losses = torch.cat(losses)
    return {
        "examples": int(all_losses.numel()),
        "mean_token_nll": float(all_losses.mean()),
    }


def main() -> None:
    args = parse_args()
    if not 0 <= args.ordinary_examples <= args.ordinary_pool_size:
        raise ValueError("ordinary-examples must be within ordinary-pool-size")
    if args.grounding_examples < 0:
        raise ValueError("grounding-examples must be non-negative")
    if args.ordinary_examples + args.grounding_examples <= 0:
        raise ValueError("At least one training example is required")
    seed_everything(args.seed)
    paths = ProjectPaths.from_environment()
    paths.ensure()
    output_dir = paths.output_root / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    questions, annotations, images_root = resolve_data_files(paths.data_root, "train")
    vqa_samples = load_vqav2(questions, annotations, images_root, "train")
    ordinary_pool = select_samples(vqa_samples, args.ordinary_pool_size, args.seed)
    ordinary_samples = ordinary_pool[: args.ordinary_examples]
    grounding_samples = select_samples(
        load_grounding_samples(args.grounding_train, images_root),
        args.grounding_examples,
        args.seed,
    )
    training_samples = [*ordinary_samples, *grounding_samples]
    validation_samples = select_samples(
        load_grounding_samples(args.grounding_val, images_root),
        args.validation_grounding_examples,
        args.seed,
    )
    val_questions, val_annotations, val_images_root = resolve_data_files(
        paths.data_root, "val"
    )
    vqa_validation_pool = load_vqav2(
        val_questions,
        val_annotations,
        val_images_root,
        "val",
    )
    evaluation_samples = select_samples(
        vqa_validation_pool,
        args.evaluation_exclusion_size,
        args.evaluation_exclusion_seed,
    )
    evaluation_question_ids = {sample.question_id for sample in evaluation_samples}
    vqa_validation_pool = [
        sample
        for sample in vqa_validation_pool
        if sample.question_id not in evaluation_question_ids
    ]
    vqa_validation_samples = select_samples(
        vqa_validation_pool,
        args.validation_vqa_examples,
        args.seed,
    )

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
        training_samples,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        validation_samples,
        batch_size=args.validation_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    vqa_val_loader = DataLoader(
        vqa_validation_samples,
        batch_size=args.validation_batch_size,
        shuffle=False,
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

    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "adapter_init": str(args.adapter_init),
                "grounding_train": str(args.grounding_train),
                "grounding_val": str(args.grounding_val),
                "actual_ordinary_examples": len(ordinary_samples),
                "actual_grounding_examples": len(grounding_samples),
                "grounding_fraction": len(grounding_samples) / len(training_samples),
                "actual_validation_vqa_examples": len(vqa_validation_samples),
                "evaluation_overlap_count": sum(
                    sample.question_id in evaluation_question_ids
                    for sample in vqa_validation_samples
                ),
                "trainable_parameters": parameter_report,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "selected_question_ids.json").write_text(
        json.dumps(
            {
                "ordinary_pool": [sample.question_id for sample in ordinary_pool],
                "ordinary_used": [sample.question_id for sample in ordinary_samples],
                "grounding_used": [sample.question_id for sample in grounding_samples],
                "grounding_validation": [
                    sample.question_id for sample in validation_samples
                ],
                "vqa_validation": [
                    sample.question_id for sample in vqa_validation_samples
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    validation_log = output_dir / "validation.jsonl"
    baseline_validation = evaluate_grounding_nll(model, val_loader, device)
    baseline_vqa_validation = evaluate_grounding_nll(model, vqa_val_loader, device)
    best_nll = float(baseline_validation["mean_token_nll"])
    baseline_vqa_nll = float(baseline_vqa_validation["mean_token_nll"])
    best_vqa_nll = baseline_vqa_nll
    best_optimizer_step = 0
    bad_validations = 0
    with validation_log.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "optimizer_step": 0,
                    "grounding_mean_token_nll": best_nll,
                    "vqa_mean_token_nll": baseline_vqa_nll,
                    "vqa_within_tolerance": True,
                    "is_best": True,
                }
            )
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
        for batch in progress:
            global_step += 1
            batch, metadata = move_batch_to_device(batch, device)
            loss = model(**batch).loss
            (loss / args.gradient_accumulation).backward()
            performed_optimizer_step = global_step % args.gradient_accumulation == 0
            if performed_optimizer_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            raw_loss = float(loss.detach())
            grounding_in_batch = sum(int(item["question_id"]) < 0 for item in metadata)
            progress.set_postfix(loss=f"{raw_loss:.4f}")
            if global_step % args.log_every == 0:
                with train_log.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "global_step": global_step,
                                "optimizer_step": optimizer_step,
                                "loss": raw_loss,
                                "grounding_in_batch": grounding_in_batch,
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
                validation = evaluate_grounding_nll(model, val_loader, device)
                vqa_validation = evaluate_grounding_nll(model, vqa_val_loader, device)
                current_nll = float(validation["mean_token_nll"])
                current_vqa_nll = float(vqa_validation["mean_token_nll"])
                vqa_within_tolerance = (
                    current_vqa_nll <= baseline_vqa_nll + args.vqa_nll_tolerance
                )
                is_best = (
                    current_nll < best_nll - args.early_stop_min_delta
                    and vqa_within_tolerance
                )
                if is_best:
                    best_nll = current_nll
                    best_vqa_nll = current_vqa_nll
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
                                "grounding_mean_token_nll": current_nll,
                                "vqa_mean_token_nll": current_vqa_nll,
                                "vqa_within_tolerance": vqa_within_tolerance,
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
                "best_validation_nll": best_nll,
                "best_vqa_validation_nll": best_vqa_nll,
                "best_optimizer_step": best_optimizer_step,
                "baseline_validation": baseline_validation,
                "baseline_vqa_validation": baseline_vqa_validation,
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
