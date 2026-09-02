from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from grounded_vqa.cli.train_distilled_grounding import distillation_kl
from grounded_vqa.cli.train_lora import resolve_data_files, seed_everything
from grounded_vqa.data.coco_grounding import load_grounding_samples
from grounded_vqa.data.collator import VQAGenerativeCollator, move_batch_to_device
from grounded_vqa.data.selection import select_samples
from grounded_vqa.data.vqav2 import load_vqav2
from grounded_vqa.models.loading import load_model, model_input_device
from grounded_vqa.models.lora import load_trainable_lora, trainable_parameter_summary
from grounded_vqa.paths import ProjectPaths

STUDENT_ADAPTER = "default"
TEACHER_ADAPTER = "teacher"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue E6 with VQA rehearsal, hard negatives, and frozen-adapter KL"
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-kind", choices=["blip2", "instructblip"], required=True)
    parser.add_argument("--adapter-init", type=Path, required=True)
    parser.add_argument("--negative-train", type=Path, required=True)
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="4bit")
    parser.add_argument("--ordinary-examples", type=int, default=9000)
    parser.add_argument("--negative-examples", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--negative-loss-weight", type=float, default=1.0)
    parser.add_argument("--distillation-weight", type=float, default=0.5)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--distillation-interval", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=5000)
    return parser.parse_args()


def set_active_adapter(model: torch.nn.Module, adapter: str) -> None:
    model.set_adapter(adapter)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = adapter == STUDENT_ADAPTER and ".default." in name


def save_student_adapter(model: torch.nn.Module, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    set_active_adapter(model, STUDENT_ADAPTER)
    model.save_pretrained(output_dir, selected_adapters=[STUDENT_ADAPTER])


def per_sample_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Return token-mean cross entropy for each sample in a padded batch."""
    token_losses = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(labels.shape)
    valid_tokens = labels.ne(-100)
    token_counts = valid_tokens.sum(dim=1).clamp_min(1)
    return (token_losses * valid_tokens).sum(dim=1) / token_counts


def select_batch_rows(
    batch: dict[str, object], row_mask: torch.Tensor
) -> dict[str, object]:
    """Select examples without slicing scalar/non-batched model arguments."""
    batch_size = int(row_mask.shape[0])
    return {
        key: (
            value[row_mask]
            if isinstance(value, torch.Tensor)
            and value.ndim > 0
            and value.shape[0] == batch_size
            else value
        )
        for key, value in batch.items()
    }


def main() -> None:
    args = parse_args()
    if args.ordinary_examples <= 0 or args.negative_examples <= 0:
        raise ValueError("ordinary-examples and negative-examples must be positive")
    if (
        args.distillation_interval <= 0
        or args.batch_size <= 0
        or args.gradient_accumulation <= 0
    ):
        raise ValueError(
            "distillation-interval, batch-size, and gradient-accumulation must be positive"
        )
    seed_everything(args.seed)
    paths = ProjectPaths.from_environment()
    paths.ensure()
    output_dir = paths.output_root / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_questions, train_annotations, images_root = resolve_data_files(
        paths.data_root, "train"
    )
    ordinary_pool = load_vqav2(
        train_questions, train_annotations, images_root, "train"
    )
    ordinary = select_samples(ordinary_pool, args.ordinary_examples, args.seed)
    negative_pool = load_grounding_samples(args.negative_train, images_root)
    negative = select_samples(negative_pool, args.negative_examples, args.seed)
    training_samples = [*ordinary, *negative]

    loaded = load_model(args.model_id, args.model_kind, args.quantization)
    model = load_trainable_lora(loaded.model, str(args.adapter_init))
    model.load_adapter(
        str(args.adapter_init), adapter_name=TEACHER_ADAPTER, is_trainable=False
    )
    set_active_adapter(model, STUDENT_ADAPTER)
    model.eval()
    device = model_input_device(model)
    collator = VQAGenerativeCollator(loaded.processor, args.model_kind)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        training_samples,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    optimizer.zero_grad(set_to_none=True)

    config = {
        **vars(args),
        "adapter_init": str(args.adapter_init),
        "negative_train": str(args.negative_train),
        "training_examples": len(training_samples),
        "actual_ordinary_examples": len(ordinary),
        "actual_negative_examples": len(negative),
        "parameters": trainable_parameter_summary(model),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (output_dir / "selected_question_ids.json").write_text(
        json.dumps(
            {
                "ordinary": [sample.question_id for sample in ordinary],
                "negative": [sample.question_id for sample in negative],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    global_step = 0
    batch_step = 0
    optimizer_step = 0
    started = time.monotonic()
    log_path = output_dir / "train.jsonl"
    for epoch in range(args.epochs):
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            batch_step += 1
            batch, metadata = move_batch_to_device(batch, device)
            batch_size = len(metadata)
            sample_positions = torch.arange(
                global_step + 1,
                global_step + batch_size + 1,
                device=device,
            )
            ordinary_mask = torch.tensor(
                [int(item["question_id"]) >= 0 for item in metadata],
                device=device,
                dtype=torch.bool,
            )
            distillation_mask = ordinary_mask & sample_positions.remainder(
                args.distillation_interval
            ).eq(0)
            global_step += batch_size
            teacher_logits: torch.Tensor | None = None
            if distillation_mask.any() and args.distillation_weight > 0:
                set_active_adapter(model, TEACHER_ADAPTER)
                with torch.no_grad():
                    teacher_batch = select_batch_rows(batch, distillation_mask)
                    teacher_logits = model(**teacher_batch).logits.detach()
                set_active_adapter(model, STUDENT_ADAPTER)

            student = model(**batch)
            sample_ce = per_sample_cross_entropy(student.logits, batch["labels"])
            sample_weights = torch.where(
                ordinary_mask,
                sample_ce.new_ones(()),
                sample_ce.new_full((), args.negative_loss_weight),
            )
            ce_loss = (sample_ce * sample_weights).mean()
            kl_loss = ce_loss.new_zeros(())
            if teacher_logits is not None:
                kl_loss = distillation_kl(
                    student.logits[distillation_mask],
                    teacher_logits,
                    batch["labels"][distillation_mask],
                    temperature=args.distillation_temperature,
                )
            loss = ce_loss + args.distillation_weight * kl_loss
            (loss / args.gradient_accumulation).backward()
            if batch_step % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            if global_step % args.log_every == 0:
                with log_path.open("a", encoding="utf-8") as output:
                    output.write(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "global_step": global_step,
                                "batch_step": batch_step,
                                "batch_size": batch_size,
                                "optimizer_step": optimizer_step,
                                "loss": float(loss.detach()),
                                "ce_loss": float(ce_loss.detach()),
                                "kl_loss": float(kl_loss.detach()),
                                "ordinary_examples": int(ordinary_mask.sum()),
                                "negative_examples": int((~ordinary_mask).sum()),
                                "distilled": teacher_logits is not None,
                                "elapsed_seconds": time.monotonic() - started,
                                "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                            }
                        )
                        + "\n"
                    )
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
            if args.save_every > 0 and global_step % args.save_every == 0:
                save_student_adapter(
                    model, output_dir / f"checkpoint-{global_step:08d}"
                )

    if batch_step % args.gradient_accumulation:
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1
    save_student_adapter(model, output_dir / "final-adapter")
    loaded.processor.save_pretrained(output_dir / "processor")
    (output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "global_step": global_step,
                "optimizer_step": optimizer_step,
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
