from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from grounded_vqa.cli.evaluate_grounding import compute_grounding_metrics
from grounded_vqa.cli.train_lora import resolve_data_files, seed_everything
from grounded_vqa.data.answers import consensus_answer
from grounded_vqa.data.coco_grounding import load_grounding_samples
from grounded_vqa.data.collator import VQAGenerativeCollator, move_batch_to_device
from grounded_vqa.data.selection import select_samples
from grounded_vqa.data.vqav2 import VQASample, load_vqav2
from grounded_vqa.evaluation.vqa import EvaluationExample, evaluate_examples
from grounded_vqa.models.loading import format_prompt, load_model, model_input_device
from grounded_vqa.models.lora import (
    add_trainable_qformer_to_frozen_adapter,
    qformer_lora_disabled,
    save_adapter_stack,
    trainable_parameter_summary,
)
from grounded_vqa.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Knowledge-preserving Q-Former grounding")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-kind", choices=["blip2", "instructblip"], required=True)
    parser.add_argument("--adapter-init", type=Path, required=True)
    parser.add_argument("--grounding-train", type=Path, required=True)
    parser.add_argument("--grounding-val", type=Path, required=True)
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="4bit")
    parser.add_argument("--qformer-rank", type=int, default=4)
    parser.add_argument("--qformer-alpha", type=int, default=8)
    parser.add_argument("--qformer-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--distillation-weight", type=float, default=0.5)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--ordinary-examples", type=int, default=1000)
    parser.add_argument("--grounding-examples", type=int, default=100)
    parser.add_argument("--validation-vqa-examples", type=int, default=64)
    parser.add_argument("--validation-grounding-examples", type=int, default=64)
    parser.add_argument("--vqa-accuracy-tolerance", type=float, default=0.2)
    parser.add_argument("--evaluation-exclusion-size", type=int, default=1000)
    parser.add_argument("--evaluation-exclusion-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=250)
    return parser.parse_args()


def distillation_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    token_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    mask = labels.ne(-100)
    if not mask.any():
        return student_logits.sum() * 0.0
    return token_kl[mask].mean() * temperature**2


def generate_answers(
    model: torch.nn.Module,
    processor: Any,
    model_kind: str,
    samples: list[VQASample],
    device: torch.device,
    *,
    batch_size: int,
) -> list[str]:
    was_training = model.training
    model.eval()
    predictions: list[str] = []
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            images: list[Image.Image] = []
            prompts: list[str] = []
            for sample in batch:
                with Image.open(sample.image_path) as image:
                    images.append(image.convert("RGB"))
                prompts.append(format_prompt(model_kind, sample.question, "short"))
            inputs = processor(
                images=images,
                text=prompts,
                padding=True,
                return_tensors="pt",
            )
            for key, value in list(inputs.items()):
                if isinstance(value, torch.Tensor):
                    value = value.to(device)
                    if key == "pixel_values":
                        value = value.to(dtype=torch.bfloat16)
                    inputs[key] = value
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=10,
            )
            predictions.extend(
                answer.strip()
                for answer in processor.batch_decode(generated, skip_special_tokens=True)
            )
    if was_training:
        model.train()
    return predictions


def evaluate_generation(
    model: torch.nn.Module,
    processor: Any,
    model_kind: str,
    vqa_samples: list[VQASample],
    grounding_samples: list[VQASample],
    device: torch.device,
    *,
    batch_size: int,
) -> dict[str, Any]:
    vqa_predictions = generate_answers(
        model,
        processor,
        model_kind,
        vqa_samples,
        device,
        batch_size=batch_size,
    )
    grounding_predictions = generate_answers(
        model,
        processor,
        model_kind,
        grounding_samples,
        device,
        batch_size=batch_size,
    )
    vqa_metrics = evaluate_examples(
        EvaluationExample(
            prediction=prediction,
            references=sample.answers,
            answer_type=sample.answer_type or "unknown",
            question_type=sample.question_type or "unknown",
        )
        for sample, prediction in zip(vqa_samples, vqa_predictions, strict=True)
    )
    grounding_rows = [
        {
            "prediction": prediction,
            "answer": consensus_answer(sample.answers),
            "task_type": sample.question_type,
        }
        for sample, prediction in zip(
            grounding_samples,
            grounding_predictions,
            strict=True,
        )
    ]
    return {
        "vqa": vqa_metrics,
        "grounding": compute_grounding_metrics(grounding_rows),
    }


def main() -> None:
    args = parse_args()
    if args.ordinary_examples <= 0 or args.grounding_examples < 0:
        raise ValueError("ordinary-examples must be positive and grounding-examples non-negative")
    seed_everything(args.seed)
    paths = ProjectPaths.from_environment()
    paths.ensure()
    output_dir = paths.output_root / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_q, train_a, images_root = resolve_data_files(paths.data_root, "train")
    train_samples = load_vqav2(train_q, train_a, images_root, "train")
    ordinary_samples = select_samples(train_samples, args.ordinary_examples, args.seed)
    grounding_pool = load_grounding_samples(args.grounding_train, images_root)
    grounding_samples = select_samples(grounding_pool, args.grounding_examples, args.seed)
    training_samples = [*ordinary_samples, *grounding_samples]

    grounding_validation = select_samples(
        load_grounding_samples(args.grounding_val, images_root),
        args.validation_grounding_examples,
        args.seed,
    )
    val_q, val_a, val_images_root = resolve_data_files(paths.data_root, "val")
    vqa_validation_pool = load_vqav2(val_q, val_a, val_images_root, "val")
    reporting_samples = select_samples(
        vqa_validation_pool,
        args.evaluation_exclusion_size,
        args.evaluation_exclusion_seed,
    )
    reporting_ids = {sample.question_id for sample in reporting_samples}
    vqa_validation_pool = [
        sample for sample in vqa_validation_pool if sample.question_id not in reporting_ids
    ]
    vqa_validation = select_samples(
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
    # Keep both teacher and student deterministic. Eval mode does not disable
    # gradients, so Q-Former LoRA still trains while frozen-model dropout stays off.
    model.eval()
    device = model_input_device(model)
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
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)

    run_config = {
        **vars(args),
        "adapter_init": str(args.adapter_init),
        "grounding_train": str(args.grounding_train),
        "grounding_val": str(args.grounding_val),
        "actual_ordinary_examples": len(ordinary_samples),
        "actual_grounding_examples": len(grounding_samples),
        "training_examples": len(training_samples),
        "evaluation_overlap_count": sum(
            sample.question_id in reporting_ids for sample in vqa_validation
        ),
        "trainable_parameters": trainable_parameter_summary(model),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (output_dir / "selected_question_ids.json").write_text(
        json.dumps(
            {
                "ordinary": [sample.question_id for sample in ordinary_samples],
                "grounding": [sample.question_id for sample in grounding_samples],
                "vqa_validation": [sample.question_id for sample in vqa_validation],
                "grounding_validation": [
                    sample.question_id for sample in grounding_validation
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    baseline = evaluate_generation(
        model,
        loaded.processor,
        args.model_kind,
        vqa_validation,
        grounding_validation,
        device,
        batch_size=args.validation_batch_size,
    )
    best_grounding = float(baseline["grounding"]["overall_exact_accuracy"])
    best_vqa = float(baseline["vqa"]["overall"])
    baseline_vqa = best_vqa
    best_step = 0
    bad_validations = 0
    validation_log = output_dir / "validation.jsonl"
    validation_log.write_text(
        json.dumps({"optimizer_step": 0, **baseline, "is_best": True}) + "\n",
        encoding="utf-8",
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
            ordinary_batch = all(int(item["question_id"]) >= 0 for item in metadata)
            teacher_logits: torch.Tensor | None = None
            if ordinary_batch and args.distillation_weight > 0:
                with qformer_lora_disabled(model), torch.no_grad():
                    teacher_logits = model(**batch).logits.detach()

            student_output = model(**batch)
            ce_loss = student_output.loss
            kl_loss = ce_loss.new_zeros(())
            if teacher_logits is not None:
                kl_loss = distillation_kl(
                    student_output.logits,
                    teacher_logits,
                    batch["labels"],
                    temperature=args.distillation_temperature,
                )
            loss = ce_loss + args.distillation_weight * kl_loss
            (loss / args.gradient_accumulation).backward()
            performed_step = global_step % args.gradient_accumulation == 0
            if performed_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            if global_step % args.log_every == 0:
                with train_log.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "global_step": global_step,
                                "optimizer_step": optimizer_step,
                                "loss": float(loss.detach()),
                                "ce_loss": float(ce_loss.detach()),
                                "kl_loss": float(kl_loss.detach()),
                                "ordinary_batch": ordinary_batch,
                                "elapsed_seconds": time.monotonic() - started,
                                "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                            }
                        )
                        + "\n"
                    )
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")

            should_validate = (
                performed_step
                and args.eval_every > 0
                and optimizer_step % args.eval_every == 0
            )
            if should_validate:
                validation = evaluate_generation(
                    model,
                    loaded.processor,
                    args.model_kind,
                    vqa_validation,
                    grounding_validation,
                    device,
                    batch_size=args.validation_batch_size,
                )
                current_vqa = float(validation["vqa"]["overall"])
                current_grounding = float(
                    validation["grounding"]["overall_exact_accuracy"]
                )
                vqa_within_tolerance = (
                    current_vqa >= baseline_vqa - args.vqa_accuracy_tolerance
                )
                is_best = vqa_within_tolerance and (
                    current_grounding > best_grounding
                    or (current_grounding == best_grounding and current_vqa > best_vqa)
                )
                if is_best:
                    best_grounding = current_grounding
                    best_vqa = current_vqa
                    best_step = optimizer_step
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
            if performed_step and optimizer_step % args.save_every == 0:
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
                "best_optimizer_step": best_step,
                "best_vqa_accuracy": best_vqa,
                "best_grounding_accuracy": best_grounding,
                "baseline": baseline,
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
