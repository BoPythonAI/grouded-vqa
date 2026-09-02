from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from grounded_vqa.data.collator import VQAGenerativeCollator, move_batch_to_device
from grounded_vqa.data.selection import select_answer_type_fraction, select_samples
from grounded_vqa.data.vqav2 import load_vqav2
from grounded_vqa.models.loading import load_model, model_input_device
from grounded_vqa.models.lora import attach_lora, trainable_parameter_summary
from grounded_vqa.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning on VQAv2")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-kind", choices=["blip2", "instructblip"], required=True)
    parser.add_argument("--scope", choices=["qformer", "llm", "dual"], required=True)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="4bit")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument(
        "--answer-target",
        choices=["consensus", "frequency"],
        default="consensus",
    )
    parser.add_argument(
        "--number-fraction",
        type=float,
        help="Exact fraction of number-answer examples in the selected training subset",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_data_files(data_root: Path, split: str) -> tuple[Path, Path, Path]:
    names = {
        "train": (
            "v2_OpenEnded_mscoco_train2014_questions.json",
            "v2_mscoco_train2014_annotations.json",
        ),
        "val": (
            "v2_OpenEnded_mscoco_val2014_questions.json",
            "v2_mscoco_val2014_annotations.json",
        ),
    }
    question_name, annotation_name = names[split]
    return data_root / question_name, data_root / annotation_name, data_root


def save_checkpoint(model: torch.nn.Module, output_dir: Path, step: int) -> None:
    checkpoint = output_dir / f"checkpoint-{step:08d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    paths = ProjectPaths.from_environment()
    paths.ensure()
    output_dir = paths.output_root / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_model(args.model_id, args.model_kind, args.quantization)
    model = attach_lora(
        loaded.model,
        args.scope,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
    )
    model.train()
    parameter_report = trainable_parameter_summary(model)
    (output_dir / "run_config.json").write_text(
        json.dumps({**vars(args), "parameters": parameter_report}, indent=2) + "\n",
        encoding="utf-8",
    )

    questions, annotations, images_root = resolve_data_files(paths.data_root, args.split)
    samples = load_vqav2(questions, annotations, images_root, args.split)
    if args.number_fraction is None:
        samples = select_samples(samples, args.max_samples, args.seed)
    else:
        if args.max_samples is None:
            raise ValueError("--number-fraction requires --max-samples")
        samples = select_answer_type_fraction(
            samples,
            args.max_samples,
            answer_type="number",
            fraction=args.number_fraction,
            seed=args.seed,
        )
    collator = VQAGenerativeCollator(
        loaded.processor,
        args.model_kind,
        answer_target=args.answer_target,
        answer_seed=args.seed,
    )
    loader = DataLoader(
        samples,
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
        for batch in progress:
            global_step += 1
            batch, _ = move_batch_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation
            loss.backward()

            if global_step % args.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            raw_loss = float(loss.detach()) * args.gradient_accumulation
            progress.set_postfix(loss=f"{raw_loss:.4f}")
            if global_step % args.log_every == 0:
                record = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "optimizer_step": optimizer_step,
                    "loss": raw_loss,
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
