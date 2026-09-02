from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from grounded_vqa.evaluation.hallusionbench import compute_hallusionbench_metrics, row_key
from grounded_vqa.models.loading import format_prompt, load_model, model_input_device
from grounded_vqa.models.lora import load_inference_lora
from grounded_vqa.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate strict yes/no HallusionBench")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-kind", choices=["blip2", "instructblip"], required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--parquet-root", type=Path)
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="4bit")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-name", required=True)
    return parser.parse_args()


def load_rows(parquet_root: Path) -> list[dict[str, Any]]:
    files = {
        "image": parquet_root / "image-00000-of-00001.parquet",
        "non_image": parquet_root / "non_image-00000-of-00001.parquet",
    }
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    dataset = load_dataset(
        "parquet", data_files={name: str(path) for name, path in files.items()}
    )
    rows = [dict(row) for split in ("image", "non_image") for row in dataset[split]]
    return sorted(rows, key=row_key)


def serializable_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "image"}


def load_resume_rows(path: Path, source: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if [row_key(row) for row in rows] != [row_key(row) for row in source[: len(rows)]]:
        raise ValueError("HallusionBench resume file is not an exact prefix")
    return rows


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_environment()
    project_root = paths.data_root.parent.parent
    parquet_root = (
        args.parquet_root
        or project_root / "data" / "hallusionbench" / "hf" / "data"
    )
    source_rows = load_rows(parquet_root)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("max-samples must be positive")
        source_rows = source_rows[: args.max_samples]
    output_dir = paths.output_root / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    predictions = load_resume_rows(predictions_path, source_rows) if args.resume else []

    loaded = load_model(args.model_id, args.model_kind, args.quantization)
    model = loaded.model
    if args.adapter:
        model = load_inference_lora(model, args.adapter)
    model.eval()
    device = model_input_device(model)
    mode = "a" if args.resume else "w"
    started = time.monotonic()
    with predictions_path.open(mode, encoding="utf-8") as output:
        for start in tqdm(
            range(len(predictions), len(source_rows), args.batch_size),
            desc="hallusionbench",
        ):
            batch = source_rows[start : start + args.batch_size]
            images = [
                row["image"].convert("RGB")
                if row.get("image") is not None
                else Image.new("RGB", (384, 384), "white")
                for row in batch
            ]
            prompts = [
                format_prompt(
                    args.model_kind,
                    f"{str(row['question']).strip()}\nAnswer using only yes or no.",
                    "default",
                )
                for row in batch
            ]
            inputs = loaded.processor(
                images=images, text=prompts, padding=True, return_tensors="pt"
            )
            for key, value in list(inputs.items()):
                if isinstance(value, torch.Tensor):
                    value = value.to(device)
                    if key == "pixel_values":
                        value = value.to(dtype=torch.bfloat16)
                    inputs[key] = value
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                )
            answers = loaded.processor.batch_decode(generated, skip_special_tokens=True)
            for source, answer in zip(batch, answers):
                row = serializable_row(source)
                row["prediction"] = answer.strip()
                predictions.append(row)
                output.write(json.dumps(row) + "\n")
            output.flush()

    metrics = compute_hallusionbench_metrics(predictions)
    metrics.pop("evaluated_rows")
    payload = {
        "benchmark": "HallusionBench strict yes/no evaluation",
        "text_only_convention": "white image supplied when visual_input=0",
        "metrics": metrics,
        "elapsed_seconds": time.monotonic() - started,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
