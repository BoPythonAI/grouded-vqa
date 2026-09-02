from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

from grounded_vqa.evaluation.chair import build_ground_truth_objects, compute_chair_metrics
from grounded_vqa.models.loading import format_prompt, load_model, model_input_device
from grounded_vqa.models.lora import load_inference_lora
from grounded_vqa.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate free-form COCO captions with CHAIR")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-kind", choices=["blip2", "instructblip"], required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--annotation-root", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="4bit")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-name", required=True)
    return parser.parse_args()


def select_images(
    annotation_root: Path,
    image_root: Path,
    selection_file: Path,
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    instances = json.loads(
        (annotation_root / "instances_val2014.json").read_text(encoding="utf-8")
    )
    by_id = {
        int(image["id"]): {
            "image_id": int(image["id"]),
            "file_name": str(image["file_name"]),
        }
        for image in instances["images"]
        if (image_root / str(image["file_name"])).is_file()
    }
    if selection_file.is_file():
        selected_ids = [int(value) for value in json.loads(selection_file.read_text())]
        if len(selected_ids) != sample_size:
            raise ValueError(
                f"Selection has {len(selected_ids)} images, expected {sample_size}: {selection_file}"
            )
    else:
        if sample_size <= 0 or sample_size > len(by_id):
            raise ValueError(f"sample-size must be in [1, {len(by_id)}]")
        selected_ids = random.Random(seed).sample(sorted(by_id), sample_size)
        selection_file.parent.mkdir(parents=True, exist_ok=True)
        selection_file.write_text(json.dumps(selected_ids, indent=2) + "\n", encoding="utf-8")
    missing = [image_id for image_id in selected_ids if image_id not in by_id]
    if missing:
        raise FileNotFoundError(f"Selected COCO images are missing: {missing[:5]}")
    return [by_id[image_id] for image_id in selected_ids]


def load_resume_rows(path: Path, selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    expected = [int(row["image_id"]) for row in selected[: len(rows)]]
    actual = [int(row["image_id"]) for row in rows]
    if actual != expected:
        raise ValueError("CHAIR resume file is not an exact prefix of the selection")
    return rows


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("batch-size and max-new-tokens must be positive")
    paths = ProjectPaths.from_environment()
    paths.ensure()
    project_root = paths.data_root.parent.parent
    annotation_root = args.annotation_root or project_root / "data" / "coco" / "annotations"
    image_root = args.image_root or paths.data_root / "val2014"
    selection_file = args.selection_file or paths.output_root / "H2_CHAIR_selection_seed42.json"
    output_dir = paths.output_root / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = select_images(
        annotation_root, image_root, selection_file, args.sample_size, args.seed
    )
    predictions_path = output_dir / "captions.jsonl"
    rows = load_resume_rows(predictions_path, selected) if args.resume else []

    loaded = load_model(args.model_id, args.model_kind, args.quantization)
    model = loaded.model
    if args.adapter:
        model = load_inference_lora(model, args.adapter)
    model.eval()
    device = model_input_device(model)
    prompt = format_prompt(args.model_kind, "Describe this image in detail.", "default")
    mode = "a" if args.resume else "w"
    started = time.monotonic()
    with predictions_path.open(mode, encoding="utf-8") as output:
        for start in tqdm(
            range(len(rows), len(selected), args.batch_size), desc="chair-captions"
        ):
            batch_rows = selected[start : start + args.batch_size]
            images: list[Image.Image] = []
            for row in batch_rows:
                with Image.open(image_root / str(row["file_name"])) as image:
                    images.append(image.convert("RGB"))
            inputs = loaded.processor(
                images=images,
                text=[prompt] * len(images),
                padding=True,
                return_tensors="pt",
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
            captions = loaded.processor.batch_decode(generated, skip_special_tokens=True)
            for source, caption in zip(batch_rows, captions):
                row = {
                    "image_id": int(source["image_id"]),
                    "file_name": str(source["file_name"]),
                    "caption": caption.strip(),
                }
                rows.append(row)
                output.write(json.dumps(row) + "\n")
            output.flush()

    ground_truth = build_ground_truth_objects(
        [int(row["image_id"]) for row in rows], annotation_root
    )
    metrics = compute_chair_metrics(rows, ground_truth)
    per_caption = metrics.pop("per_caption")
    (output_dir / "chair_details.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in per_caption), encoding="utf-8"
    )
    payload = {
        "benchmark": "CHAIR on deterministic COCO val2014 sample",
        "selection_file": str(selection_file),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "prompt": "Describe this image in detail.",
        "max_new_tokens": args.max_new_tokens,
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
