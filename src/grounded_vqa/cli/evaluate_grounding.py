from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from grounded_vqa.data.answers import normalize_answer
from grounded_vqa.models.loading import format_prompt, load_model, model_input_device
from grounded_vqa.models.lora import load_inference_lora
from grounded_vqa.paths import ProjectPaths


def compute_grounding_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    by_task: dict[str, list[bool]] = defaultdict(list)
    negative_predictions: list[str] = []
    positive_predictions: list[str] = []
    count_errors: list[float] = []
    for row in rows:
        prediction = normalize_answer(str(row["prediction"]))
        answer = normalize_answer(str(row["answer"]))
        task = str(row["task_type"])
        by_task[task].append(prediction == answer)
        if task == "existence_negative":
            negative_predictions.append(prediction)
        elif task == "existence_positive":
            positive_predictions.append(prediction)
        elif task == "count":
            try:
                count_errors.append(abs(float(prediction) - float(answer)))
            except ValueError:
                count_errors.append(float("nan"))
    valid_count_errors = [value for value in count_errors if not torch.isnan(torch.tensor(value))]
    return {
        "count": len(rows),
        "overall_exact_accuracy": 100.0
        * sum(value for values in by_task.values() for value in values)
        / len(rows),
        "by_task": {
            task: {"count": len(values), "accuracy": 100.0 * sum(values) / len(values)}
            for task, values in by_task.items()
        },
        "false_yes_rate": 100.0
        * sum(prediction == "yes" for prediction in negative_predictions)
        / len(negative_predictions)
        if negative_predictions
        else None,
        "false_no_rate": 100.0
        * sum(prediction == "no" for prediction in positive_predictions)
        / len(positive_predictions)
        if positive_predictions
        else None,
        "invalid_yes_no_rate": 100.0
        * (
            sum(prediction not in {"yes", "no"} for prediction in negative_predictions)
            + sum(prediction not in {"yes", "no"} for prediction in positive_predictions)
        )
        / (len(negative_predictions) + len(positive_predictions))
        if negative_predictions or positive_predictions
        else None,
        "count_mae": (
            sum(valid_count_errors) / len(valid_count_errors) if valid_count_errors else None
        ),
        "count_valid_number_rate": (
            100.0 * len(valid_count_errors) / len(count_errors) if count_errors else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate COCO grounding VQA data")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-kind", choices=["blip2", "instructblip"], required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="4bit")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--prompt-style", choices=["default", "short"], default="short")
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    paths = ProjectPaths.from_environment()
    paths.ensure()
    output_dir = paths.output_root / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    loaded = load_model(args.model_id, args.model_kind, args.quantization)
    model = loaded.model
    if args.adapter:
        model = load_inference_lora(model, args.adapter)
    model.eval()
    device = model_input_device(model)

    rows: list[dict[str, object]] = []
    started = time.monotonic()
    for start in tqdm(range(0, len(records), args.batch_size), desc="grounding-eval"):
        batch = records[start : start + args.batch_size]
        images: list[Image.Image] = []
        prompts: list[str] = []
        for item in batch:
            split = str(item["split"])
            year_split = {"train": "train2014", "val": "val2014"}[split]
            image_path = (
                paths.data_root
                / year_split
                / f"COCO_{year_split}_{int(item['image_id']):012d}.jpg"
            )
            with Image.open(image_path) as image:
                images.append(image.convert("RGB"))
            prompts.append(
                format_prompt(args.model_kind, str(item["question"]), args.prompt_style)
            )
        inputs = loaded.processor(
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
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
            )
        predictions = loaded.processor.batch_decode(generated, skip_special_tokens=True)
        for item, prediction in zip(batch, predictions, strict=True):
            rows.append({**item, "prediction": prediction.strip()})

    metrics = {
        **compute_grounding_metrics(rows),
        "elapsed_seconds": time.monotonic() - started,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
