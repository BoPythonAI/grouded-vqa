from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

from grounded_vqa.cli.predict_vqav2 import load_resume_rows
from grounded_vqa.data.multiple_choice import (
    OPTION_LETTERS,
    MultipleChoiceExample,
    format_multiple_choice_prompt,
    parse_multiple_choice_prediction,
)
from grounded_vqa.models.loading import load_model, model_input_device
from grounded_vqa.models.lora import load_inference_lora
from grounded_vqa.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a VQAv2-derived MCQ benchmark")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-kind", choices=["blip2", "instructblip"], required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="4bit")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-name", required=True)
    return parser.parse_args()


def load_examples(path: Path) -> list[MultipleChoiceExample]:
    examples: list[MultipleChoiceExample] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append(
                MultipleChoiceExample(
                    question_id=int(row["question_id"]),
                    image_id=int(row["image_id"]),
                    image_path=str(row["image_path"]),
                    question=str(row["question"]),
                    options=tuple(str(option) for option in row["options"]),
                    correct_index=int(row["correct_index"]),
                    correct_answer=str(row["correct_answer"]),
                    answer_type=str(row["answer_type"]),
                    question_type=str(row["question_type"]),
                )
            )
    if not examples:
        raise ValueError(f"No MCQ examples found in {path}")
    return examples


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, object]:
    by_answer_type: dict[str, list[bool]] = defaultdict(list)
    by_option_count: dict[int, list[bool]] = defaultdict(list)
    predicted_positions: Counter[str] = Counter()
    correct = 0
    invalid = 0
    random_sum = 0.0
    for row in rows:
        prediction = row.get("predicted_index")
        is_correct = prediction == row["correct_index"]
        correct += int(is_correct)
        invalid += int(prediction is None)
        by_answer_type[str(row["answer_type"])].append(is_correct)
        option_count = len(row["options"])
        by_option_count[option_count].append(is_correct)
        random_sum += 1.0 / option_count
        if prediction is not None:
            predicted_positions[OPTION_LETTERS[int(prediction)]] += 1

    count = len(rows)
    accuracy = lambda values: 100.0 * sum(values) / len(values)
    return {
        "count": count,
        "accuracy": 100.0 * correct / count,
        "random_baseline": 100.0 * random_sum / count,
        "invalid_output_rate": 100.0 * invalid / count,
        "by_answer_type": {
            key: {"accuracy": accuracy(values), "count": len(values)}
            for key, values in sorted(by_answer_type.items())
        },
        "by_option_count": {
            str(key): {"accuracy": accuracy(values), "count": len(values)}
            for key, values in sorted(by_option_count.items())
        },
        "predicted_position_distribution": dict(sorted(predicted_positions.items())),
    }


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_environment()
    output_dir = paths.output_root / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = load_examples(args.dataset)
    details_path = output_dir / "predictions.jsonl"
    resume_rows = (
        load_resume_rows(details_path, [example.question_id for example in examples])
        if args.resume
        else []
    )

    loaded = load_model(args.model_id, args.model_kind, args.quantization)
    model = loaded.model
    if args.adapter:
        model = load_inference_lora(model, args.adapter)
    model.eval()
    device = model_input_device(model)
    if resume_rows:
        print(f"Resuming after {len(resume_rows)} completed MCQ predictions")

    started = time.monotonic()
    file_mode = "a" if args.resume else "w"
    with details_path.open(file_mode, encoding="utf-8") as output:
        for example in tqdm(
            examples[len(resume_rows) :],
            desc="mcq-eval",
            initial=len(resume_rows),
            total=len(examples),
        ):
            with Image.open(example.image_path) as image:
                image = image.convert("RGB")
            prompt = format_multiple_choice_prompt(example)
            inputs = loaded.processor(images=image, text=prompt, return_tensors="pt")
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
            prediction = loaded.processor.batch_decode(
                generated,
                skip_special_tokens=True,
            )[0].strip()
            predicted_index = parse_multiple_choice_prediction(prediction, example.options)
            row = example.to_dict()
            row.update(
                {
                    "prompt": prompt,
                    "prediction": prediction,
                    "predicted_index": predicted_index,
                    "correct": predicted_index == example.correct_index,
                }
            )
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()

    rows = load_resume_rows(details_path, [example.question_id for example in examples])
    metrics = compute_metrics(rows)
    metrics["elapsed_seconds"] = time.monotonic() - started
    metrics["peak_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
    metrics["dataset"] = str(args.dataset)
    metrics["model_kind"] = args.model_kind
    metrics["adapter"] = str(args.adapter) if args.adapter else None
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
