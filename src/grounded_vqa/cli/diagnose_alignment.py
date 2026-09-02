from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from grounded_vqa.cli.train_lora import resolve_data_files
from grounded_vqa.data.answers import normalize_answer, vqa_soft_accuracy
from grounded_vqa.data.selection import build_mismatched_indices, select_samples
from grounded_vqa.data.vqav2 import VQASample, load_vqav2
from grounded_vqa.evaluation.vqa import EvaluationExample, evaluate_examples
from grounded_vqa.models.loading import format_prompt, load_model, model_input_device
from grounded_vqa.models.lora import load_inference_lora
from grounded_vqa.paths import ProjectPaths

CONDITIONS = ("normal", "mismatched", "blank", "noise")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose visual dependence on VQAv2")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-kind", choices=["blip2", "instructblip"], required=True)
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="4bit")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--prompt-style", choices=["default", "short"], default="default")
    parser.add_argument("--output-name", required=True)
    return parser.parse_args()


def make_noise_image(size: tuple[int, int], seed: int) -> Image.Image:
    width, height = size
    generator = np.random.default_rng(seed)
    pixels = generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def dependence_summary(rows: list[dict[str, object]], condition: str) -> dict[str, float]:
    if condition == "normal":
        raise ValueError("A perturbed condition is required")
    count = len(rows)
    if count == 0:
        raise ValueError("No diagnostic rows supplied")

    normal_scores = [float(row["scores"]["normal"]) for row in rows]  # type: ignore[index]
    perturbed_scores = [float(row["scores"][condition]) for row in rows]  # type: ignore[index]
    changed = [
        normalize_answer(str(row["predictions"]["normal"]))  # type: ignore[index]
        != normalize_answer(str(row["predictions"][condition]))  # type: ignore[index]
        for row in rows
    ]
    percent = lambda value: 100.0 * value / count
    return {
        "count": count,
        "answer_change_rate": percent(sum(changed)),
        "unchanged_answer_rate": percent(count - sum(changed)),
        "normal_advantage_rate": percent(
            sum(normal > perturbed for normal, perturbed in zip(normal_scores, perturbed_scores))
        ),
        "perturbed_advantage_rate": percent(
            sum(normal < perturbed for normal, perturbed in zip(normal_scores, perturbed_scores))
        ),
        "equal_score_rate": percent(
            sum(normal == perturbed for normal, perturbed in zip(normal_scores, perturbed_scores))
        ),
        "accuracy_drop_points": 100.0
        * sum(normal - perturbed for normal, perturbed in zip(normal_scores, perturbed_scores))
        / count,
    }


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def make_condition_images(
    sample: VQASample,
    mismatched_sample: VQASample,
    noise_seed: int,
) -> list[Image.Image]:
    normal = load_rgb(sample.image_path)
    mismatched = load_rgb(mismatched_sample.image_path)
    blank = Image.new("RGB", normal.size, color=(127, 127, 127))
    noise = make_noise_image(normal.size, noise_seed)
    return [normal, mismatched, blank, noise]


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_environment()
    paths.ensure()
    output_dir = paths.output_root / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_model(args.model_id, args.model_kind, args.quantization)
    model = loaded.model
    if args.adapter:
        model = load_inference_lora(model, args.adapter)
    model.eval()
    device = model_input_device(model)

    questions, annotations, images_root = resolve_data_files(paths.data_root, args.split)
    samples = load_vqav2(questions, annotations, images_root, args.split)
    samples = select_samples(samples, args.max_samples, args.seed)
    mismatch_indices = build_mismatched_indices(
        [sample.image_id for sample in samples],
        args.seed,
    )

    run_config = {
        **vars(args),
        "adapter": str(args.adapter) if args.adapter else None,
        "conditions": CONDITIONS,
        "blank_rgb": [127, 127, 127],
        "mismatch_policy": "seeded different-image permutation",
        "noise_policy": "seeded independent uniform RGB noise",
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n",
        encoding="utf-8",
    )

    examples: dict[str, list[EvaluationExample]] = {condition: [] for condition in CONDITIONS}
    rows: list[dict[str, object]] = []
    started = time.monotonic()
    details_path = output_dir / "diagnostics.jsonl"
    with details_path.open("w", encoding="utf-8") as detail_file:
        for index, sample in enumerate(tqdm(samples, desc="alignment-diagnostic")):
            mismatch = samples[mismatch_indices[index]]
            images = make_condition_images(
                sample,
                mismatch,
                (args.seed * 1_000_003 + sample.question_id) % (2**32),
            )
            prompt = format_prompt(args.model_kind, sample.question, args.prompt_style)
            inputs = loaded.processor(
                images=images,
                text=[prompt] * len(CONDITIONS),
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
            decoded = loaded.processor.batch_decode(generated, skip_special_tokens=True)
            predictions = {
                condition: answer.strip()
                for condition, answer in zip(CONDITIONS, decoded, strict=True)
            }
            scores = {
                condition: vqa_soft_accuracy(prediction, sample.answers)
                for condition, prediction in predictions.items()
            }
            for condition, prediction in predictions.items():
                examples[condition].append(
                    EvaluationExample(
                        prediction=prediction,
                        references=sample.answers,
                        answer_type=sample.answer_type or "unknown",
                        question_type=sample.question_type or "unknown",
                    )
                )
            row: dict[str, object] = {
                "question_id": sample.question_id,
                "image_id": sample.image_id,
                "mismatched_image_id": mismatch.image_id,
                "question": sample.question,
                "references": sample.answers,
                "answer_type": sample.answer_type,
                "question_type": sample.question_type,
                "predictions": predictions,
                "scores": scores,
            }
            rows.append(row)
            detail_file.write(json.dumps(row) + "\n")
            detail_file.flush()

    condition_metrics = {
        condition: evaluate_examples(condition_examples)
        for condition, condition_examples in examples.items()
    }
    metrics = {
        "count": len(rows),
        "conditions": condition_metrics,
        "visual_dependence": {
            condition: dependence_summary(rows, condition)
            for condition in CONDITIONS
            if condition != "normal"
        },
        "elapsed_seconds": time.monotonic() - started,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
