from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

from grounded_vqa.cli.train_lora import resolve_data_files
from grounded_vqa.data.selection import select_samples
from grounded_vqa.data.vqav2 import load_vqav2
from grounded_vqa.evaluation.vqa import EvaluationExample, evaluate_examples
from grounded_vqa.models.loading import format_prompt, load_model, model_input_device
from grounded_vqa.models.lora import load_inference_lora
from grounded_vqa.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and score VQAv2 predictions")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-kind", choices=["blip2", "instructblip"], required=True)
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="4bit")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-selection-size",
        type=int,
        default=0,
        help="Exclude the deterministic subset used by an earlier run",
    )
    parser.add_argument("--exclude-selection-seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--prompt-style", choices=["default", "short"], default="default")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a prefix already written to predictions.jsonl",
    )
    parser.add_argument("--output-name", required=True)
    return parser.parse_args()


def load_resume_rows(
    details_path: Path,
    expected_question_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Load a completed prediction prefix and reject incompatible output."""
    if not details_path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with details_path.open(encoding="utf-8") as detail_file:
        for line_number, line in enumerate(detail_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {details_path} at line {line_number}"
                ) from error
            if not isinstance(row, dict) or "question_id" not in row:
                raise ValueError(
                    f"Invalid prediction row in {details_path} at line {line_number}"
                )
            rows.append(row)

    if len(rows) > len(expected_question_ids):
        raise ValueError(
            f"Resume output has {len(rows)} rows, but only "
            f"{len(expected_question_ids)} samples were selected"
        )
    completed_ids = [int(row["question_id"]) for row in rows]
    expected_prefix = list(expected_question_ids[: len(rows)])
    if completed_ids != expected_prefix:
        raise ValueError(
            "Resume output is not an exact prefix of the current deterministic selection"
        )
    return rows


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
    excluded_question_ids: set[int] = set()
    if args.exclude_selection_size:
        excluded = select_samples(
            samples,
            args.exclude_selection_size,
            args.exclude_selection_seed,
        )
        excluded_question_ids = {sample.question_id for sample in excluded}
        samples = [
            sample for sample in samples if sample.question_id not in excluded_question_ids
        ]
    samples = select_samples(samples, args.max_samples, args.seed)
    (output_dir / "selection_audit.json").write_text(
        json.dumps(
            {
                "selected_count": len(samples),
                "selected_question_ids": [sample.question_id for sample in samples],
                "excluded_count": len(excluded_question_ids),
                "exclude_selection_size": args.exclude_selection_size,
                "exclude_selection_seed": args.exclude_selection_seed,
                "overlap_count": sum(
                    sample.question_id in excluded_question_ids for sample in samples
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    details_path = output_dir / "predictions.jsonl"
    official_predictions: list[dict[str, object]] = []
    evaluation_examples: list[EvaluationExample] = []
    resume_rows = (
        load_resume_rows(details_path, [sample.question_id for sample in samples])
        if args.resume
        else []
    )
    for row in resume_rows:
        official_predictions.append(
            {"question_id": int(row["question_id"]), "answer": str(row["prediction"])}
        )
        evaluation_examples.append(
            EvaluationExample(
                prediction=str(row["prediction"]),
                references=list(row["references"]),
                answer_type=str(row.get("answer_type") or "unknown"),
                question_type=str(row.get("question_type") or "unknown"),
            )
        )
    if resume_rows:
        print(f"Resuming after {len(resume_rows)} completed predictions")

    started = time.monotonic()
    file_mode = "a" if args.resume else "w"
    remaining_samples = samples[len(resume_rows) :]
    with details_path.open(file_mode, encoding="utf-8") as detail_file:
        for sample in tqdm(
            remaining_samples,
            desc="predict",
            initial=len(resume_rows),
            total=len(samples),
        ):
            with Image.open(sample.image_path) as image:
                image = image.convert("RGB")
            prompt = format_prompt(args.model_kind, sample.question, args.prompt_style)
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
            official_predictions.append(
                {"question_id": sample.question_id, "answer": prediction}
            )
            evaluation_examples.append(
                EvaluationExample(
                    prediction=prediction,
                    references=sample.answers,
                    answer_type=sample.answer_type or "unknown",
                    question_type=sample.question_type or "unknown",
                )
            )
            detail_file.write(
                json.dumps(
                    {
                        "question_id": sample.question_id,
                        "image_id": sample.image_id,
                        "question": sample.question,
                        "prediction": prediction,
                        "references": sample.answers,
                        "answer_type": sample.answer_type,
                        "question_type": sample.question_type,
                    }
                )
                + "\n"
            )
            detail_file.flush()

    metrics = evaluate_examples(evaluation_examples)
    metrics["elapsed_seconds"] = time.monotonic() - started
    metrics["peak_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
    (output_dir / "official_predictions.json").write_text(
        json.dumps(official_predictions) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
