from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from grounded_vqa.cli.train_complementary import sequence_token_nll
from grounded_vqa.data.answer_routing import (
    choose_yes_no,
    is_valid_yes_no_answer,
    is_yes_no_question,
)
from grounded_vqa.data.collator import VQAGenerativeCollator, move_batch_to_device
from grounded_vqa.data.vqav2 import VQASample, coco_image_path
from grounded_vqa.evaluation.vqa import EvaluationExample, evaluate_examples
from grounded_vqa.models.loading import load_model, model_input_device
from grounded_vqa.models.lora import load_inference_lora
from grounded_vqa.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conservatively rerank invalid short VQA answers"
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-kind", choices=["blip2", "instructblip"], required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="4bit")
    parser.add_argument("--batch-size", type=int, default=8, help="Questions per batch")
    parser.add_argument("--prompt-style", choices=["default", "short"], default="short")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-name", required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"question_id", "image_id", "question", "prediction", "references"}
            if not isinstance(row, dict) or not required.issubset(row):
                raise ValueError(f"Invalid row at {path}:{line_number}")
            rows.append(row)
    return rows


def eligible_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if is_yes_no_question(str(row["question"]))
        and not is_valid_yes_no_answer(str(row["prediction"]))
    ]


def load_resume_scores(
    path: Path, expected_question_ids: list[int]
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    scores = load_rows(path)
    completed = [int(row["question_id"]) for row in scores]
    if completed != expected_question_ids[: len(completed)]:
        raise ValueError("Rerank resume file is not an exact eligible-row prefix")
    return scores


def make_candidate_samples(
    rows: list[dict[str, Any]], data_root: Path, split: str
) -> list[VQASample]:
    samples: list[VQASample] = []
    for row in rows:
        base = VQASample(
            question_id=int(row["question_id"]),
            image_id=int(row["image_id"]),
            image_path=coco_image_path(data_root, split, int(row["image_id"])),
            question=str(row["question"]),
            answers=("yes",),
            answer_type=None,
            question_type=None,
        )
        samples.extend((base, replace(base, answers=("no",))))
    return samples


def apply_reranking(
    rows: list[dict[str, Any]], score_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    score_by_id = {int(row["question_id"]): row for row in score_rows}
    updated: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        score = score_by_id.get(int(row["question_id"]))
        if score is not None:
            row["original_prediction"] = row["prediction"]
            row["prediction"] = score["reranked_prediction"]
            row["rerank_yes_nll"] = score["yes_nll"]
            row["rerank_no_nll"] = score["no_nll"]
        updated.append(row)
    return updated


def evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    examples = [
        EvaluationExample(
            prediction=str(row["prediction"]),
            references=list(row["references"]),
            answer_type=str(row.get("answer_type") or "unknown"),
            question_type=str(row.get("question_type") or "unknown"),
        )
        for row in rows
    ]
    return evaluate_examples(examples)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    paths = ProjectPaths.from_environment()
    paths.ensure()
    output_dir = paths.output_root / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.predictions)
    eligible = eligible_rows(rows)
    eligible_ids = [int(row["question_id"]) for row in eligible]
    score_path = output_dir / "rerank_scores.jsonl"
    score_rows = load_resume_scores(score_path, eligible_ids) if args.resume else []

    loaded = load_model(args.model_id, args.model_kind, args.quantization)
    model = load_inference_lora(loaded.model, args.adapter)
    model.eval()
    device = model_input_device(model)
    collator = VQAGenerativeCollator(
        loaded.processor,
        args.model_kind,
        prompt_style=args.prompt_style,
    )

    started = time.monotonic()
    file_mode = "a" if args.resume else "w"
    with score_path.open(file_mode, encoding="utf-8") as score_file:
        for start in tqdm(
            range(len(score_rows), len(eligible), args.batch_size),
            desc="rerank-invalid-yes-no",
        ):
            batch_rows = eligible[start : start + args.batch_size]
            samples = make_candidate_samples(batch_rows, paths.data_root, args.split)
            batch = collator(samples)
            batch, _ = move_batch_to_device(batch, device)
            with torch.inference_mode():
                nll = sequence_token_nll(
                    model(**batch).logits,
                    batch["labels"],
                ).detach().cpu()
            for index, row in enumerate(batch_rows):
                yes_nll = float(nll[2 * index])
                no_nll = float(nll[2 * index + 1])
                score = {
                    "question_id": int(row["question_id"]),
                    "image_id": int(row["image_id"]),
                    "question": str(row["question"]),
                    "prediction": str(row["prediction"]),
                    "references": list(row["references"]),
                    "answer_type": row.get("answer_type"),
                    "question_type": row.get("question_type"),
                    "yes_nll": yes_nll,
                    "no_nll": no_nll,
                    "reranked_prediction": choose_yes_no(yes_nll, no_nll),
                }
                score_rows.append(score)
                score_file.write(json.dumps(score) + "\n")
            score_file.flush()

    updated = apply_reranking(rows, score_rows)
    metrics = evaluate_rows(updated)
    baseline_metrics = evaluate_rows(rows)
    metrics["baseline_overall"] = baseline_metrics["overall"]
    metrics["overall_delta"] = metrics["overall"] - baseline_metrics["overall"]
    metrics["eligible_reranked"] = len(score_rows)
    metrics["router_total"] = sum(
        is_yes_no_question(str(row["question"])) for row in rows
    )
    metrics["elapsed_seconds"] = time.monotonic() - started
    metrics["peak_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30

    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in updated:
            handle.write(json.dumps(row) + "\n")
    (output_dir / "official_predictions.json").write_text(
        json.dumps(
            [
                {"question_id": int(row["question_id"]), "answer": row["prediction"]}
                for row in updated
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
