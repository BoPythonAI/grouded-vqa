from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from grounded_vqa.data.coco_grounding import grounding_summary
from grounded_vqa.data.error_grounding import (
    generate_error_grounding_records,
    nested_task_subset,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build E6-error-driven grounding data")
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--positive-count", type=int, required=True)
    parser.add_argument("--negative-count", type=int, required=True)
    parser.add_argument("--counting-count", type=int, required=True)
    parser.add_argument("--min-area-ratio", type=float, default=0.001)
    parser.add_argument("--max-count-answer", type=int, default=10)
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subset-output", type=Path)
    parser.add_argument("--subset-positive-count", type=int, default=0)
    parser.add_argument("--subset-negative-count", type=int, default=0)
    parser.add_argument("--subset-counting-count", type=int, default=0)
    args = parser.parse_args()

    instances = json.loads(args.instances.read_text(encoding="utf-8"))
    predictions = read_jsonl(args.predictions)
    records, audit = generate_error_grounding_records(
        instances,
        predictions,
        positive_count=args.positive_count,
        negative_count=args.negative_count,
        counting_count=args.counting_count,
        min_area_ratio=args.min_area_ratio,
        max_count_answer=args.max_count_answer,
        seed=args.seed,
    )
    write_jsonl(args.output, records)
    summary: dict[str, Any] = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "instances_sha256": hashlib.sha256(args.instances.read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        **audit,
        "grounding_summary": grounding_summary(records),
    }

    if args.subset_output:
        subset = nested_task_subset(
            records,
            positive_count=args.subset_positive_count,
            negative_count=args.subset_negative_count,
            counting_count=args.subset_counting_count,
        )
        write_jsonl(args.subset_output, subset)
        summary["nested_subset"] = {
            "path": str(args.subset_output),
            "is_strict_subset": {item["question_id"] for item in subset}.issubset(
                item["question_id"] for item in records
            ),
            **grounding_summary(subset),
        }

    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
