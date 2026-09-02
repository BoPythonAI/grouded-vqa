from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from grounded_vqa.evaluation.pope import parse_strict_yes_no


def row_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(row[key])
        for key in ("category", "subcategory", "set_id", "figure_id", "question_id")
    )


def pair_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(row[key]) for key in ("category", "subcategory", "set_id", "question_id")
    )


def figure_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row[key]) for key in ("category", "subcategory", "set_id", "figure_id"))


def _group_accuracy(
    rows: list[dict[str, Any]], key_fn: Any, include: Any | None = None
) -> dict[str, Any]:
    groups: dict[tuple[str, ...], list[bool]] = defaultdict(list)
    for row in rows:
        if include is None or include(row):
            groups[key_fn(row)].append(bool(row["correct"]))
    correct = sum(all(values) for values in groups.values())
    total = len(groups)
    return {"correct": correct, "total": total, "accuracy": correct / total if total else None}


def compute_hallusionbench_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    evaluated: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for source in rows:
        row = dict(source)
        key = row_key(row)
        if key in seen:
            raise ValueError(f"Duplicate HallusionBench row: {key}")
        seen.add(key)
        prediction = parse_strict_yes_no(str(row.get("prediction", "")))
        label = "yes" if str(row["gt_answer"]) == "1" else "no"
        row["parsed_prediction"] = prediction
        row["label"] = label
        row["correct"] = prediction == label
        evaluated.append(row)
    if not evaluated:
        raise ValueError("At least one HallusionBench prediction is required")

    total = len(evaluated)
    correct = sum(bool(row["correct"]) for row in evaluated)
    valid = [row for row in evaluated if row["parsed_prediction"] in {"yes", "no"}]
    positives = sum(row["label"] == "yes" for row in evaluated)
    negatives = total - positives
    false_positive = sum(
        row["label"] == "no" and row["parsed_prediction"] == "yes" for row in evaluated
    )
    false_negative = sum(
        row["label"] == "yes" and row["parsed_prediction"] == "no" for row in evaluated
    )

    by_category: dict[str, Any] = {}
    for category in sorted({str(row["category"]) for row in evaluated}):
        subset = [row for row in evaluated if str(row["category"]) == category]
        subset_correct = sum(bool(row["correct"]) for row in subset)
        by_category[category] = {
            "correct": subset_correct,
            "total": len(subset),
            "accuracy": subset_correct / len(subset),
        }

    return {
        "question": {"correct": correct, "total": total, "accuracy": correct / total},
        "question_pair": _group_accuracy(evaluated, pair_key),
        "easy_question_pair": _group_accuracy(
            evaluated, pair_key, lambda row: str(row["visual_input"]) != "2"
        ),
        "hard_question_pair": _group_accuracy(
            evaluated, pair_key, lambda row: str(row["visual_input"]) == "2"
        ),
        "figure": _group_accuracy(
            evaluated,
            figure_key,
            lambda row: not (
                str(row["category"]) == "VS" and str(row["figure_id"]) == "0"
            ),
        ),
        "by_category": by_category,
        "yes_ratio": sum(row["parsed_prediction"] == "yes" for row in valid) / len(valid)
        if valid
        else None,
        "invalid_ratio": (total - len(valid)) / total,
        "false_positive_rate": false_positive / negatives if negatives else None,
        "false_negative_rate": false_negative / positives if positives else None,
        "evaluated_rows": evaluated,
    }
