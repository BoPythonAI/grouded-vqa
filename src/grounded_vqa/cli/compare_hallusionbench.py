from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from grounded_vqa.evaluation.hallusionbench import row_key
from grounded_vqa.evaluation.pope import paired_pope_comparison, parse_strict_yes_no


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired HallusionBench comparison")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    args = parse_args()
    baseline = load_rows(args.baseline / "predictions.jsonl")
    candidate = load_rows(args.candidate / "predictions.jsonl")
    if [row_key(row) for row in baseline] != [row_key(row) for row in candidate]:
        raise ValueError("HallusionBench rows do not align")
    labels = ["yes" if str(row["gt_answer"]) == "1" else "no" for row in baseline]
    baseline_predictions = [parse_strict_yes_no(str(row["prediction"])) for row in baseline]
    candidate_predictions = [parse_strict_yes_no(str(row["prediction"])) for row in candidate]
    payload = paired_pope_comparison(
        labels,
        baseline_predictions,
        candidate_predictions,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
