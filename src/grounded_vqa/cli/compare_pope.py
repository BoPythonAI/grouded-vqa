from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from grounded_vqa.evaluation.pope import paired_pope_comparison

STRATEGIES = ("random", "popular", "adversarial")
DECODERS = ("generated_official_prediction", "pair_logprob_prediction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired comparison of two POPE outputs")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--decoder", choices=DECODERS, default=DECODERS[0])
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> None:
    args = parse_args()
    results: dict[str, Any] = {}
    combined_labels: list[str] = []
    combined_baseline: list[str] = []
    combined_candidate: list[str] = []
    for strategy in STRATEGIES:
        baseline_rows = load_rows(args.baseline / f"predictions_{strategy}.jsonl")
        candidate_rows = load_rows(args.candidate / f"predictions_{strategy}.jsonl")
        baseline_ids = [int(row["question_id"]) for row in baseline_rows]
        candidate_ids = [int(row["question_id"]) for row in candidate_rows]
        if baseline_ids != candidate_ids:
            raise ValueError(f"POPE question IDs do not align for {strategy}")
        labels = [str(row["label"]) for row in baseline_rows]
        if labels != [str(row["label"]) for row in candidate_rows]:
            raise ValueError(f"POPE labels do not align for {strategy}")
        baseline_predictions = [str(row[args.decoder]) for row in baseline_rows]
        candidate_predictions = [str(row[args.decoder]) for row in candidate_rows]
        results[strategy] = paired_pope_comparison(
            labels,
            baseline_predictions,
            candidate_predictions,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        combined_labels.extend(labels)
        combined_baseline.extend(baseline_predictions)
        combined_candidate.extend(candidate_predictions)
    results["combined"] = paired_pope_comparison(
        combined_labels,
        combined_baseline,
        combined_candidate,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    payload = {"decoder": args.decoder, "strategies": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
