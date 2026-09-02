from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from grounded_vqa.evaluation.chair import paired_chair_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired comparison of two CHAIR outputs")
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
    payload = paired_chair_comparison(
        load_rows(args.baseline / "chair_details.jsonl"),
        load_rows(args.candidate / "chair_details.jsonl"),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
