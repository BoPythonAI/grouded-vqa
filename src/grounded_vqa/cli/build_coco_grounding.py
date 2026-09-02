from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from grounded_vqa.data.coco_grounding import generate_grounding_records, grounding_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build audited COCO object-grounding VQA data")
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--positive-count", type=int, required=True)
    parser.add_argument("--negative-count", type=int, required=True)
    parser.add_argument("--counting-count", type=int, required=True)
    parser.add_argument("--min-area-ratio", type=float, default=0.001)
    parser.add_argument("--max-count-answer", type=int, default=10)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.min_area_ratio < 1:
        raise ValueError("min-area-ratio must be in [0, 1)")

    payload = json.loads(args.instances.read_text(encoding="utf-8"))
    records = generate_grounding_records(
        payload,
        split=args.split,
        positive_count=args.positive_count,
        negative_count=args.negative_count,
        counting_count=args.counting_count,
        min_area_ratio=args.min_area_ratio,
        max_count_answer=args.max_count_answer,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    source_sha256 = hashlib.sha256(args.instances.read_bytes()).hexdigest()
    summary = {
        "arguments": {**vars(args), "instances": str(args.instances), "output": str(args.output)},
        "source_sha256": source_sha256,
        **grounding_summary(records),
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
