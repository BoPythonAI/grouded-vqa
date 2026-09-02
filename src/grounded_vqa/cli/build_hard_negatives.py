from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from grounded_vqa.data.coco_grounding import (
    generate_hard_negative_records,
    grounding_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build COCO co-occurrence hard negatives")
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--max-per-category", type=int, default=30)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    instances = json.loads(args.instances.read_text(encoding="utf-8"))
    captions = json.loads(args.captions.read_text(encoding="utf-8"))
    records = generate_hard_negative_records(
        instances,
        captions,
        split=args.split,
        count=args.count,
        seed=args.seed,
        max_per_category=args.max_per_category,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    hardness = [float(record["conditional_cooccurrence"]) for record in records]
    summary = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "instances_sha256": hashlib.sha256(args.instances.read_bytes()).hexdigest(),
        "captions_sha256": hashlib.sha256(args.captions.read_bytes()).hexdigest(),
        **grounding_summary(records),
        "conditional_cooccurrence": {
            "min": min(hardness),
            "mean": sum(hardness) / len(hardness),
            "max": max(hardness),
        },
    }
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
