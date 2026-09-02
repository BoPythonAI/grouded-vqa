from __future__ import annotations

import argparse
import json
import random
from collections import Counter

from PIL import Image

from grounded_vqa.cli.train_lora import resolve_data_files
from grounded_vqa.data.answers import consensus_answer
from grounded_vqa.data.vqav2 import load_vqav2
from grounded_vqa.paths import ProjectPaths


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit VQAv2 joins and image files")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = ProjectPaths.from_environment()
    questions, annotations, images_root = resolve_data_files(paths.data_root, args.split)
    samples = load_vqav2(questions, annotations, images_root, args.split)
    selected = random.Random(args.seed).sample(samples, min(args.samples, len(samples)))
    answer_types: Counter[str] = Counter()
    for sample in selected:
        with Image.open(sample.image_path) as image:
            image.verify()
        if len(sample.answers) != 10:
            raise ValueError(
                f"question_id={sample.question_id} has {len(sample.answers)} answers, expected 10"
            )
        if not consensus_answer(sample.answers):
            raise ValueError(f"question_id={sample.question_id} has an empty consensus answer")
        answer_types[sample.answer_type or "unknown"] += 1

    report = {
        "split": args.split,
        "dataset_size": len(samples),
        "audited": len(selected),
        "answer_types": dict(sorted(answer_types.items())),
        "status": "ok",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
