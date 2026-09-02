from __future__ import annotations

import argparse
import json
from pathlib import Path

from grounded_vqa.cli.train_lora import resolve_data_files
from grounded_vqa.data.multiple_choice import build_multiple_choice_examples
from grounded_vqa.data.selection import select_samples
from grounded_vqa.data.vqav2 import load_vqav2
from grounded_vqa.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a VQAv2-derived MCQ benchmark")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--max-options", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_environment()
    questions, annotations, images_root = resolve_data_files(paths.data_root, args.split)
    samples = load_vqav2(questions, annotations, images_root, args.split)
    examples, audit = build_multiple_choice_examples(
        samples,
        max_options=args.max_options,
        seed=args.seed,
    )
    examples = select_samples(examples, args.max_samples, args.seed)
    audit["selected_examples"] = len(examples)
    audit["selected_question_ids"] = [example.question_id for example in examples]

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as output:
        for example in examples:
            output.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
    audit_path = args.output_file.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "selected_question_ids"}, indent=2))


if __name__ == "__main__":
    main()
