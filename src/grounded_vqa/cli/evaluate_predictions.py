from __future__ import annotations

import argparse
import json
from pathlib import Path

from grounded_vqa.evaluation.vqa import EvaluationExample, evaluate_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate JSONL VQA predictions")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    examples: list[EvaluationExample] = []
    with args.predictions.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            examples.append(
                EvaluationExample(
                    prediction=item["prediction"],
                    references=tuple(item["references"]),
                    answer_type=item.get("answer_type", "unknown"),
                    question_type=item.get("question_type", "unknown"),
                )
            )
    metrics = evaluate_examples(examples)
    rendered = json.dumps(metrics, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

