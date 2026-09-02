from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from grounded_vqa.cli.train_complementary import sequence_token_nll
from grounded_vqa.data.coco_grounding import load_grounding_samples
from grounded_vqa.data.collator import VQAGenerativeCollator, move_batch_to_device
from grounded_vqa.models.loading import load_model, model_input_device
from grounded_vqa.models.lora import load_inference_lora
from grounded_vqa.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine E6-hard COCO negative questions")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-kind", choices=["blip2", "instructblip"], required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="4bit")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--selected-count", type=int, default=1000)
    parser.add_argument("--max-per-category", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-name", required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def select_hardest(
    rows: list[dict[str, Any]], count: int, max_per_category: int
) -> list[dict[str, Any]]:
    if count <= 0 or max_per_category <= 0:
        raise ValueError("count and max_per_category must be positive")
    category_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: float(item["yes_advantage"]), reverse=True):
        category = str(row["category"])
        if category_counts[category] >= max_per_category:
            continue
        selected.append(row)
        category_counts[category] += 1
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(
            f"Selected {len(selected)} of {count}; increase max-per-category or candidate pool"
        )
    return selected


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    paths = ProjectPaths.from_environment()
    paths.ensure()
    output_dir = paths.output_root / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows = load_jsonl(args.candidates)
    samples = load_grounding_samples(args.candidates, paths.data_root)
    if [int(row["question_id"]) for row in candidate_rows] != [
        sample.question_id for sample in samples
    ]:
        raise ValueError("Candidate metadata and loaded samples do not align")

    loaded = load_model(args.model_id, args.model_kind, args.quantization)
    model = load_inference_lora(loaded.model, args.adapter)
    model.eval()
    device = model_input_device(model)
    collator = VQAGenerativeCollator(loaded.processor, args.model_kind)
    score_path = output_dir / "scores.jsonl"
    scores = load_jsonl(score_path) if args.resume and score_path.is_file() else []
    expected = [sample.question_id for sample in samples[: len(scores)]]
    if [int(row["question_id"]) for row in scores] != expected:
        raise ValueError("Hard-negative score resume file is not an exact prefix")
    mode = "a" if args.resume else "w"
    with score_path.open(mode, encoding="utf-8") as output:
        for start in tqdm(
            range(len(scores), len(samples), args.batch_size), desc="mine-negatives"
        ):
            batch_samples = samples[start : start + args.batch_size]
            candidates = []
            for sample in batch_samples:
                candidates.extend(
                    (replace(sample, answers=("yes",) * 10), sample)
                )
            batch = collator(candidates)
            batch, _ = move_batch_to_device(batch, device)
            with torch.inference_mode():
                nll = sequence_token_nll(
                    model(**batch).logits, batch["labels"]
                ).detach().cpu()
            for index, sample in enumerate(batch_samples):
                source = dict(candidate_rows[start + index])
                yes_nll = float(nll[2 * index])
                no_nll = float(nll[2 * index + 1])
                source.update(
                    {
                        "yes_nll": yes_nll,
                        "no_nll": no_nll,
                        "yes_advantage": no_nll - yes_nll,
                        "pair_prediction": "yes" if yes_nll < no_nll else "no",
                    }
                )
                scores.append(source)
                output.write(json.dumps(source) + "\n")
            output.flush()

    selected = select_hardest(scores, args.selected_count, args.max_per_category)
    selected_path = output_dir / "selected.jsonl"
    selected_path.write_text(
        "".join(json.dumps(row) + "\n" for row in selected), encoding="utf-8"
    )
    summary = {
        "candidate_count": len(scores),
        "selected_count": len(selected),
        "candidate_yes_preference_rate": sum(
            row["pair_prediction"] == "yes" for row in scores
        )
        / len(scores),
        "selected_yes_preference_rate": sum(
            row["pair_prediction"] == "yes" for row in selected
        )
        / len(selected),
        "selected_yes_advantage": {
            "min": min(float(row["yes_advantage"]) for row in selected),
            "mean": sum(float(row["yes_advantage"]) for row in selected) / len(selected),
            "max": max(float(row["yes_advantage"]) for row in selected),
        },
        "selected_category_counts": dict(Counter(str(row["category"]) for row in selected)),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
