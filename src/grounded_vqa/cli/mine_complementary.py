from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from grounded_vqa.cli.train_complementary import (
    ComplementaryPairCollator,
    ComplementaryPairDataset,
    sequence_token_nll,
)
from grounded_vqa.cli.train_lora import resolve_data_files, seed_everything
from grounded_vqa.data.collator import VQAGenerativeCollator, move_batch_to_device
from grounded_vqa.data.complementary import (
    build_complementary_pairs,
    load_complementary_pair_ids,
)
from grounded_vqa.data.selection import select_samples
from grounded_vqa.data.vqav2 import VQASample, load_vqav2
from grounded_vqa.models.loading import load_model, model_input_device
from grounded_vqa.models.lora import load_inference_lora
from grounded_vqa.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine difficult VQAv2 complementary pairs with a frozen model"
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-kind", choices=["blip2", "instructblip"], required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--pairs-file", type=Path)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="4bit")
    parser.add_argument("--candidate-pairs", type=int, default=5000)
    parser.add_argument("--selected-pairs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4, help="Number of pairs per batch")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--output-name", required=True)
    return parser.parse_args()


@dataclass
class ScoringCollator:
    base: ComplementaryPairCollator

    def __call__(
        self,
        pairs: list[tuple[VQASample, VQASample]],
    ) -> tuple[dict[str, Any], dict[str, Any], list[tuple[int, int]]]:
        positive, negative = self.base(pairs)
        pair_ids = [(first.question_id, second.question_id) for first, second in pairs]
        return positive, negative, pair_ids


def summarize_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    margins = np.asarray([row["mean_margin"] for row in rows], dtype=np.float64)
    minimum_margins = np.asarray(
        [row["minimum_direction_margin"] for row in rows], dtype=np.float64
    )
    return {
        "count": len(rows),
        "mean_margin": float(margins.mean()),
        "median_margin": float(np.median(margins)),
        "margin_percentiles": {
            str(percentile): float(np.percentile(margins, percentile))
            for percentile in (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)
        },
        "pair_preference_accuracy": float(100.0 * np.mean(margins > 0)),
        "any_direction_wrong_rate": float(100.0 * np.mean(minimum_margins <= 0)),
    }


def main() -> None:
    args = parse_args()
    if args.candidate_pairs <= 0 or args.selected_pairs <= 0:
        raise ValueError("candidate-pairs and selected-pairs must be positive")
    if args.selected_pairs > args.candidate_pairs:
        raise ValueError("selected-pairs cannot exceed candidate-pairs")
    seed_everything(args.seed)

    paths = ProjectPaths.from_environment()
    paths.ensure()
    output_dir = paths.output_root / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_path = args.pairs_file or (
        paths.data_root
        / "complementary_pairs"
        / f"v2_mscoco_{args.split}2014_complementary_pairs.json"
    )

    questions, annotations, images_root = resolve_data_files(paths.data_root, args.split)
    samples = load_vqav2(questions, annotations, images_root, args.split)
    pair_ids = load_complementary_pair_ids(pair_path)
    usable_pairs, audit = build_complementary_pairs(samples, pair_ids)
    candidates = select_samples(usable_pairs, args.candidate_pairs, args.seed)

    loaded = load_model(args.model_id, args.model_kind, args.quantization)
    model = load_inference_lora(loaded.model, args.adapter)
    model.eval()
    device = model_input_device(model)
    collator = ScoringCollator(
        ComplementaryPairCollator(VQAGenerativeCollator(loaded.processor, args.model_kind))
    )
    loader = DataLoader(
        ComplementaryPairDataset(candidates),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    with torch.inference_mode():
        for positive_batch, negative_batch, batch_pair_ids in tqdm(loader, desc="mine-pairs"):
            positive_batch, positive_metadata = move_batch_to_device(positive_batch, device)
            negative_batch, _ = move_batch_to_device(negative_batch, device)
            positive_nll = sequence_token_nll(
                model(**positive_batch).logits,
                positive_batch["labels"],
            )
            negative_nll = sequence_token_nll(
                model(**negative_batch).logits,
                negative_batch["labels"],
            )
            margins = negative_nll - positive_nll
            for pair_index, pair in enumerate(batch_pair_ids):
                first_index = 2 * pair_index
                direction_margins = margins[first_index : first_index + 2]
                first_metadata = positive_metadata[first_index]
                second_metadata = positive_metadata[first_index + 1]
                rows.append(
                    {
                        "question_ids": [int(pair[0]), int(pair[1])],
                        "answer_types": [
                            first_metadata["answer_type"],
                            second_metadata["answer_type"],
                        ],
                        "positive_nll": [
                            float(positive_nll[first_index]),
                            float(positive_nll[first_index + 1]),
                        ],
                        "swapped_nll": [
                            float(negative_nll[first_index]),
                            float(negative_nll[first_index + 1]),
                        ],
                        "direction_margins": [
                            float(direction_margins[0]),
                            float(direction_margins[1]),
                        ],
                        "mean_margin": float(direction_margins.mean()),
                        "minimum_direction_margin": float(direction_margins.min()),
                    }
                )

    ranked = sorted(rows, key=lambda row: (row["mean_margin"], row["question_ids"]))
    selected = ranked[: args.selected_pairs]
    selected_ids = [row["question_ids"] for row in selected]
    with (output_dir / "candidate_scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in ranked:
            handle.write(json.dumps(row) + "\n")
    (output_dir / "selected_pairs.json").write_text(
        json.dumps(selected_ids) + "\n",
        encoding="utf-8",
    )
    summary = {
        "arguments": {**vars(args), "adapter": str(args.adapter)},
        "pairs_file": str(pair_path),
        "pair_audit": asdict(audit),
        "candidate_summary": summarize_scores(rows),
        "selected_summary": summarize_scores(selected),
        "elapsed_seconds": time.monotonic() - started,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
