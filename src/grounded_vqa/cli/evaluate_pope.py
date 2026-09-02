from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

from grounded_vqa.cli.train_complementary import sequence_token_nll
from grounded_vqa.data.answer_routing import choose_yes_no
from grounded_vqa.data.collator import VQAGenerativeCollator, move_batch_to_device
from grounded_vqa.data.pope import PopeExample, load_pope
from grounded_vqa.data.vqav2 import VQASample
from grounded_vqa.evaluation.pope import (
    compute_pope_metrics,
    parse_official_pope_answer,
    parse_strict_yes_no,
)
from grounded_vqa.models.loading import ModelKind, format_prompt, load_model, model_input_device
from grounded_vqa.models.lora import load_inference_lora
from grounded_vqa.paths import ProjectPaths

STRATEGIES = ("random", "popular", "adversarial")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a BLIP model on official COCO POPE")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-kind", choices=["blip2", "instructblip"], required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--pope-root", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="4bit")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--max-samples", type=int, help="Per-strategy prefix for smoke tests")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-name", required=True)
    return parser.parse_args()


def yes_no_question(question: str) -> str:
    return f"{question.strip()}\nAnswer using only yes or no."


def prediction_prompt(kind: ModelKind, question: str) -> str:
    return format_prompt(kind, yes_no_question(question), "default")


def image_id_from_path(path: Path, fallback: int) -> int:
    match = re.search(r"(\d{12})$", path.stem)
    return int(match.group(1)) if match else fallback


def candidate_samples(examples: list[PopeExample]) -> list[VQASample]:
    samples: list[VQASample] = []
    for example in examples:
        base = VQASample(
            question_id=example.question_id,
            image_id=image_id_from_path(example.image_path, example.question_id),
            image_path=example.image_path,
            question=yes_no_question(example.question),
            answers=("yes",),
            answer_type="yes/no",
            question_type="pope",
        )
        samples.extend((base, replace(base, answers=("no",))))
    return samples


def load_resume_rows(path: Path, examples: list[PopeExample]) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    completed = [int(row["question_id"]) for row in rows]
    expected = [example.question_id for example in examples[: len(rows)]]
    if completed != expected:
        raise ValueError(f"POPE resume file is not an exact prefix: {path}")
    return rows


def evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [str(row["label"]) for row in rows]
    generated = [str(row["generated_prediction"]) for row in rows]
    return {
        "generated_official_parser": compute_pope_metrics(
            labels, [parse_official_pope_answer(answer) for answer in generated]
        ),
        "generated_strict_parser": compute_pope_metrics(
            labels, [parse_strict_yes_no(answer) for answer in generated]
        ),
        "pair_logprob": compute_pope_metrics(
            labels, [str(row["pair_logprob_prediction"]) for row in rows]
        ),
    }


def macro_average(strategy_metrics: dict[str, Any], decoder: str) -> dict[str, float]:
    keys = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "yes_ratio", "invalid_ratio")
    return {
        key: sum(float(metrics[decoder][key]) for metrics in strategy_metrics.values())
        / len(strategy_metrics)
        for key in keys
        if all(metrics[decoder][key] is not None for metrics in strategy_metrics.values())
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    paths = ProjectPaths.from_environment()
    paths.ensure()
    pope_root = args.pope_root or paths.data_root / "pope"
    image_root = args.image_root or paths.data_root / "val2014"
    output_dir = paths.output_root / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_model(args.model_id, args.model_kind, args.quantization)
    model = loaded.model
    if args.adapter:
        model = load_inference_lora(model, args.adapter)
    model.eval()
    device = model_input_device(model)
    collator = VQAGenerativeCollator(loaded.processor, args.model_kind, prompt_style="default")

    started = time.monotonic()
    all_metrics: dict[str, Any] = {}
    for strategy in args.strategies:
        source = pope_root / f"coco_pope_{strategy}.json"
        examples = load_pope(source, image_root)
        if args.max_samples is not None:
            if args.max_samples <= 0:
                raise ValueError("max-samples must be positive")
            examples = examples[: args.max_samples]
        predictions_path = output_dir / f"predictions_{strategy}.jsonl"
        rows = load_resume_rows(predictions_path, examples) if args.resume else []
        mode = "a" if args.resume else "w"
        with predictions_path.open(mode, encoding="utf-8") as output:
            for start in tqdm(
                range(len(rows), len(examples), args.batch_size),
                desc=f"pope-{strategy}",
            ):
                batch_examples = examples[start : start + args.batch_size]
                images: list[Image.Image] = []
                for example in batch_examples:
                    with Image.open(example.image_path) as image:
                        images.append(image.convert("RGB"))
                prompts = [
                    prediction_prompt(args.model_kind, example.question)
                    for example in batch_examples
                ]
                inputs = loaded.processor(
                    images=images,
                    text=prompts,
                    padding=True,
                    return_tensors="pt",
                )
                for key, value in list(inputs.items()):
                    if isinstance(value, torch.Tensor):
                        value = value.to(device)
                        if key == "pixel_values":
                            value = value.to(dtype=torch.bfloat16)
                        inputs[key] = value
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=args.max_new_tokens,
                    )
                generated_answers = loaded.processor.batch_decode(
                    generated, skip_special_tokens=True
                )

                candidate_batch = collator(candidate_samples(batch_examples))
                candidate_batch, _ = move_batch_to_device(candidate_batch, device)
                with torch.inference_mode():
                    nll = sequence_token_nll(
                        model(**candidate_batch).logits,
                        candidate_batch["labels"],
                    ).detach().cpu()
                for index, (example, generated_answer) in enumerate(
                    zip(batch_examples, generated_answers)
                ):
                    yes_nll = float(nll[2 * index])
                    no_nll = float(nll[2 * index + 1])
                    row = {
                        "question_id": example.question_id,
                        "image": example.image_path.name,
                        "question": example.question,
                        "label": example.label,
                        "generated_prediction": generated_answer.strip(),
                        "generated_official_prediction": parse_official_pope_answer(
                            generated_answer
                        ),
                        "generated_strict_prediction": parse_strict_yes_no(
                            generated_answer
                        ),
                        "yes_nll": yes_nll,
                        "no_nll": no_nll,
                        "pair_logprob_prediction": choose_yes_no(yes_nll, no_nll),
                    }
                    rows.append(row)
                    output.write(json.dumps(row) + "\n")
                output.flush()
        all_metrics[strategy] = evaluate_rows(rows)

    metrics = {
        "benchmark": "official COCO POPE",
        "strategies": all_metrics,
        "macro_average": {
            decoder: macro_average(all_metrics, decoder)
            for decoder in (
                "generated_official_parser",
                "generated_strict_parser",
                "pair_logprob",
            )
        },
        "elapsed_seconds": time.monotonic() - started,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
