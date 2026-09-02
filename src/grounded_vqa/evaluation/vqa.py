from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from grounded_vqa.data.answers import consensus_answer, normalize_answer, vqa_soft_accuracy


@dataclass(frozen=True)
class EvaluationExample:
    prediction: str
    references: tuple[str, ...]
    answer_type: str = "unknown"
    question_type: str = "unknown"


def evaluate_examples(examples: Iterable[EvaluationExample]) -> dict[str, object]:
    total_scores: list[float] = []
    by_answer_type: dict[str, list[float]] = defaultdict(list)
    by_question_type: dict[str, list[float]] = defaultdict(list)
    yes_no_predictions: list[tuple[str, str]] = []

    for example in examples:
        score = vqa_soft_accuracy(example.prediction, example.references)
        total_scores.append(score)
        by_answer_type[example.answer_type].append(score)
        by_question_type[example.question_type].append(score)
        if example.answer_type == "yes/no":
            yes_no_predictions.append(
                (
                    normalize_answer(example.prediction),
                    consensus_answer(example.references),
                )
            )

    if not total_scores:
        raise ValueError("No evaluation examples supplied")

    average = lambda values: 100.0 * sum(values) / len(values)
    metrics: dict[str, object] = {
        "overall": average(total_scores),
        "count": len(total_scores),
        "by_answer_type": {
            key: {"accuracy": average(values), "count": len(values)}
            for key, values in sorted(by_answer_type.items())
        },
        "by_question_type": {
            key: {"accuracy": average(values), "count": len(values)}
            for key, values in sorted(by_question_type.items())
        },
    }
    if yes_no_predictions:
        total = len(yes_no_predictions)
        valid = [
            (prediction, target)
            for prediction, target in yes_no_predictions
            if prediction in {"yes", "no"}
        ]
        negative = [(prediction, target) for prediction, target in valid if target == "no"]
        positive = [(prediction, target) for prediction, target in valid if target == "yes"]
        metrics["yes_no_hallucination_proxy"] = {
            "count": total,
            "predicted_yes_rate": 100.0
            * sum(prediction == "yes" for prediction, _ in yes_no_predictions)
            / total,
            "invalid_answer_rate": 100.0 * (total - len(valid)) / total,
            # A false "yes" on a human-consensus "no" is a useful VQAv2
            # presence-hallucination proxy, but is not a full POPE-style score.
            "false_yes_rate": (
                100.0 * sum(prediction == "yes" for prediction, _ in negative) / len(negative)
                if negative
                else None
            ),
            "false_no_rate": (
                100.0 * sum(prediction == "no" for prediction, _ in positive) / len(positive)
                if positive
                else None
            ),
        }
    return metrics
