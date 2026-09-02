from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from grounded_vqa.data.answers import normalize_answer


def parse_official_pope_answer(answer: str) -> str:
    """Reproduce the original POPE evaluator's permissive answer parser."""

    first_sentence = answer.split(".", maxsplit=1)[0].replace(",", "")
    words = first_sentence.split(" ")
    return "no" if any(token in words for token in ("No", "not", "no")) else "yes"


def parse_strict_yes_no(answer: str) -> str | None:
    """Accept a direct yes/no response while exposing non-compliant generations."""

    normalized = normalize_answer(answer)
    if normalized in {"yes", "no"}:
        return normalized
    return None


def compute_pope_metrics(
    labels: Sequence[str], predictions: Sequence[str | None]
) -> dict[str, float | int | None]:
    if len(labels) != len(predictions):
        raise ValueError("POPE labels and predictions must have equal length")
    if not labels:
        raise ValueError("At least one POPE prediction is required")
    if any(label not in {"yes", "no"} for label in labels):
        raise ValueError("POPE labels must be yes or no")

    tp = sum(prediction == "yes" and label == "yes" for label, prediction in zip(labels, predictions))
    tn = sum(prediction == "no" and label == "no" for label, prediction in zip(labels, predictions))
    fp = sum(prediction == "yes" and label == "no" for label, prediction in zip(labels, predictions))
    fn = sum(prediction == "no" and label == "yes" for label, prediction in zip(labels, predictions))
    invalid = sum(prediction not in {"yes", "no"} for prediction in predictions)
    total = len(labels)

    def safe_ratio(numerator: float, denominator: float) -> float | None:
        return numerator / denominator if denominator else None

    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    specificity = safe_ratio(tn, tn + fp)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    balanced_accuracy = (
        (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None
    )
    return {
        "count": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "invalid": invalid,
        "accuracy": (tp + tn) / total,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "yes_ratio": sum(prediction == "yes" for prediction in predictions) / total,
        "invalid_ratio": invalid / total,
    }


def paired_pope_comparison(
    labels: Sequence[str],
    baseline: Sequence[str],
    candidate: Sequence[str],
    *,
    bootstrap_samples: int = 5000,
    seed: int = 42,
) -> dict[str, float | int | list[float]]:
    """Compare two POPE decoders on identical examples with paired uncertainty."""

    if len(labels) != len(baseline) or len(labels) != len(candidate):
        raise ValueError("Paired POPE inputs must have equal length")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    baseline_correct = np.asarray(
        [prediction == label for label, prediction in zip(labels, baseline)], dtype=np.int8
    )
    candidate_correct = np.asarray(
        [prediction == label for label, prediction in zip(labels, candidate)], dtype=np.int8
    )
    difference = candidate_correct - baseline_correct
    improved = int(np.sum(difference == 1))
    worsened = int(np.sum(difference == -1))
    discordant = improved + worsened
    chi_square = (
        (abs(improved - worsened) - 1) ** 2 / discordant if discordant else 0.0
    )
    p_value = math.erfc(math.sqrt(chi_square / 2)) if discordant else 1.0

    generator = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        sampled = generator.integers(0, len(difference), size=len(difference))
        bootstrap[index] = 100.0 * float(np.mean(difference[sampled]))
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "count": len(labels),
        "baseline_correct": int(np.sum(baseline_correct)),
        "candidate_correct": int(np.sum(candidate_correct)),
        "improved": improved,
        "worsened": worsened,
        "unchanged": len(labels) - discordant,
        "accuracy_delta_points": 100.0 * float(np.mean(difference)),
        "bootstrap_95_ci_points": [float(lower), float(upper)],
        "mcnemar_chi_square": chi_square,
        "mcnemar_p_value": p_value,
    }
