from __future__ import annotations

import random
from collections import Counter
from typing import TypeVar

from grounded_vqa.data.vqav2 import VQASample

Item = TypeVar("Item")


def select_samples(samples: list[Item], max_samples: int | None, seed: int) -> list[Item]:
    """Select a deterministic random subset while preserving source order."""
    if max_samples is None or max_samples >= len(samples):
        return samples
    generator = random.Random(seed)
    indices = sorted(generator.sample(range(len(samples)), max_samples))
    return [samples[index] for index in indices]


def select_answer_type_fraction(
    samples: list[VQASample],
    max_samples: int,
    *,
    answer_type: str,
    fraction: float,
    seed: int,
) -> list[VQASample]:
    """Select a fixed-size subset with an exact answer-type fraction."""

    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1")
    if max_samples > len(samples):
        raise ValueError("max_samples exceeds available samples")

    target_count = round(max_samples * fraction)
    other_count = max_samples - target_count
    target_indices = [
        index for index, sample in enumerate(samples) if sample.answer_type == answer_type
    ]
    other_indices = [
        index for index, sample in enumerate(samples) if sample.answer_type != answer_type
    ]
    if target_count > len(target_indices) or other_count > len(other_indices):
        raise ValueError("Requested answer-type mix exceeds available samples")

    generator = random.Random(seed)
    selected = generator.sample(target_indices, target_count)
    selected.extend(generator.sample(other_indices, other_count))
    return [samples[index] for index in sorted(selected)]


def build_mismatched_indices(image_ids: list[int], seed: int) -> list[int]:
    """Return a deterministic permutation with a different image at every position."""
    if len(image_ids) < 2 or len(set(image_ids)) < 2:
        raise ValueError("At least two distinct images are required for mismatch diagnostics")
    largest_group = max(Counter(image_ids).values())
    if largest_group > len(image_ids) // 2:
        raise ValueError("No complete different-image permutation exists for this sample")

    generator = random.Random(seed)
    candidate = list(range(len(image_ids)))
    for _ in range(2000):
        generator.shuffle(candidate)
        if all(image_ids[index] != image_ids[other] for index, other in enumerate(candidate)):
            return candidate.copy()
    raise RuntimeError("Unable to construct a deterministic mismatched-image permutation")
