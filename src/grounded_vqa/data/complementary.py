from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from grounded_vqa.data.answers import consensus_answer
from grounded_vqa.data.vqav2 import VQASample


@dataclass(frozen=True)
class ComplementaryPairAudit:
    source_pairs: int
    usable_pairs: int
    missing_question_ids: int
    question_mismatches: int
    same_image_pairs: int
    same_target_pairs: int


def load_complementary_pair_ids(path: Path) -> list[tuple[int, int]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError("Complementary-pair file must contain a JSON list")

    pairs: list[tuple[int, int]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"Invalid complementary pair at index {index}: {item!r}")
        pairs.append((int(item[0]), int(item[1])))
    return pairs


def build_complementary_pairs(
    samples: list[VQASample],
    pair_ids: list[tuple[int, int]],
) -> tuple[list[tuple[VQASample, VQASample]], ComplementaryPairAudit]:
    """Join and validate official VQAv2 complementary pairs.

    Pairs with the same deterministic generative target are excluded because
    swapping their images would not create a useful answer-level hard negative.
    """

    by_question_id = {sample.question_id: sample for sample in samples}
    usable: list[tuple[VQASample, VQASample]] = []
    missing = 0
    question_mismatches = 0
    same_image = 0
    same_target = 0
    for first_id, second_id in pair_ids:
        first = by_question_id.get(first_id)
        second = by_question_id.get(second_id)
        if first is None or second is None:
            missing += 1
            continue
        if first.question != second.question:
            question_mismatches += 1
            continue
        if first.image_id == second.image_id:
            same_image += 1
            continue
        if consensus_answer(first.answers) == consensus_answer(second.answers):
            same_target += 1
            continue
        usable.append((first, second))

    audit = ComplementaryPairAudit(
        source_pairs=len(pair_ids),
        usable_pairs=len(usable),
        missing_question_ids=missing,
        question_mismatches=question_mismatches,
        same_image_pairs=same_image,
        same_target_pairs=same_target,
    )
    return usable, audit
