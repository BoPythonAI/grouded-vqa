import json
from pathlib import Path

from grounded_vqa.data.complementary import (
    build_complementary_pairs,
    load_complementary_pair_ids,
)
from grounded_vqa.data.vqav2 import VQASample


def sample(question_id: int, image_id: int, answer: str) -> VQASample:
    return VQASample(
        question_id=question_id,
        image_id=image_id,
        image_path=Path(f"{image_id}.jpg"),
        question="Is it visible?",
        answers=(answer,) * 10,
        answer_type="yes/no",
        question_type="is",
    )


def test_load_complementary_pair_ids(tmp_path: Path) -> None:
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps([[1, 2], [3, 4]]), encoding="utf-8")
    assert load_complementary_pair_ids(path) == [(1, 2), (3, 4)]


def test_build_complementary_pairs_filters_same_target() -> None:
    samples = [sample(1, 10, "yes"), sample(2, 20, "no"), sample(3, 30, "yes")]
    pairs, audit = build_complementary_pairs(samples, [(1, 2), (1, 3), (1, 999)])
    assert [(first.question_id, second.question_id) for first, second in pairs] == [(1, 2)]
    assert audit.source_pairs == 3
    assert audit.usable_pairs == 1
    assert audit.same_target_pairs == 1
    assert audit.missing_question_ids == 1
