from pathlib import Path

import torch

from grounded_vqa.cli.train_mixed_qformer import (
    MixedUnitCollator,
    build_mixed_units,
    evaluate_pair_margin,
)
from grounded_vqa.data.vqav2 import VQASample


def sample(question_id: int) -> VQASample:
    return VQASample(
        question_id=question_id,
        image_id=question_id,
        image_path=Path(f"{question_id}.jpg"),
        question="Question?",
        answers=(str(question_id),) * 10,
        answer_type="other",
        question_type="what",
    )


def test_build_mixed_units_preserves_requested_ratio() -> None:
    ordinary = [sample(index) for index in range(6)]
    hard = [(sample(10), sample(11)), (sample(12), sample(13))]
    units = build_mixed_units(ordinary, hard)
    assert len(units) == 5
    assert sum(unit.is_hard_pair for unit in units) == 2
    assert sum(2 for _ in units) == 10


def test_mixed_collator_marks_only_hard_positive_indices() -> None:
    class FakeBase:
        def __call__(self, samples: list[VQASample]) -> dict[str, object]:
            return {"question_ids": [item.question_id for item in samples]}

    units = build_mixed_units(
        [sample(0), sample(1)],
        [(sample(10), sample(11))],
    )
    positive, negative, indices = MixedUnitCollator(FakeBase())(units)  # type: ignore[arg-type]
    assert positive["question_ids"] == [0, 1, 10, 11]
    assert negative is not None
    assert negative["question_ids"] == [10, 11]
    assert indices == [2, 3]


def test_evaluate_pair_margin_is_importable() -> None:
    assert callable(evaluate_pair_margin)
    assert torch.tensor(1).item() == 1
