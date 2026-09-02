from pathlib import Path

from grounded_vqa.data.selection import select_answer_type_fraction, select_samples
from grounded_vqa.data.vqav2 import VQASample


def make_sample(question_id: int, answer_type: str = "yes/no") -> VQASample:
    return VQASample(
        question_id=question_id,
        image_id=question_id,
        image_path=Path(f"{question_id}.jpg"),
        question="Q?",
        answers=("yes",) * 10,
        answer_type=answer_type,
        question_type="is",
    )


def test_select_samples_is_deterministic_and_ordered() -> None:
    samples = [make_sample(index) for index in range(20)]
    first = select_samples(samples, 5, 42)
    second = select_samples(samples, 5, 42)
    assert [sample.question_id for sample in first] == [sample.question_id for sample in second]
    assert [sample.question_id for sample in first] == sorted(
        sample.question_id for sample in first
    )


def test_nested_exclusion_has_no_overlap() -> None:
    samples = [make_sample(index) for index in range(100)]
    excluded = select_samples(samples, 20, 42)
    excluded_ids = {sample.question_id for sample in excluded}
    remaining = [sample for sample in samples if sample.question_id not in excluded_ids]
    selected = select_samples(remaining, 30, 49)

    assert len(selected) == 30
    assert not excluded_ids.intersection(sample.question_id for sample in selected)


def test_select_answer_type_fraction_is_exact_and_deterministic() -> None:
    samples = [make_sample(index, "number" if index < 40 else "other") for index in range(100)]
    first = select_answer_type_fraction(
        samples, 20, answer_type="number", fraction=0.25, seed=42
    )
    second = select_answer_type_fraction(
        samples, 20, answer_type="number", fraction=0.25, seed=42
    )
    assert [sample.question_id for sample in first] == [sample.question_id for sample in second]
    assert sum(sample.answer_type == "number" for sample in first) == 5
    assert [sample.question_id for sample in first] == sorted(
        sample.question_id for sample in first
    )
