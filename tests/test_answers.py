import pytest

from grounded_vqa.data.answers import (
    consensus_answer,
    normalize_answer,
    sampled_reference_answer,
    vqa_soft_accuracy,
)


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("The two.", "2"),
        ("Dont!", "don't"),
        ("1,000", "1000"),
        ("2.5", "2.5"),
        ("A red-and-white car", "red and white car"),
    ],
)
def test_normalize_answer(raw: str, normalized: str) -> None:
    assert normalize_answer(raw) == normalized


def test_vqa_soft_accuracy_full_agreement() -> None:
    assert vqa_soft_accuracy("two", ["2"] * 10) == 1.0


def test_vqa_soft_accuracy_partial_agreement() -> None:
    references = ["cat"] * 3 + ["dog"] * 7
    assert vqa_soft_accuracy("cat", references) == pytest.approx(0.9)


def test_vqa_soft_accuracy_rejects_empty_references() -> None:
    with pytest.raises(ValueError):
        vqa_soft_accuracy("yes", [])


def test_consensus_answer_normalizes_and_preserves_first_tie() -> None:
    assert consensus_answer(["Two", "2", "three", "3"]) == "2"


def test_sampled_reference_answer_is_reproducible_and_normalized() -> None:
    references = ["The cat."] * 7 + ["A dog!"] * 3
    first = sampled_reference_answer(references, seed=42, sample_key=100)
    second = sampled_reference_answer(references, seed=42, sample_key=100)
    assert first == second
    assert first in {"cat", "dog"}


def test_sampled_reference_answer_preserves_frequency_across_samples() -> None:
    references = ["cat"] * 8 + ["dog"] * 2
    selected = [
        sampled_reference_answer(references, seed=42, sample_key=key)
        for key in range(1000)
    ]
    assert 750 <= selected.count("cat") <= 850
