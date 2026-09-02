import pytest

from grounded_vqa.cli.diagnose_alignment import dependence_summary
from grounded_vqa.data.selection import build_mismatched_indices


def test_mismatched_indices_are_deterministic_and_use_different_images() -> None:
    image_ids = [10, 10, 20, 20, 30, 30]
    first = build_mismatched_indices(image_ids, seed=42)
    second = build_mismatched_indices(image_ids, seed=42)
    assert first == second
    assert sorted(first) == list(range(len(image_ids)))
    assert all(image_ids[index] != image_ids[other] for index, other in enumerate(first))


def test_mismatched_indices_reject_impossible_sample() -> None:
    with pytest.raises(ValueError, match="No complete"):
        build_mismatched_indices([1, 1, 1, 2], seed=42)


def test_dependence_summary_reports_change_and_normal_advantage() -> None:
    rows = [
        {
            "predictions": {"normal": "yes", "mismatched": "no"},
            "scores": {"normal": 1.0, "mismatched": 0.0},
        },
        {
            "predictions": {"normal": "two", "mismatched": "two"},
            "scores": {"normal": 0.5, "mismatched": 0.5},
        },
    ]
    summary = dependence_summary(rows, "mismatched")
    assert summary["answer_change_rate"] == 50.0
    assert summary["normal_advantage_rate"] == 50.0
    assert summary["equal_score_rate"] == 50.0
    assert summary["accuracy_drop_points"] == 50.0
