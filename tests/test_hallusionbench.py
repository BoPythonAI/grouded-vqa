import pytest

from grounded_vqa.evaluation.hallusionbench import compute_hallusionbench_metrics


def row(figure: str, question: str, gt: str, prediction: str, visual: str) -> dict:
    return {
        "category": "VS",
        "subcategory": "chart",
        "set_id": "0",
        "figure_id": figure,
        "question_id": question,
        "gt_answer": gt,
        "visual_input": visual,
        "prediction": prediction,
    }


def test_hallusionbench_question_pair_and_figure_metrics() -> None:
    metrics = compute_hallusionbench_metrics(
        [
            row("0", "0", "1", "yes", "0"),
            row("1", "0", "0", "no", "1"),
            row("2", "0", "1", "no", "2"),
            row("0", "1", "0", "no", "0"),
            row("1", "1", "1", "yes", "1"),
            row("2", "1", "0", "no", "2"),
        ]
    )
    assert metrics["question"]["accuracy"] == pytest.approx(5 / 6)
    assert metrics["question_pair"]["accuracy"] == 0.5
    assert metrics["easy_question_pair"]["accuracy"] == 1.0
    assert metrics["hard_question_pair"]["accuracy"] == 0.5
    assert metrics["figure"]["accuracy"] == pytest.approx(1 / 2)
    assert metrics["false_negative_rate"] == pytest.approx(1 / 3)


def test_hallusionbench_invalid_answer_is_incorrect() -> None:
    metrics = compute_hallusionbench_metrics([row("1", "0", "1", "maybe", "1")])
    assert metrics["question"]["accuracy"] == 0.0
    assert metrics["invalid_ratio"] == 1.0
