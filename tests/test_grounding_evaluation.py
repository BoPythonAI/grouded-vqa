from grounded_vqa.cli.evaluate_grounding import compute_grounding_metrics


def test_compute_grounding_metrics() -> None:
    rows = [
        {"task_type": "existence_positive", "answer": "yes", "prediction": "yes"},
        {"task_type": "existence_negative", "answer": "no", "prediction": "yes"},
        {"task_type": "count", "answer": "2", "prediction": "3"},
    ]
    metrics = compute_grounding_metrics(rows)
    assert metrics["overall_exact_accuracy"] == 100 / 3
    assert metrics["false_yes_rate"] == 100.0
    assert metrics["false_no_rate"] == 0.0
    assert metrics["count_mae"] == 1.0


def test_compute_grounding_metrics_handles_missing_task_types() -> None:
    metrics = compute_grounding_metrics(
        [{"task_type": "existence_positive", "answer": "yes", "prediction": "yes"}]
    )
    assert metrics["overall_exact_accuracy"] == 100.0
    assert metrics["false_yes_rate"] is None
    assert metrics["false_no_rate"] == 0.0
    assert metrics["count_mae"] is None
    assert metrics["count_valid_number_rate"] is None
