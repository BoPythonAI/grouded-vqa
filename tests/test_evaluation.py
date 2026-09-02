from grounded_vqa.evaluation.vqa import EvaluationExample, evaluate_examples


def test_evaluate_examples_reports_slices() -> None:
    metrics = evaluate_examples(
        [
            EvaluationExample("yes", ("yes",) * 10, "yes/no", "is"),
            EvaluationExample("2", ("two",) * 10, "number", "how many"),
            EvaluationExample("cat", ("dog",) * 10, "other", "what"),
        ]
    )
    assert metrics["overall"] == 100 * 2 / 3
    assert metrics["count"] == 3
    assert metrics["by_answer_type"]["yes/no"]["accuracy"] == 100.0
    assert metrics["by_answer_type"]["other"]["accuracy"] == 0.0


def test_yes_no_hallucination_proxy_reports_bias_and_errors() -> None:
    metrics = evaluate_examples(
        [
            EvaluationExample("yes", ("no",) * 10, "yes/no", "is"),
            EvaluationExample("no", ("yes",) * 10, "yes/no", "is"),
            EvaluationExample("maybe", ("yes",) * 10, "yes/no", "is"),
            EvaluationExample("yes", ("yes",) * 10, "yes/no", "is"),
        ]
    )
    proxy = metrics["yes_no_hallucination_proxy"]
    assert proxy["count"] == 4
    assert proxy["predicted_yes_rate"] == 50.0
    assert proxy["invalid_answer_rate"] == 25.0
    assert proxy["false_yes_rate"] == 100.0
    assert proxy["false_no_rate"] == 50.0
