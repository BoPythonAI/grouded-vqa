from grounded_vqa.cli.train_grounding_qformer import evaluate_grounding_nll


def test_evaluate_grounding_nll_is_importable() -> None:
    assert callable(evaluate_grounding_nll)
