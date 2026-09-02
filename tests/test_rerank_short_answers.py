from grounded_vqa.cli.rerank_short_answers import apply_reranking, eligible_rows


def test_eligible_rows_only_selects_invalid_routed_answers() -> None:
    rows = [
        {"question_id": 1, "question": "Is there a dog?", "prediction": "dog"},
        {"question_id": 2, "question": "Is there a cat?", "prediction": "yes"},
        {"question_id": 3, "question": "What animal is shown?", "prediction": "dog"},
    ]
    assert [row["question_id"] for row in eligible_rows(rows)] == [1]


def test_apply_reranking_preserves_unselected_rows() -> None:
    rows = [
        {"question_id": 1, "prediction": "dog"},
        {"question_id": 2, "prediction": "cat"},
    ]
    scores = [
        {
            "question_id": 1,
            "reranked_prediction": "yes",
            "yes_nll": 0.1,
            "no_nll": 0.8,
        }
    ]
    updated = apply_reranking(rows, scores)
    assert updated[0]["prediction"] == "yes"
    assert updated[0]["original_prediction"] == "dog"
    assert updated[1] == rows[1]
