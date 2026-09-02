from grounded_vqa.cli.mine_complementary import summarize_scores


def test_summarize_scores_reports_hard_pair_rates() -> None:
    rows = [
        {"mean_margin": -1.0, "minimum_direction_margin": -2.0},
        {"mean_margin": 1.0, "minimum_direction_margin": 0.5},
    ]
    summary = summarize_scores(rows)
    assert summary["count"] == 2
    assert summary["mean_margin"] == 0.0
    assert summary["pair_preference_accuracy"] == 50.0
    assert summary["any_direction_wrong_rate"] == 50.0
