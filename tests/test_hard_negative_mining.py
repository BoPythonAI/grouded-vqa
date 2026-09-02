import pytest

from grounded_vqa.cli.mine_hard_negatives import select_hardest


def test_select_hardest_respects_category_cap() -> None:
    rows = [
        {"question_id": index, "category": "car", "yes_advantage": 10 - index}
        for index in range(5)
    ] + [
        {"question_id": 100 + index, "category": "dog", "yes_advantage": 5 - index}
        for index in range(5)
    ]
    selected = select_hardest(rows, count=4, max_per_category=2)
    assert [row["question_id"] for row in selected] == [0, 1, 100, 101]


def test_select_hardest_fails_when_diversity_cap_is_impossible() -> None:
    rows = [{"category": "car", "yes_advantage": 1.0} for _ in range(3)]
    with pytest.raises(RuntimeError):
        select_hardest(rows, count=2, max_per_category=1)
