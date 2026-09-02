from grounded_vqa.data.coco_grounding import (
    article_for,
    generate_grounding_records,
    generate_hard_negative_records,
    grounding_summary,
)


def payload() -> dict[str, object]:
    return {
        "categories": [{"id": 1, "name": "apple"}, {"id": 2, "name": "dog"}],
        "images": [
            {"id": 10, "width": 100, "height": 100},
            {"id": 20, "width": 100, "height": 100},
            {"id": 30, "width": 100, "height": 100},
        ],
        "annotations": [
            {"image_id": 10, "category_id": 1, "area": 1000, "iscrowd": 0},
            {"image_id": 20, "category_id": 2, "area": 1000, "iscrowd": 0},
        ],
    }


def test_generate_grounding_records_balances_tasks_and_images() -> None:
    records = generate_grounding_records(
        payload(),
        split="train",
        positive_count=1,
        negative_count=1,
        counting_count=1,
        min_area_ratio=0.001,
        max_count_answer=10,
        seed=7,
    )
    summary = grounding_summary(records)
    assert summary["count"] == 3
    assert summary["unique_images"] == 3
    assert summary["task_counts"] == {
        "count": 1,
        "existence_positive": 1,
        "existence_negative": 1,
    }


def test_article_for() -> None:
    assert article_for("apple") == "an"
    assert article_for("dog") == "a"


def test_generate_hard_negatives_excludes_instance_and_caption_objects() -> None:
    captions = {
        "annotations": [
            {"image_id": 10, "caption": "A red apple is on a table."},
            {"image_id": 20, "caption": "A dog is outside."},
        ]
    }
    records = generate_hard_negative_records(
        payload(),
        captions,
        split="train",
        count=2,
        seed=9,
        max_per_category=2,
    )
    assert len(records) == 2
    present = {10: {1}, 20: {2}}
    assert all(record["category_id"] not in present[record["image_id"]] for record in records)
    assert {record["answer"] for record in records} == {"no"}
    assert len({record["image_id"] for record in records}) == 2
