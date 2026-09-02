from grounded_vqa.data.error_grounding import (
    generate_error_grounding_records,
    match_coco_category,
    nested_task_subset,
)


def payload() -> dict:
    return {
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 2, "name": "dog"},
            {"id": 3, "name": "bicycle"},
        ],
        "images": [
            {"id": 10, "width": 100, "height": 100},
            {"id": 11, "width": 100, "height": 100},
            {"id": 12, "width": 100, "height": 100},
        ],
        "annotations": [
            {"image_id": 10, "category_id": 1, "area": 1000, "iscrowd": 0},
            {"image_id": 11, "category_id": 2, "area": 1000, "iscrowd": 0},
            {"image_id": 11, "category_id": 2, "area": 900, "iscrowd": 0},
            {"image_id": 12, "category_id": 3, "area": 1000, "iscrowd": 0},
        ],
    }


def prediction(qid: int, image_id: int, question: str, answer: str, predicted: str) -> dict:
    return {
        "question_id": qid,
        "image_id": image_id,
        "question": question,
        "prediction": predicted,
        "references": [answer] * 10,
    }


def test_category_alias_matching() -> None:
    categories = {1: "person", 2: "dog", 3: "bicycle"}
    assert match_coco_category("How many people are there?", categories) == (1, "person")
    assert match_coco_category("Is there a bike?", categories) == (3, "bicycle")


def test_error_grounding_requires_coco_verification() -> None:
    predictions = [
        prediction(1, 12, "Is there a dog in the image?", "no", "yes"),
        prediction(2, 10, "Can you see a person?", "yes", "no"),
        prediction(3, 11, "How many dogs are there?", "2", "1"),
    ]
    records, audit = generate_error_grounding_records(
        payload(),
        predictions,
        positive_count=1,
        negative_count=1,
        counting_count=1,
        min_area_ratio=0.001,
        max_count_answer=10,
        seed=49,
    )

    assert len(records) == 3
    assert audit["unique_images"] == 3
    assert {record["source_error"] for record in records} == {
        "false_yes",
        "false_no",
        "count_error",
    }
    assert all(record["question_id"] < 0 for record in records)


def test_nested_subset_preserves_master_ids() -> None:
    records = [
        {"task_type": "existence_positive", "question_id": -1},
        {"task_type": "existence_positive", "question_id": -2},
        {"task_type": "existence_negative", "question_id": -3},
        {"task_type": "existence_negative", "question_id": -4},
        {"task_type": "count", "question_id": -5},
        {"task_type": "count", "question_id": -6},
    ]
    subset = nested_task_subset(
        records,
        positive_count=1,
        negative_count=1,
        counting_count=1,
    )
    assert len(subset) == 3
    assert {item["question_id"] for item in subset}.issubset(
        item["question_id"] for item in records
    )
