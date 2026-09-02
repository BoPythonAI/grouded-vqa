import json
from pathlib import Path

import pytest

from grounded_vqa.evaluation.chair import (
    build_ground_truth_objects,
    compute_chair_metrics,
    extract_coco_objects,
    paired_chair_comparison,
)


def test_extract_coco_objects_normalizes_plural_and_double_words() -> None:
    mentions = extract_coco_objects(
        "Two men stand by bicycles, a traffic light, and wine glasses."
    )
    assert [mention.canonical for mention in mentions] == [
        "person",
        "bicycle",
        "traffic light",
        "wine glass",
    ]


def test_extract_coco_objects_avoids_toilet_seat_chair_false_positive() -> None:
    mentions = extract_coco_objects("A toilet seat is beside a chair.")
    assert [mention.canonical for mention in mentions] == ["toilet", "chair"]


def test_build_ground_truth_and_compute_chair(tmp_path: Path) -> None:
    instances = {
        "categories": [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}],
        "annotations": [{"image_id": 7, "category_id": 1}],
    }
    captions = {
        "annotations": [{"image_id": 7, "caption": "A man stands by a car."}]
    }
    (tmp_path / "instances_val2014.json").write_text(json.dumps(instances))
    (tmp_path / "captions_val2014.json").write_text(json.dumps(captions))
    ground_truth = build_ground_truth_objects([7], tmp_path)
    assert ground_truth == {7: {"person", "car"}}

    metrics = compute_chair_metrics(
        [{"image_id": 7, "caption": "A man rides a bicycle beside the car."}],
        ground_truth,
    )
    assert metrics["chair_s"] == 1.0
    assert metrics["chair_i"] == pytest.approx(1 / 3)
    assert metrics["object_recall"] == 1.0


def test_compute_chair_requires_known_image() -> None:
    with pytest.raises(KeyError):
        compute_chair_metrics([{"image_id": 2, "caption": "a dog"}], {1: {"dog"}})


def test_paired_chair_comparison_reports_direction_and_ci() -> None:
    baseline = [
        {
            "image_id": image_id,
            "ground_truth_objects": ["person"],
            "generated_objects": ["person"],
            "hallucinated_objects": [],
        }
        for image_id in range(10)
    ]
    candidate = [
        {
            "image_id": image_id,
            "ground_truth_objects": ["person"],
            "generated_objects": ["person", "car"],
            "hallucinated_objects": ["car"],
        }
        for image_id in range(10)
    ]
    comparison = paired_chair_comparison(
        baseline, candidate, bootstrap_samples=100, seed=7
    )
    assert comparison["metrics"]["chair_s"]["delta_points"] == 100.0
    assert comparison["metrics"]["chair_i"]["delta_points"] == 50.0
    assert comparison["metrics"]["object_recall"]["delta_points"] == 0.0
