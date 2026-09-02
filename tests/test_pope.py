import json
from pathlib import Path

import pytest

from grounded_vqa.data.pope import load_pope
from grounded_vqa.evaluation.pope import (
    compute_pope_metrics,
    paired_pope_comparison,
    parse_official_pope_answer,
    parse_strict_yes_no,
)


def test_official_and_strict_pope_parsers_expose_permissive_default() -> None:
    assert parse_official_pope_answer("No, there is not.") == "no"
    assert parse_official_pope_answer("I cannot tell.") == "yes"
    assert parse_strict_yes_no("yes") == "yes"
    assert parse_strict_yes_no("I cannot tell") is None


def test_compute_pope_metrics_tracks_confusion_and_invalid() -> None:
    metrics = compute_pope_metrics(
        ["yes", "yes", "no", "no"],
        ["yes", "no", "yes", None],
    )
    assert metrics["tp"] == 1
    assert metrics["fn"] == 1
    assert metrics["fp"] == 1
    assert metrics["invalid"] == 1
    assert metrics["accuracy"] == 0.25
    assert metrics["invalid_ratio"] == 0.25


def test_load_pope_validates_image_join(tmp_path: Path) -> None:
    image = tmp_path / "COCO_val2014_000000000001.jpg"
    image.write_bytes(b"image-placeholder")
    source = tmp_path / "coco_pope_random.json"
    source.write_text(
        json.dumps(
            {
                "question_id": 1,
                "image": image.name,
                "text": "Is there a cat in the image?",
                "label": "no",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    examples = load_pope(source, tmp_path)
    assert examples[0].image_path == image
    assert examples[0].label == "no"


def test_load_pope_rejects_invalid_label(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image-placeholder")
    source = tmp_path / "pope.json"
    source.write_text(
        json.dumps(
            {"question_id": 1, "image": image.name, "text": "Question?", "label": "maybe"}
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid POPE label"):
        load_pope(source, tmp_path)


def test_paired_pope_comparison_reports_direction_and_interval() -> None:
    comparison = paired_pope_comparison(
        ["yes", "no", "yes", "no"],
        ["no", "no", "yes", "yes"],
        ["yes", "no", "yes", "no"],
        bootstrap_samples=100,
    )
    assert comparison["improved"] == 2
    assert comparison["worsened"] == 0
    assert comparison["accuracy_delta_points"] == 50.0
    assert comparison["bootstrap_95_ci_points"][0] >= 0.0
