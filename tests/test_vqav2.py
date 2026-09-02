import json
from pathlib import Path

import pytest

from grounded_vqa.data.vqav2 import coco_image_path, load_vqav2


def test_coco_image_path() -> None:
    path = coco_image_path(Path("/data"), "val", 42)
    assert path == Path("/data/val2014/COCO_val2014_000000000042.jpg")


def test_load_vqav2_without_images(tmp_path: Path) -> None:
    questions = tmp_path / "questions.json"
    annotations = tmp_path / "annotations.json"
    questions.write_text(
        json.dumps(
            {"questions": [{"question_id": 7, "image_id": 42, "question": "How many?"}]}
        ),
        encoding="utf-8",
    )
    annotations.write_text(
        json.dumps(
            {
                "annotations": [
                    {
                        "question_id": 7,
                        "answers": [{"answer": "2"}] * 10,
                        "answer_type": "number",
                        "question_type": "how many",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    samples = load_vqav2(
        questions,
        annotations,
        tmp_path / "images",
        "val",
        require_images=False,
    )
    assert len(samples) == 1
    assert samples[0].question_id == 7
    assert samples[0].answers == ("2",) * 10


def test_load_vqav2_rejects_missing_annotation(tmp_path: Path) -> None:
    questions = tmp_path / "questions.json"
    annotations = tmp_path / "annotations.json"
    questions.write_text(
        json.dumps({"questions": [{"question_id": 7, "image_id": 42, "question": "Q?"}]}),
        encoding="utf-8",
    )
    annotations.write_text(json.dumps({"annotations": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing annotation"):
        load_vqav2(
            questions,
            annotations,
            tmp_path / "images",
            "val",
            require_images=False,
        )

