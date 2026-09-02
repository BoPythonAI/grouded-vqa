from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VQASample:
    question_id: int
    image_id: int
    image_path: Path
    question: str
    answers: tuple[str, ...]
    answer_type: str | None
    question_type: str | None
    multiple_choice_answer: str | None = None


def coco_image_path(images_root: Path, split: str, image_id: int) -> Path:
    year_split = {"train": "train2014", "val": "val2014"}[split]
    return images_root / year_split / f"COCO_{year_split}_{image_id:012d}.jpg"


def load_vqav2(
    questions_path: Path,
    annotations_path: Path,
    images_root: Path,
    split: str,
    *,
    require_images: bool = True,
) -> list[VQASample]:
    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported split: {split}")

    with questions_path.open("r", encoding="utf-8") as handle:
        question_payload: dict[str, Any] = json.load(handle)
    with annotations_path.open("r", encoding="utf-8") as handle:
        annotation_payload: dict[str, Any] = json.load(handle)

    annotations = {
        int(item["question_id"]): item for item in annotation_payload["annotations"]
    }
    samples: list[VQASample] = []
    for item in question_payload["questions"]:
        question_id = int(item["question_id"])
        annotation = annotations.get(question_id)
        if annotation is None:
            raise ValueError(f"Missing annotation for question_id={question_id}")
        image_id = int(item["image_id"])
        image_path = coco_image_path(images_root, split, image_id)
        if require_images and not image_path.is_file():
            raise FileNotFoundError(image_path)
        samples.append(
            VQASample(
                question_id=question_id,
                image_id=image_id,
                image_path=image_path,
                question=str(item["question"]),
                answers=tuple(answer["answer"] for answer in annotation["answers"]),
                answer_type=annotation.get("answer_type"),
                question_type=annotation.get("question_type"),
                multiple_choice_answer=annotation.get("multiple_choice_answer"),
            )
        )
    return samples
