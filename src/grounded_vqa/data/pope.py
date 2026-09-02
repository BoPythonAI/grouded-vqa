from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PopeExample:
    question_id: int
    image_path: Path
    question: str
    label: str


def load_pope(path: Path, image_root: Path) -> list[PopeExample]:
    """Load an official POPE JSONL split and validate its COCO image joins."""

    examples: list[PopeExample] = []
    seen_ids: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            required = {"question_id", "image", "text", "label"}
            if not isinstance(item, dict) or not required.issubset(item):
                raise ValueError(f"Invalid POPE row at {path}:{line_number}")
            question_id = int(item["question_id"])
            if question_id in seen_ids:
                raise ValueError(f"Duplicate POPE question_id={question_id} in {path}")
            seen_ids.add(question_id)
            image_name = Path(str(item["image"])).name
            image_path = image_root / image_name
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            label = str(item["label"]).strip().lower()
            if label not in {"yes", "no"}:
                raise ValueError(f"Invalid POPE label at {path}:{line_number}: {label}")
            examples.append(
                PopeExample(
                    question_id=question_id,
                    image_path=image_path,
                    question=str(item["text"]).strip(),
                    label=label,
                )
            )
    if not examples:
        raise ValueError(f"No POPE examples found in {path}")
    return examples
