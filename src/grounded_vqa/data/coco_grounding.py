from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from grounded_vqa.data.vqav2 import VQASample, coco_image_path
from grounded_vqa.evaluation.chair import (
    ALIAS_TO_CANONICAL,
    extract_coco_objects,
    normalize_phrase,
)


def article_for(category: str) -> str:
    return "an" if category[0].lower() in "aeiou" else "a"


def generate_grounding_records(
    payload: dict[str, Any],
    *,
    split: str,
    positive_count: int,
    negative_count: int,
    counting_count: int,
    min_area_ratio: float,
    max_count_answer: int,
    seed: int,
) -> list[dict[str, Any]]:
    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported split: {split}")
    if min(positive_count, negative_count, counting_count) < 0:
        raise ValueError("Requested task counts must be non-negative")

    rng = random.Random(seed)
    category_names = {int(item["id"]): str(item["name"]) for item in payload["categories"]}
    image_sizes = {
        int(item["id"]): (int(item["width"]), int(item["height"]))
        for item in payload["images"]
    }
    present_categories: dict[int, set[int]] = defaultdict(set)
    valid_counts: dict[int, Counter[int]] = defaultdict(Counter)
    for annotation in payload["annotations"]:
        if int(annotation.get("iscrowd", 0)):
            continue
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        present_categories[image_id].add(category_id)
        width, height = image_sizes[image_id]
        area_ratio = float(annotation["area"]) / float(width * height)
        if area_ratio >= min_area_ratio:
            valid_counts[image_id][category_id] += 1

    category_to_images: dict[int, list[int]] = defaultdict(list)
    for image_id, counts in valid_counts.items():
        for category_id in counts:
            category_to_images[category_id].append(image_id)
    for images in category_to_images.values():
        rng.shuffle(images)

    used_images: set[int] = set()
    records: list[dict[str, Any]] = []

    def add_balanced_present(task: str, requested: int) -> None:
        category_ids = list(category_names)
        rng.shuffle(category_ids)
        attempts = 0
        while sum(record["task_type"] == task for record in records) < requested:
            if attempts > requested * max(100, len(category_ids) * 4):
                raise RuntimeError(f"Unable to construct {requested} records for {task}")
            category_id = category_ids[attempts % len(category_ids)]
            attempts += 1
            candidates = category_to_images.get(category_id, [])
            while candidates and candidates[-1] in used_images:
                candidates.pop()
            if not candidates:
                continue
            image_id = candidates.pop()
            count = valid_counts[image_id][category_id]
            if task == "count" and not 1 <= count <= max_count_answer:
                continue
            used_images.add(image_id)
            category = category_names[category_id]
            if task == "existence_positive":
                question = f"Is {article_for(category)} {category} visible in the image?"
                answer = "yes"
            else:
                question = f"How many instances of {category} are visible in the image?"
                answer = str(count)
            records.append(
                {
                    "split": split,
                    "image_id": image_id,
                    "question": question,
                    "answer": answer,
                    "task_type": task,
                    "category_id": category_id,
                    "category": category,
                    "visible_count": count,
                }
            )

    add_balanced_present("count", counting_count)
    add_balanced_present("existence_positive", positive_count)

    all_categories = set(category_names)
    negative_candidates = list(image_sizes)
    rng.shuffle(negative_candidates)
    for image_id in negative_candidates:
        if sum(record["task_type"] == "existence_negative" for record in records) >= negative_count:
            break
        if image_id in used_images:
            continue
        absent = list(all_categories - present_categories.get(image_id, set()))
        if not absent:
            continue
        category_id = rng.choice(absent)
        category = category_names[category_id]
        used_images.add(image_id)
        records.append(
            {
                "split": split,
                "image_id": image_id,
                "question": f"Is {article_for(category)} {category} visible in the image?",
                "answer": "no",
                "task_type": "existence_negative",
                "category_id": category_id,
                "category": category,
                "visible_count": 0,
            }
        )
    if sum(record["task_type"] == "existence_negative" for record in records) != negative_count:
        raise RuntimeError(f"Unable to construct {negative_count} negative records")

    rng.shuffle(records)
    for index, record in enumerate(records):
        record["question_id"] = -(seed * 10_000_000 + index + 1)
    return records


def load_grounding_samples(path: Path, images_root: Path) -> list[VQASample]:
    samples: list[VQASample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            split = str(item["split"])
            answer = str(item["answer"])
            image_id = int(item["image_id"])
            image_path = coco_image_path(images_root, split, image_id)
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            samples.append(
                VQASample(
                    question_id=int(item["question_id"]),
                    image_id=image_id,
                    image_path=image_path,
                    question=str(item["question"]),
                    answers=(answer,) * 10,
                    answer_type="number" if item["task_type"] == "count" else "yes/no",
                    question_type=str(item["task_type"]),
                )
            )
    return samples


def grounding_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(records),
        "unique_images": len({int(record["image_id"]) for record in records}),
        "task_counts": dict(Counter(str(record["task_type"]) for record in records)),
        "answer_counts": dict(Counter(str(record["answer"]) for record in records)),
        "category_counts": dict(Counter(str(record["category"]) for record in records)),
    }


def generate_hard_negative_records(
    instances: dict[str, Any],
    captions: dict[str, Any],
    *,
    split: str,
    count: int,
    seed: int,
    max_per_category: int = 30,
) -> list[dict[str, Any]]:
    """Construct plausible absent-object questions without using benchmark predictions."""

    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported split: {split}")
    if count <= 0 or max_per_category <= 0:
        raise ValueError("count and max_per_category must be positive")
    rng = random.Random(seed)
    category_names = {
        int(category["id"]): str(category["name"])
        for category in instances["categories"]
    }
    canonical_to_id = {
        ALIAS_TO_CANONICAL[normalize_phrase(name)]: category_id
        for category_id, name in category_names.items()
    }
    present: dict[int, set[int]] = {
        int(image["id"]): set() for image in instances["images"]
    }
    for annotation in instances["annotations"]:
        present[int(annotation["image_id"])].add(int(annotation["category_id"]))
    captioned_images: set[int] = set()
    for annotation in captions["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id not in present:
            continue
        captioned_images.add(image_id)
        for mention in extract_coco_objects(str(annotation["caption"])):
            category_id = canonical_to_id.get(mention.canonical)
            if category_id is not None:
                present[image_id].add(category_id)

    frequency: Counter[int] = Counter()
    cooccurrence: dict[int, Counter[int]] = defaultdict(Counter)
    for categories in present.values():
        for source in categories:
            frequency[source] += 1
            for target in categories - {source}:
                cooccurrence[source][target] += 1
    all_categories = set(category_names)
    image_ids = sorted(captioned_images)
    rng.shuffle(image_ids)
    per_category: Counter[int] = Counter()
    records: list[dict[str, Any]] = []
    templates = (
        "Is {article} {category} visible in the image?",
        "Is there {article} {category} in this image?",
        "Can you see {article} {category} in the image?",
        "Does the image contain {article} {category}?",
    )
    total_images = max(len(present), 1)
    for image_id in image_ids:
        present_categories = present[image_id]
        if not present_categories:
            continue
        candidates = [
            category_id
            for category_id in all_categories - present_categories
            if per_category[category_id] < max_per_category
        ]
        if not candidates:
            continue

        context_categories = tuple(present_categories)

        def hardness(
            category_id: int,
            context: tuple[int, ...] = context_categories,
        ) -> tuple[float, float, int]:
            conditional = sum(
                cooccurrence[source][category_id] / frequency[source]
                for source in context
                if frequency[source]
            ) / len(context)
            popularity = frequency[category_id] / total_images
            return conditional, popularity, -category_id

        category_id = max(candidates, key=hardness)
        category = category_names[category_id]
        conditional, popularity, _ = hardness(category_id)
        template = templates[len(records) % len(templates)]
        records.append(
            {
                "split": split,
                "image_id": image_id,
                "question": template.format(
                    article=article_for(category), category=category
                ),
                "answer": "no",
                "task_type": "hard_existence_negative",
                "category_id": category_id,
                "category": category,
                "visible_count": 0,
                "conditional_cooccurrence": conditional,
                "category_popularity": popularity,
                "present_category_ids": sorted(present_categories),
            }
        )
        per_category[category_id] += 1
        if len(records) == count:
            break
    if len(records) != count:
        raise RuntimeError(
            f"Constructed {len(records)} of {count} hard negatives; "
            "increase max_per_category or provide more images"
        )
    rng.shuffle(records)
    for index, record in enumerate(records):
        record["question_id"] = -(seed * 10_000_000 + index + 1)
    return records
