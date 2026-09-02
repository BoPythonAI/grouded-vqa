from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from typing import Any

from grounded_vqa.data.answers import consensus_answer, normalize_answer

_EXISTENCE_PATTERN = re.compile(
    r"^(?:are there|is there|do you see|can you see|can you spot|is .* visible)\b",
    re.IGNORECASE,
)
_COUNT_PATTERN = re.compile(r"^how many\b", re.IGNORECASE)
_IRREGULAR_PLURALS = {
    "person": "people",
    "mouse": "mice",
    "sheep": "sheep",
    "knife": "knives",
}
_ALIASES = {
    "person": {"person", "people", "man", "men", "woman", "women", "boy", "girl", "child", "children", "kid", "kids"},
    "bicycle": {"bicycle", "bicycles", "bike", "bikes"},
    "motorcycle": {"motorcycle", "motorcycles", "motorbike", "motorbikes"},
    "airplane": {"airplane", "airplanes", "plane", "planes"},
    "couch": {"couch", "couches", "sofa", "sofas"},
    "tv": {"tv", "tvs", "television", "televisions"},
    "cell phone": {"cell phone", "cell phones", "phone", "phones"},
    "sports ball": {"sports ball", "sports balls", "ball", "balls"},
    "dining table": {"dining table", "dining tables", "table", "tables"},
    "potted plant": {"potted plant", "potted plants", "plant", "plants"},
    "hair drier": {"hair drier", "hair driers", "hair dryer", "hair dryers"},
}


def pluralize(category: str) -> str:
    if category in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[category]
    words = category.split()
    last = words[-1]
    if last.endswith(("s", "x", "z", "ch", "sh")):
        last += "es"
    elif last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
        last = last[:-1] + "ies"
    else:
        last += "s"
    return " ".join([*words[:-1], last])


def match_coco_category(
    question: str,
    category_names: dict[int, str],
) -> tuple[int, str] | None:
    text = question.lower()
    matches: list[tuple[int, int, int, str]] = []
    for category_id, category in category_names.items():
        aliases = set(_ALIASES.get(category, set()))
        aliases.update({category, pluralize(category)})
        for alias in aliases:
            match = re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text)
            if match:
                matches.append((match.start(), -len(alias), category_id, category))
    if not matches:
        return None
    _, _, category_id, category = min(matches)
    return category_id, category


def _balanced_select(
    candidates: list[dict[str, Any]],
    requested: int,
    used_images: set[int],
    rng: random.Random,
    *,
    allow_shortfall: bool = False,
) -> list[dict[str, Any]]:
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        buckets[int(candidate["category_id"])].append(candidate)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    category_ids = list(buckets)
    rng.shuffle(category_ids)
    selected: list[dict[str, Any]] = []
    while len(selected) < requested:
        made_progress = False
        for category_id in category_ids:
            bucket = buckets[category_id]
            while bucket:
                candidate = bucket.pop()
                image_id = int(candidate["image_id"])
                if image_id in used_images:
                    continue
                used_images.add(image_id)
                selected.append(candidate)
                made_progress = True
                break
            if len(selected) == requested:
                break
        if not made_progress:
            if allow_shortfall:
                break
            raise RuntimeError(
                f"Only found {len(selected)} unique-image candidates; requested {requested}"
            )
    return selected


def _expansion_question(task_type: str, category: str, variant: int) -> str:
    article = "an" if category[0].lower() in "aeiou" else "a"
    if task_type in {"existence_positive", "existence_negative"}:
        templates = (
            f"Is there {article} {category} in the image?",
            f"Can you see {article} {category} in the image?",
            f"Are there any {pluralize(category)} in the image?",
        )
    else:
        templates = (
            f"How many {pluralize(category)} are there?",
            f"How many {pluralize(category)} can you see?",
            f"How many {pluralize(category)} are in the image?",
        )
    return templates[variant % len(templates)]


def _build_expansion_candidates(
    task_type: str,
    sources: dict[int, list[dict[str, Any]]],
    category_names: dict[int, str],
    image_ids: list[int],
    visible_counts: dict[int, Counter[int]],
    present_categories: dict[int, set[int]],
    max_count_answer: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for category_id, source_items in sources.items():
        category = category_names[category_id]
        candidate_images = image_ids.copy()
        rng.shuffle(candidate_images)
        category_candidates = 0
        for variant, image_id in enumerate(candidate_images):
            visible_count = visible_counts[image_id][category_id]
            if task_type == "existence_positive" and visible_count == 0:
                continue
            if (
                task_type == "existence_negative"
                and category_id in present_categories.get(image_id, set())
            ):
                continue
            if task_type == "count" and not 1 <= visible_count <= max_count_answer:
                continue
            source = source_items[variant % len(source_items)]
            answer = (
                "yes"
                if task_type == "existence_positive"
                else "no"
                if task_type == "existence_negative"
                else str(visible_count)
            )
            candidates.append(
                {
                    "split": "train",
                    "image_id": image_id,
                    "question": _expansion_question(task_type, category, variant),
                    "answer": answer,
                    "task_type": task_type,
                    "category_id": category_id,
                    "category": category,
                    "visible_count": visible_count,
                    "source_error": f"{source['source_error']}_category_expansion",
                    "source_question_id": source["source_question_id"],
                    "source_prediction": source["source_prediction"],
                    "source_record_kind": "category_expansion",
                }
            )
            category_candidates += 1
            if category_candidates >= 200:
                break
    return candidates


def generate_error_grounding_records(
    instances: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    positive_count: int,
    negative_count: int,
    counting_count: int,
    min_area_ratio: float,
    max_count_answer: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    category_names = {
        int(item["id"]): str(item["name"]) for item in instances["categories"]
    }
    image_sizes = {
        int(item["id"]): (int(item["width"]), int(item["height"]))
        for item in instances["images"]
    }
    visible_counts: dict[int, Counter[int]] = defaultdict(Counter)
    present_categories: dict[int, set[int]] = defaultdict(set)
    for annotation in instances["annotations"]:
        if int(annotation.get("iscrowd", 0)):
            continue
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        present_categories[image_id].add(category_id)
        width, height = image_sizes[image_id]
        if float(annotation["area"]) / float(width * height) < min_area_ratio:
            continue
        visible_counts[image_id][category_id] += 1

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    error_sources: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    rejection_counts: Counter[str] = Counter()
    for item in predictions:
        question = str(item["question"])
        matched = match_coco_category(question, category_names)
        if matched is None:
            rejection_counts["no_coco_category"] += 1
            continue
        category_id, category = matched
        image_id = int(item["image_id"])
        visible_count = visible_counts[image_id][category_id]
        target = consensus_answer(tuple(str(answer) for answer in item["references"]))
        prediction = normalize_answer(str(item["prediction"]))
        task_type: str | None = None
        source_error: str | None = None

        if _EXISTENCE_PATTERN.search(question):
            if target == "no" and prediction == "yes":
                error_sources["existence_negative"][category_id].append(
                    {
                        "source_error": "false_yes",
                        "source_question_id": int(item["question_id"]),
                        "source_prediction": prediction,
                    }
                )
                if category_id not in present_categories.get(image_id, set()):
                    task_type = "existence_negative"
                    source_error = "false_yes"
                else:
                    rejection_counts["coco_vqa_existence_disagreement"] += 1
            elif target == "yes" and prediction == "no":
                error_sources["existence_positive"][category_id].append(
                    {
                        "source_error": "false_no",
                        "source_question_id": int(item["question_id"]),
                        "source_prediction": prediction,
                    }
                )
                if visible_count > 0:
                    task_type = "existence_positive"
                    source_error = "false_no"
                else:
                    rejection_counts["coco_vqa_existence_disagreement"] += 1
            else:
                rejection_counts["unverified_existence_error"] += 1
        elif _COUNT_PATTERN.search(question):
            try:
                target_count = int(target)
            except ValueError:
                rejection_counts["non_integer_count_target"] += 1
                continue
            if not 1 <= target_count <= max_count_answer:
                rejection_counts["count_target_out_of_range"] += 1
            elif prediction == target:
                rejection_counts["count_prediction_correct"] += 1
            else:
                error_sources["count"][category_id].append(
                    {
                        "source_error": "count_error",
                        "source_question_id": int(item["question_id"]),
                        "source_prediction": prediction,
                    }
                )
                if visible_count != target_count:
                    rejection_counts["coco_vqa_count_disagreement"] += 1
                else:
                    task_type = "count"
                    source_error = "count_error"
        else:
            rejection_counts["unsupported_question_form"] += 1

        if task_type is None or source_error is None:
            continue
        candidates[task_type].append(
            {
                "split": "train",
                "image_id": image_id,
                "question": question,
                "answer": target,
                "task_type": task_type,
                "category_id": category_id,
                "category": category,
                "visible_count": visible_count,
                "source_error": source_error,
                "source_question_id": int(item["question_id"]),
                "source_prediction": prediction,
                "source_record_kind": "direct_verified_error",
            }
        )

    rng = random.Random(seed)
    used_images: set[int] = set()
    requests = (
        ("existence_positive", positive_count),
        ("existence_negative", negative_count),
        ("count", counting_count),
    )
    selected_by_task: dict[str, list[dict[str, Any]]] = {}
    for task_type, requested in requests:
        selected_by_task[task_type] = _balanced_select(
            candidates[task_type],
            requested,
            used_images,
            rng,
            allow_shortfall=True,
        )
    image_ids = list(image_sizes)
    for task_type, requested in requests:
        remaining = requested - len(selected_by_task[task_type])
        if remaining <= 0:
            continue
        expansion_candidates = _build_expansion_candidates(
            task_type,
            error_sources[task_type],
            category_names,
            image_ids,
            visible_counts,
            present_categories,
            max_count_answer,
            rng,
        )
        selected_by_task[task_type].extend(
            _balanced_select(expansion_candidates, remaining, used_images, rng)
        )
    selected = [record for task_type, _ in requests for record in selected_by_task[task_type]]
    rng.shuffle(selected)
    for index, record in enumerate(selected):
        record["question_id"] = -(seed * 10_000_000 + index + 1)

    audit = {
        "prediction_count": len(predictions),
        "candidate_counts": {key: len(value) for key, value in candidates.items()},
        "error_source_counts": {
            task_type: sum(len(items) for items in categories.values())
            for task_type, categories in error_sources.items()
        },
        "error_source_category_counts": {
            task_type: len(categories)
            for task_type, categories in error_sources.items()
        },
        "rejection_counts": dict(rejection_counts),
        "selected_count": len(selected),
        "unique_images": len(used_images),
        "selected_task_counts": dict(Counter(item["task_type"] for item in selected)),
        "selected_category_counts": dict(Counter(item["category"] for item in selected)),
        "selected_source_error_counts": dict(
            Counter(item["source_error"] for item in selected)
        ),
        "selected_record_kind_counts": dict(
            Counter(item["source_record_kind"] for item in selected)
        ),
    }
    return selected, audit


def nested_task_subset(
    records: list[dict[str, Any]],
    *,
    positive_count: int,
    negative_count: int,
    counting_count: int,
) -> list[dict[str, Any]]:
    requested = {
        "existence_positive": positive_count,
        "existence_negative": negative_count,
        "count": counting_count,
    }
    selected: list[dict[str, Any]] = []
    seen: Counter[str] = Counter()
    for record in records:
        task_type = str(record["task_type"])
        if seen[task_type] < requested[task_type]:
            selected.append(record)
            seen[task_type] += 1
    if dict(seen) != {key: value for key, value in requested.items() if value}:
        raise RuntimeError(f"Unable to make requested nested subset: selected={dict(seen)}")
    return selected
