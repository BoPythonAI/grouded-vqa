from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
IRREGULAR_SINGULARS = {
    "children": "child",
    "feet": "foot",
    "geese": "goose",
    "knives": "knife",
    "men": "man",
    "mice": "mouse",
    "oxen": "ox",
    "people": "person",
    "teeth": "tooth",
    "women": "woman",
}
SINGULAR_EXCEPTIONS = {
    "bus",
    "glass",
    "grass",
    "news",
    "scissors",
    "sheep",
    "sports",
}
ANIMAL_WORDS = {
    "animal",
    "bear",
    "bird",
    "cat",
    "cow",
    "cub",
    "dog",
    "elephant",
    "giraffe",
    "horse",
    "sheep",
    "zebra",
}


@dataclass(frozen=True)
class ObjectMention:
    surface: str
    canonical: str
    token_index: int


def singularize_token(token: str) -> str:
    token = token.lower()
    if token in IRREGULAR_SINGULARS:
        return IRREGULAR_SINGULARS[token]
    if token in SINGULAR_EXCEPTIONS or len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("ches", "shes", "xes", "zes", "ses")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def normalize_phrase(phrase: str) -> str:
    return " ".join(singularize_token(token) for token in TOKEN_RE.findall(phrase.lower()))


def load_synonym_groups() -> tuple[tuple[str, ...], ...]:
    source = files("grounded_vqa.evaluation").joinpath("chair_synonyms.txt")
    groups: list[tuple[str, ...]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        aliases = tuple(
            dict.fromkeys(normalize_phrase(alias.strip()) for alias in line.split(",") if alias.strip())
        )
        if aliases:
            groups.append(aliases)
    return tuple(groups)


SYNONYM_GROUPS = load_synonym_groups()
ALIAS_TO_CANONICAL = {
    alias: aliases[0] for aliases in SYNONYM_GROUPS for alias in aliases
}
CANONICAL_OBJECTS = frozenset(aliases[0] for aliases in SYNONYM_GROUPS)


def extract_coco_objects(caption: str) -> list[ObjectMention]:
    raw_tokens = TOKEN_RE.findall(caption.lower())
    tokens = [singularize_token(token) for token in raw_tokens]
    mentions: list[ObjectMention] = []
    index = 0
    while index < len(tokens):
        phrase = " ".join(tokens[index : index + 2])
        if tokens[index] in {"baby", "adult"} and index + 1 < len(tokens):
            if tokens[index + 1] in ANIMAL_WORDS:
                phrase = tokens[index + 1]
                consumed = 2
            else:
                consumed = 1
        elif tokens[index] == "passenger" and index + 1 < len(tokens):
            if tokens[index + 1] in {"jet", "train"}:
                phrase = tokens[index + 1]
                consumed = 2
            else:
                consumed = 1
        elif phrase in ALIAS_TO_CANONICAL:
            consumed = 2
        else:
            phrase = tokens[index]
            consumed = 1
        canonical = ALIAS_TO_CANONICAL.get(phrase)
        if canonical is not None:
            mentions.append(ObjectMention(phrase, canonical, index))
        index += consumed

    if any(mention.canonical == "toilet" for mention in mentions):
        mentions = [
            mention
            for mention in mentions
            if not (mention.surface == "seat" and mention.canonical == "chair")
        ]
    return mentions


def _read_annotation(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def build_ground_truth_objects(
    image_ids: Iterable[int],
    annotation_root: Path,
    splits: tuple[str, ...] = ("val",),
) -> dict[int, set[str]]:
    selected = {int(image_id) for image_id in image_ids}
    ground_truth = {image_id: set() for image_id in selected}
    for split in splits:
        instances = _read_annotation(annotation_root / f"instances_{split}2014.json")
        category_names = {
            int(category["id"]): normalize_phrase(str(category["name"]))
            for category in instances["categories"]
        }
        for annotation in instances["annotations"]:
            image_id = int(annotation["image_id"])
            if image_id not in selected:
                continue
            category = category_names[int(annotation["category_id"])]
            canonical = ALIAS_TO_CANONICAL.get(category)
            if canonical is not None:
                ground_truth[image_id].add(canonical)

        captions = _read_annotation(annotation_root / f"captions_{split}2014.json")
        for annotation in captions["annotations"]:
            image_id = int(annotation["image_id"])
            if image_id in selected:
                ground_truth[image_id].update(
                    mention.canonical
                    for mention in extract_coco_objects(str(annotation["caption"]))
                )
    return ground_truth


def compute_chair_metrics(
    captions: Iterable[dict[str, Any]],
    ground_truth: dict[int, set[str]],
) -> dict[str, Any]:
    rows = list(captions)
    if not rows:
        raise ValueError("At least one caption is required")
    hallucinated_caption_count = 0
    hallucinated_mention_count = 0
    object_mention_count = 0
    matched_unique_count = 0
    ground_truth_unique_count = 0
    word_count = 0
    zero_object_caption_count = 0
    details: list[dict[str, Any]] = []
    for row in rows:
        image_id = int(row["image_id"])
        caption = str(row["caption"])
        if image_id not in ground_truth:
            raise KeyError(f"Missing ground-truth objects for image {image_id}")
        mentions = extract_coco_objects(caption)
        gt_objects = ground_truth[image_id]
        hallucinated = [mention for mention in mentions if mention.canonical not in gt_objects]
        generated_unique = {mention.canonical for mention in mentions}
        matched_unique = generated_unique & gt_objects
        hallucinated_caption_count += int(bool(hallucinated))
        hallucinated_mention_count += len(hallucinated)
        object_mention_count += len(mentions)
        matched_unique_count += len(matched_unique)
        ground_truth_unique_count += len(gt_objects)
        word_count += len(TOKEN_RE.findall(caption))
        zero_object_caption_count += int(not mentions)
        details.append(
            {
                "image_id": image_id,
                "caption": caption,
                "ground_truth_objects": sorted(gt_objects),
                "generated_objects": [mention.canonical for mention in mentions],
                "hallucinated_objects": [mention.canonical for mention in hallucinated],
                "chair_s": int(bool(hallucinated)),
                "chair_i": len(hallucinated) / len(mentions) if mentions else 0.0,
            }
        )
    count = len(rows)
    return {
        "caption_count": count,
        "chair_s": hallucinated_caption_count / count,
        "chair_i": (
            hallucinated_mention_count / object_mention_count if object_mention_count else 0.0
        ),
        "object_recall": (
            matched_unique_count / ground_truth_unique_count if ground_truth_unique_count else 0.0
        ),
        "hallucinated_caption_count": hallucinated_caption_count,
        "hallucinated_object_mention_count": hallucinated_mention_count,
        "object_mention_count": object_mention_count,
        "mean_caption_words": word_count / count,
        "mean_object_mentions": object_mention_count / count,
        "zero_object_caption_ratio": zero_object_caption_count / count,
        "per_caption": details,
        "implementation": (
            "Python 3 port of the BSD-licensed CHAIR object rules; regex tokenization and "
            "deterministic singularization replace the original Python 2 pattern/nltk tokenizer"
        ),
    }


def _detail_arrays(details: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "chair_s": np.asarray([int(bool(row["hallucinated_objects"])) for row in details]),
        "hallucinated": np.asarray(
            [len(row["hallucinated_objects"]) for row in details], dtype=np.int64
        ),
        "mentions": np.asarray(
            [len(row["generated_objects"]) for row in details], dtype=np.int64
        ),
        "matched": np.asarray(
            [
                len(set(row["generated_objects"]) & set(row["ground_truth_objects"]))
                for row in details
            ],
            dtype=np.int64,
        ),
        "ground_truth": np.asarray(
            [len(row["ground_truth_objects"]) for row in details], dtype=np.int64
        ),
    }


def _aggregate_detail_arrays(arrays: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    mention_count = int(np.sum(arrays["mentions"][indices]))
    ground_truth_count = int(np.sum(arrays["ground_truth"][indices]))
    return np.asarray(
        [
            float(np.mean(arrays["chair_s"][indices])),
            float(np.sum(arrays["hallucinated"][indices]) / mention_count)
            if mention_count
            else 0.0,
            float(np.sum(arrays["matched"][indices]) / ground_truth_count)
            if ground_truth_count
            else 0.0,
        ]
    )


def paired_chair_comparison(
    baseline_details: list[dict[str, Any]],
    candidate_details: list[dict[str, Any]],
    *,
    bootstrap_samples: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    if len(baseline_details) != len(candidate_details) or not baseline_details:
        raise ValueError("Paired CHAIR inputs must have equal non-zero length")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    baseline_ids = [int(row["image_id"]) for row in baseline_details]
    candidate_ids = [int(row["image_id"]) for row in candidate_details]
    if baseline_ids != candidate_ids:
        raise ValueError("Paired CHAIR image IDs do not align")
    baseline_arrays = _detail_arrays(baseline_details)
    candidate_arrays = _detail_arrays(candidate_details)
    if not np.array_equal(baseline_arrays["ground_truth"], candidate_arrays["ground_truth"]):
        raise ValueError("Paired CHAIR ground-truth annotations do not align")

    full_indices = np.arange(len(baseline_details))
    baseline = _aggregate_detail_arrays(baseline_arrays, full_indices)
    candidate = _aggregate_detail_arrays(candidate_arrays, full_indices)
    generator = np.random.default_rng(seed)
    bootstrap_delta = np.empty((bootstrap_samples, 3), dtype=np.float64)
    for index in range(bootstrap_samples):
        sampled = generator.integers(0, len(full_indices), size=len(full_indices))
        bootstrap_delta[index] = 100.0 * (
            _aggregate_detail_arrays(candidate_arrays, sampled)
            - _aggregate_detail_arrays(baseline_arrays, sampled)
        )
    metric_names = ("chair_s", "chair_i", "object_recall")
    return {
        "count": len(baseline_details),
        "metrics": {
            name: {
                "baseline": float(baseline[metric_index]),
                "candidate": float(candidate[metric_index]),
                "delta_points": float(100.0 * (candidate[metric_index] - baseline[metric_index])),
                "bootstrap_95_ci_points": [
                    float(value)
                    for value in np.quantile(
                        bootstrap_delta[:, metric_index], [0.025, 0.975]
                    )
                ],
            }
            for metric_index, name in enumerate(metric_names)
        },
    }
