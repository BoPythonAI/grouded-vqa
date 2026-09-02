from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

from grounded_vqa.data.answers import consensus_answer, normalize_answer
from grounded_vqa.data.vqav2 import VQASample

OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class MultipleChoiceExample:
    question_id: int
    image_id: int
    image_path: str
    question: str
    options: tuple[str, ...]
    correct_index: int
    correct_answer: str
    answer_type: str
    question_type: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["options"] = list(self.options)
        return payload


def build_multiple_choice_examples(
    samples: list[VQASample],
    *,
    max_options: int = 4,
    seed: int = 42,
) -> tuple[list[MultipleChoiceExample], dict[str, object]]:
    if max_options < 2 or max_options > len(OPTION_LETTERS):
        raise ValueError("max_options must be between 2 and 26")

    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    by_answer_type: dict[str, set[str]] = defaultdict(set)
    global_answers: set[str] = set()
    correct_answers: dict[int, str] = {}
    for sample in samples:
        raw_answer = sample.multiple_choice_answer or consensus_answer(sample.answers)
        correct = normalize_answer(raw_answer)
        if not correct:
            continue
        answer_type = sample.answer_type or "unknown"
        question_type = sample.question_type or "unknown"
        correct_answers[sample.question_id] = correct
        grouped[(answer_type, question_type)].add(correct)
        by_answer_type[answer_type].add(correct)
        global_answers.add(correct)

    grouped_options = {key: tuple(sorted(values)) for key, values in grouped.items()}
    answer_type_options = {
        key: tuple(sorted(values)) for key, values in by_answer_type.items()
    }
    all_options = tuple(sorted(global_answers))

    examples: list[MultipleChoiceExample] = []
    skipped = 0
    option_counts: Counter[int] = Counter()
    correct_positions: Counter[int] = Counter()
    for sample in samples:
        correct = correct_answers.get(sample.question_id)
        if correct is None:
            skipped += 1
            continue
        answer_type = sample.answer_type or "unknown"
        question_type = sample.question_type or "unknown"
        desired_options = 2 if answer_type == "yes/no" else max_options
        needed = desired_options - 1
        excluded = {normalize_answer(answer) for answer in sample.answers}
        excluded.add(correct)

        candidates: list[str] = []
        seen: set[str] = set()
        generator = random.Random(seed * 1_000_003 + sample.question_id)
        if answer_type == "yes/no":
            opposite = "no" if correct == "yes" else "yes"
            candidates = [opposite] if correct in {"yes", "no"} else candidates
        else:
            pools = [
                grouped_options[(answer_type, question_type)],
                answer_type_options[answer_type],
                all_options,
            ]
            for pool in pools:
                attempts = min(len(pool) * 2, max(20, needed * 20))
                for _ in range(attempts):
                    candidate = pool[generator.randrange(len(pool))]
                    if candidate not in excluded and candidate not in seen:
                        candidates.append(candidate)
                        seen.add(candidate)
                        if len(candidates) >= needed:
                            break
                if len(candidates) < needed and len(pool) <= 50:
                    for candidate in pool:
                        if candidate not in excluded and candidate not in seen:
                            candidates.append(candidate)
                            seen.add(candidate)
                            if len(candidates) >= needed:
                                break
                if len(candidates) >= needed:
                    break

        if len(candidates) < needed:
            skipped += 1
            continue
        distractors = candidates[:needed]
        options = distractors + [correct]
        generator.shuffle(options)
        correct_index = options.index(correct)
        option_counts[len(options)] += 1
        correct_positions[correct_index] += 1
        examples.append(
            MultipleChoiceExample(
                question_id=sample.question_id,
                image_id=sample.image_id,
                image_path=str(sample.image_path),
                question=sample.question,
                options=tuple(options),
                correct_index=correct_index,
                correct_answer=correct,
                answer_type=answer_type,
                question_type=question_type,
            )
        )

    audit: dict[str, object] = {
        "source_samples": len(samples),
        "built_examples": len(examples),
        "skipped_examples": skipped,
        "max_options": max_options,
        "seed": seed,
        "option_count_distribution": dict(sorted(option_counts.items())),
        "correct_position_distribution": {
            OPTION_LETTERS[index]: count for index, count in sorted(correct_positions.items())
        },
    }
    return examples, audit


def format_multiple_choice_prompt(example: MultipleChoiceExample) -> str:
    option_lines = "\n".join(
        f"({OPTION_LETTERS[index]}) {option}" for index, option in enumerate(example.options)
    )
    return (
        f"{example.question}\nOptions:\n{option_lines}\n"
        "Answer using only the option letter."
    )


def parse_multiple_choice_prediction(prediction: str, options: tuple[str, ...]) -> int | None:
    text = prediction.strip()
    valid_letters = OPTION_LETTERS[: len(options)]
    letter_match = re.search(
        rf"(?<![A-Za-z])([{re.escape(valid_letters)}])(?![A-Za-z])",
        text.upper(),
    )
    if letter_match:
        return OPTION_LETTERS.index(letter_match.group(1))

    normalized = normalize_answer(text)
    matches = [
        index for index, option in enumerate(options) if normalize_answer(option) == normalized
    ]
    return matches[0] if len(matches) == 1 else None
