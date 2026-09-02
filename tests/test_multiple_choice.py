from pathlib import Path

from grounded_vqa.data.multiple_choice import (
    build_multiple_choice_examples,
    format_multiple_choice_prompt,
    parse_multiple_choice_prediction,
)
from grounded_vqa.data.vqav2 import VQASample


def sample(
    question_id: int,
    answer: str,
    *,
    answer_type: str = "other",
    question_type: str = "what color",
) -> VQASample:
    return VQASample(
        question_id=question_id,
        image_id=question_id,
        image_path=Path(f"{question_id}.jpg"),
        question="What color is it?",
        answers=(answer,) * 10,
        answer_type=answer_type,
        question_type=question_type,
        multiple_choice_answer=answer,
    )


def test_build_multiple_choice_is_deterministic_and_balanced() -> None:
    samples = [sample(index, answer) for index, answer in enumerate(
        ["red", "blue", "green", "yellow", "black"], start=1
    )]
    first, audit = build_multiple_choice_examples(samples, seed=7)
    second, _ = build_multiple_choice_examples(samples, seed=7)

    assert first == second
    assert audit["built_examples"] == 5
    assert all(len(example.options) == 4 for example in first)
    assert all(example.options[example.correct_index] == example.correct_answer for example in first)


def test_yes_no_examples_use_two_options() -> None:
    examples, _ = build_multiple_choice_examples(
        [
            sample(1, "yes", answer_type="yes/no", question_type="is"),
            sample(2, "no", answer_type="yes/no", question_type="is"),
        ]
    )
    assert {example.options for example in examples} <= {("yes", "no"), ("no", "yes")}


def test_prompt_and_prediction_parser() -> None:
    example = build_multiple_choice_examples(
        [sample(index, answer) for index, answer in enumerate(
            ["red", "blue", "green", "yellow"], start=1
        )]
    )[0][0]
    prompt = format_multiple_choice_prompt(example)
    assert "Options:" in prompt
    assert "Answer using only the option letter." in prompt
    assert parse_multiple_choice_prediction("The answer is (B).", ("red", "blue")) == 1
    assert parse_multiple_choice_prediction("blue", ("red", "blue")) == 1
    assert parse_multiple_choice_prediction("unknown", ("red", "blue")) is None

