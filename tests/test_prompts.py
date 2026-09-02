from grounded_vqa.models.loading import format_prompt


def test_default_blip2_prompt() -> None:
    assert format_prompt("blip2", "How many cats? ") == (
        "Question: How many cats? Answer:"
    )


def test_default_instructblip_prompt_is_unchanged() -> None:
    assert format_prompt("instructblip", " Is it raining? ") == "Is it raining?"


def test_short_instructblip_prompt_requests_vqa_answer_format() -> None:
    prompt = format_prompt("instructblip", "How many cats?", "short")
    assert prompt.startswith("How many cats?\n")
    assert "single number" in prompt
    assert "Do not use a full sentence." in prompt
