from grounded_vqa.data.answer_routing import (
    choose_yes_no,
    is_valid_yes_no_answer,
    is_yes_no_question,
)


def test_yes_no_router_accepts_binary_questions() -> None:
    assert is_yes_no_question("Is there a dog in the image?")
    assert is_yes_no_question("Does the bus have any passengers?")
    assert is_yes_no_question("Were they playing tennis?")


def test_yes_no_router_rejects_open_and_disjunctive_questions() -> None:
    assert not is_yes_no_question("What color is the bus?")
    assert not is_yes_no_question("Is the light red or green?")
    assert not is_yes_no_question("Can you identify this job?")
    assert not is_yes_no_question("Do you know what kind of bird this is?")
    assert not is_yes_no_question("Is What color is the car?")


def test_yes_no_answer_validation_uses_vqa_normalization() -> None:
    assert is_valid_yes_no_answer("YES!")
    assert is_valid_yes_no_answer(" no. ")
    assert not is_valid_yes_no_answer("blue")


def test_choose_yes_no_prefers_lower_nll_and_breaks_ties() -> None:
    assert choose_yes_no(0.5, 0.8) == "yes"
    assert choose_yes_no(0.9, 0.2) == "no"
    assert choose_yes_no(0.4, 0.4) == "yes"
