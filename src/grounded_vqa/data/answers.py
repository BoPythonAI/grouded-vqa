from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Sequence

_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldnt": "couldn't",
    "couldve": "could've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "idve": "i'd've",
    "im": "i'm",
    "ive": "i've",
    "isnt": "isn't",
    "itd": "it'd",
    "itll": "it'll",
    "lets": "let's",
    "mightnt": "mightn't",
    "mustnt": "mustn't",
    "shant": "shan't",
    "shed": "she'd",
    "shell": "she'll",
    "shes": "she's",
    "shouldnt": "shouldn't",
    "somebodyd": "somebody'd",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "wed": "we'd",
    "well": "we'll",
    "were": "we're",
    "werent": "weren't",
    "weve": "we've",
    "whatd": "what'd",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyd": "why'd",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldnt": "wouldn't",
    "yall": "y'all",
    "youd": "you'd",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

_NUMBER_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

_ARTICLES = {"a", "an", "the"}
_PUNCTUATION = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]
_COMMA_BETWEEN_DIGITS = re.compile(r"(\d)(,)(\d)")
_PERIOD_NOT_DECIMAL = re.compile(r"(?<!\d)\.(?!\d)")


def normalize_answer(answer: str) -> str:
    """Normalize an answer using the official VQA evaluation conventions."""

    text = answer.replace("\n", " ").replace("\t", " ").strip().lower()
    for punct in _PUNCTUATION:
        if punct == "," and _COMMA_BETWEEN_DIGITS.search(text):
            text = text.replace(punct, "")
        else:
            text = text.replace(punct, " ")
    text = _PERIOD_NOT_DECIMAL.sub("", text)

    normalized: list[str] = []
    for token in text.split():
        token = _NUMBER_MAP.get(token, token)
        if token in _ARTICLES:
            continue
        normalized.append(_CONTRACTIONS.get(token, token))
    return " ".join(normalized)


def vqa_soft_accuracy(prediction: str, references: Sequence[str]) -> float:
    """Compute official leave-one-annotator-out VQA soft accuracy.

    VQAv2 normally supplies ten answers. The implementation also supports
    smaller reference lists for unit tests and diagnostic subsets.
    """

    if not references:
        raise ValueError("At least one reference answer is required")

    prediction = normalize_answer(prediction)
    references = [normalize_answer(answer) for answer in references]
    scores: list[float] = []
    for index in range(len(references)):
        other_answers = references[:index] + references[index + 1 :]
        matches = sum(answer == prediction for answer in other_answers)
        scores.append(min(1.0, matches / 3.0))
    return sum(scores) / len(scores)


def consensus_answer(references: Sequence[str]) -> str:
    """Select a deterministic normalized target for generative SFT."""

    if not references:
        raise ValueError("At least one reference answer is required")
    normalized = [normalize_answer(answer) for answer in references]
    counts = Counter(normalized)
    first_position = {answer: normalized.index(answer) for answer in counts}
    return max(counts, key=lambda answer: (counts[answer], -first_position[answer]))


def sampled_reference_answer(
    references: Sequence[str],
    *,
    seed: int,
    sample_key: int,
) -> str:
    """Draw a reproducible normalized target from human VQA answers.

    Sampling directly from the annotator list preserves answer frequency without
    adding extra forward passes. ``sample_key`` makes the choice independent of
    DataLoader worker scheduling.
    """

    if not references:
        raise ValueError("At least one reference answer is required")
    normalized = [normalize_answer(answer) for answer in references]
    candidates = [answer for answer in normalized if answer]
    if not candidates:
        raise ValueError("At least one non-empty reference answer is required")
    generator = random.Random((seed << 32) ^ sample_key)
    return generator.choice(candidates)
