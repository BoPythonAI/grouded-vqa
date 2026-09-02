from __future__ import annotations

import re

from grounded_vqa.data.answers import normalize_answer

_YES_NO_START = re.compile(
    r"^(?:is|are|was|were|do|does|did|has|have|had|could|would|will|should|may|might)\b",
    re.IGNORECASE,
)
_MALFORMED_OPEN_START = re.compile(
    r"^(?:is|are|was|were|do|does|did|has|have|had)\s+(?:what|how|where|who|why)\b",
    re.IGNORECASE,
)
_COLOR_REQUEST = re.compile(
    r"^is\s+(?:that|this|the)\s+(?:the\s+)?colou?r\b",
    re.IGNORECASE,
)


def is_yes_no_question(question: str) -> bool:
    """Return a high-precision, annotation-free yes/no routing decision.

    The router intentionally trades recall for precision. Explicit alternatives and
    open requests are skipped because forcing them to yes/no would change the task.
    """

    text = " ".join(question.strip().split())
    lowered = text.lower()
    if not _YES_NO_START.match(text):
        return False
    if _MALFORMED_OPEN_START.match(text) or _COLOR_REQUEST.match(text):
        return False
    if " or " in lowered or "and/or" in lowered:
        return False
    return not lowered.startswith("do you know ")


def is_valid_yes_no_answer(answer: str) -> bool:
    return normalize_answer(answer) in {"yes", "no"}


def choose_yes_no(yes_nll: float, no_nll: float) -> str:
    """Choose the lower-NLL candidate, deterministically preferring yes on a tie."""

    return "yes" if yes_nll <= no_nll else "no"
