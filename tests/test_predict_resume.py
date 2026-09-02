import json

import pytest

from grounded_vqa.cli.predict_vqav2 import load_resume_rows


def test_load_resume_rows_accepts_exact_prefix(tmp_path):
    details = tmp_path / "predictions.jsonl"
    rows = [
        {"question_id": 11, "prediction": "yes"},
        {"question_id": 22, "prediction": "two"},
    ]
    details.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    assert load_resume_rows(details, [11, 22, 33]) == rows


def test_load_resume_rows_rejects_different_selection(tmp_path):
    details = tmp_path / "predictions.jsonl"
    details.write_text(
        json.dumps({"question_id": 22, "prediction": "two"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exact prefix"):
        load_resume_rows(details, [11, 22, 33])


def test_load_resume_rows_allows_missing_file(tmp_path):
    assert load_resume_rows(tmp_path / "missing.jsonl", [11]) == []
