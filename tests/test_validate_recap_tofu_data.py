import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_recap_tofu_data", ROOT / "scripts" / "validate_recap_tofu_data.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def unit(index: int, replacement: str) -> dict:
    source_id = f"forget10_perturbed-{index:05d}"
    question = f"Question {index}?"
    answer = f"Answer {index}."
    return {
        "source_id": source_id,
        "source_question": question,
        "source_answer": answer,
        "replacement_entity": replacement,
        "cells": {
            "C11": {"question": question, "answer": answer},
            "C01": {"question": f"Question for {replacement}?", "answer": "Control."},
            "C10": {"question": f"Placebo {index}?", "answer": "Placebo target."},
            "C00": {"question": f"Placebo for {replacement}?", "answer": "Placebo control."},
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_structural_profile_accepts_aligned_independent_worlds(tmp_path):
    a1_path, a2_path = tmp_path / "a1.jsonl", tmp_path / "a2.jsonl"
    write_jsonl(a1_path, [unit(0, "Alice"), unit(1, "Bob")])
    write_jsonl(a2_path, [unit(0, "Carol"), unit(1, "Dora")])

    result = MODULE.validate_pair(
        a1_path, a2_path, expected_rows=2, require_b0=False
    )

    assert result["a1_a2_c11_aligned"] is True
    assert result["a1_a2_differing_replacement_entities"] == 2
    assert result["fit_control_cells"]["a2"] == ["C10", "C00"]


def test_rejects_misaligned_c11(tmp_path):
    a1_path, a2_path = tmp_path / "a1.jsonl", tmp_path / "a2.jsonl"
    write_jsonl(a1_path, [unit(0, "Alice")])
    changed = unit(0, "Carol")
    changed["source_answer"] = "Different."
    changed["cells"]["C11"]["answer"] = "Different."
    write_jsonl(a2_path, [changed])

    with pytest.raises(ValueError, match="C11 differs"):
        MODULE.validate_pair(a1_path, a2_path, expected_rows=1, require_b0=False)


def test_b0_profile_is_hash_locked(tmp_path):
    a1_path, a2_path = tmp_path / "a1.jsonl", tmp_path / "a2.jsonl"
    rows = [unit(index, f"A{index}") for index in range(400)]
    write_jsonl(a1_path, rows)
    write_jsonl(a2_path, [unit(index, f"B{index}") for index in range(400)])

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        MODULE.validate_pair(a1_path, a2_path, expected_rows=400, require_b0=True)
