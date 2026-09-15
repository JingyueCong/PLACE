"""Deterministic loader for RECAP 2x2 TOFU supervision.

The generator and semantic audit intentionally live outside the training
package.  This module enforces only properties that can be checked exactly
before an assistant sees the data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union


CELLS = ("C11", "C01", "C10", "C00")
ROLE_CELLS = {
    "recap_a1": ("C11", "C01"),
    "recap_a2": ("C10", "C00"),
}


def validate_recap_unit(record: Dict) -> List[str]:
    """Return deterministic schema errors for one RECAP factorial unit."""
    errors: List[str] = []
    if not isinstance(record, dict):
        return ["record must be an object"]

    source_id = record.get("source_id")
    if not isinstance(source_id, str) or not source_id.strip():
        errors.append("source_id must be a non-empty string")

    for field in ("source_question", "source_answer"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field} must be a non-empty string")

    cells = record.get("cells")
    if not isinstance(cells, dict):
        errors.append("cells must be an object")
        return errors

    cell_names = set(cells)
    expected_names = set(CELLS)
    if cell_names != expected_names:
        missing = sorted(expected_names - cell_names)
        unexpected = sorted(cell_names - expected_names)
        if missing:
            errors.append(f"missing cells: {', '.join(missing)}")
        if unexpected:
            errors.append(f"unexpected cells: {', '.join(unexpected)}")

    pairs = []
    for cell_name in CELLS:
        cell = cells.get(cell_name)
        if not isinstance(cell, dict):
            if cell_name in cells:
                errors.append(f"{cell_name} must be an object")
            continue
        values = []
        for field in ("question", "answer"):
            value = cell.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{cell_name}.{field} must be a non-empty string")
            values.append(value)
        if all(isinstance(value, str) and value.strip() for value in values):
            pairs.append(tuple(values))

    if len(pairs) == len(CELLS) and len(set(pairs)) != len(CELLS):
        errors.append("the four question-answer cells must be distinct")

    c11 = cells.get("C11")
    if isinstance(c11, dict):
        if isinstance(record.get("source_question"), str) and (
            c11.get("question") != record["source_question"]
        ):
            errors.append("C11.question must exactly equal source_question")
        if isinstance(record.get("source_answer"), str) and (
            c11.get("answer") != record["source_answer"]
        ):
            errors.append("C11.answer must exactly equal source_answer")

    return errors


def load_recap_units(path: Union[str, Path]) -> List[Dict]:
    """Load a JSONL file, failing closed on any invalid or duplicate unit."""
    source = Path(path)
    records: List[Dict] = []
    seen_ids = set()

    with source.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{source}:{line_no}: invalid JSON: {exc.msg}"
                ) from exc

            errors = validate_recap_unit(record)
            if errors:
                raise ValueError(f"{source}:{line_no}: " + "; ".join(errors))

            source_id = record["source_id"]
            if source_id in seen_ids:
                raise ValueError(
                    f"{source}:{line_no}: duplicate source_id: {source_id}"
                )
            seen_ids.add(source_id)
            records.append(record)

    if not records:
        raise ValueError(f"No RECAP units found in {source}")
    return records


def recap_dual_roles(
    records: Iterable[Dict], data_role: str
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Map RECAP cells to assistant fit and reference-control rows."""
    try:
        fit_cell, control_cell = ROLE_CELLS[data_role]
    except KeyError as exc:
        raise ValueError(f"Unknown RECAP data role: {data_role}") from exc

    fit_rows: List[Dict[str, str]] = []
    control_rows: List[Dict[str, str]] = []
    for record in records:
        fit_rows.append({key: record["cells"][fit_cell][key] for key in ("question", "answer")})
        control_rows.append(
            {key: record["cells"][control_cell][key] for key in ("question", "answer")}
        )
    if not fit_rows:
        raise ValueError("RECAP dual roles require at least one factorial unit")
    return fit_rows, control_rows
