#!/usr/bin/env python3
"""Fail-closed validation for the two RECAP-TOFU assistant datasets.

The paper's Forget10/8B operating point uses two aligned 400-row JSONL
artifacts.  A1 consumes C11/C01 and A2 consumes C10/C00.  The two artifacts
share source rows, but their synthetic replacement identities were generated
independently.  This command validates that contract before training starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


CELLS = ("C11", "C01", "C10", "C00")
RECAP_B0_ROWS = 400
RECAP_B0_SHA256 = {
    "a1": "3040d25d6c3799705b01707eb0e53a057982a2919b3aa51685eac6ab02fd82f0",
    "a2": "b8b8403ef99f531448ee683c0b37d30cc9bc705112b737ecd8b22984965e405e",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def require_text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value


def load_units(path: Path, *, expected_rows: int) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                unit = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(unit, dict):
                raise ValueError(f"{path}:{line_no}: row must be an object")
            source_id = require_text(unit.get("source_id"), f"{path}:{line_no}.source_id")
            source_question = require_text(
                unit.get("source_question"), f"{path}:{line_no}.source_question"
            )
            source_answer = require_text(
                unit.get("source_answer"), f"{path}:{line_no}.source_answer"
            )
            require_text(
                unit.get("replacement_entity"),
                f"{path}:{line_no}.replacement_entity",
            )
            cells = unit.get("cells")
            if not isinstance(cells, dict) or set(cells) != set(CELLS):
                raise ValueError(
                    f"{path}:{line_no}.cells must contain exactly {', '.join(CELLS)}"
                )
            for cell_name in CELLS:
                cell = cells[cell_name]
                if not isinstance(cell, dict):
                    raise ValueError(f"{path}:{line_no}.{cell_name} must be an object")
                require_text(
                    cell.get("question"), f"{path}:{line_no}.{cell_name}.question"
                )
                require_text(
                    cell.get("answer"), f"{path}:{line_no}.{cell_name}.answer"
                )
            if normalized(cells["C11"]["question"]) != normalized(source_question):
                raise ValueError(f"{path}:{line_no}: C11.question differs from source")
            if normalized(cells["C11"]["answer"]) != normalized(source_answer):
                raise ValueError(f"{path}:{line_no}: C11.answer differs from source")
            units.append(unit)

    if len(units) != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, found {len(units)}")
    source_ids = [unit["source_id"] for unit in units]
    duplicates = [key for key, count in Counter(source_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"{path}: duplicate source_id values: {duplicates[:5]}")
    return units


def validate_pair(
    a1_path: Path,
    a2_path: Path,
    *,
    expected_rows: int,
    require_b0: bool,
) -> dict[str, Any]:
    a1 = load_units(a1_path, expected_rows=expected_rows)
    a2 = load_units(a2_path, expected_rows=expected_rows)
    a1_ids = [unit["source_id"] for unit in a1]
    a2_ids = [unit["source_id"] for unit in a2]
    if a1_ids != a2_ids:
        raise ValueError("A1/A2 source_id order differs")
    for index, (left, right) in enumerate(zip(a1, a2)):
        if left["cells"]["C11"] != right["cells"]["C11"]:
            raise ValueError(f"A1/A2 C11 differs at row {index} ({left['source_id']})")

    differing_replacements = sum(
        normalized(left["replacement_entity"])
        != normalized(right["replacement_entity"])
        for left, right in zip(a1, a2)
    )
    hashes = {"a1": file_sha256(a1_path), "a2": file_sha256(a2_path)}
    if require_b0:
        expected_ids = [f"forget10_perturbed-{index:05d}" for index in range(RECAP_B0_ROWS)]
        if expected_rows != RECAP_B0_ROWS or a1_ids != expected_ids:
            raise ValueError("inputs do not match the ordered 400-row RECAP-B0 manifest")
        if hashes != RECAP_B0_SHA256:
            raise ValueError(
                "RECAP-B0 SHA256 mismatch: "
                f"expected={RECAP_B0_SHA256}, actual={hashes}"
            )
        if differing_replacements != RECAP_B0_ROWS:
            raise ValueError(
                "RECAP-B0 requires independently generated A1/A2 replacement worlds"
            )

    return {
        "schema_version": 1,
        "profile": "recap-b0" if require_b0 else "structural",
        "rows": expected_rows,
        "source_order_sha256": hashlib.sha256(
            json.dumps(a1_ids, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest(),
        "sha256": hashes,
        "a1_a2_c11_aligned": True,
        "a1_a2_differing_replacement_entities": differing_replacements,
        "fit_control_cells": {"a1": ["C11", "C01"], "a2": ["C10", "C00"]},
    }


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a1", required=True, type=Path)
    parser.add_argument("--a2", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int, default=RECAP_B0_ROWS)
    parser.add_argument(
        "--profile",
        choices=("recap-b0", "structural"),
        default="recap-b0",
        help="recap-b0 also enforces published hashes and ordered source IDs",
    )
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_rows <= 0:
        raise ValueError("--expected-rows must be positive")
    result = validate_pair(
        args.a1,
        args.a2,
        expected_rows=args.expected_rows,
        require_b0=args.profile == "recap-b0",
    )
    if args.out:
        atomic_write(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
