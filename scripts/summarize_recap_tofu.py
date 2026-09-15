#!/usr/bin/env python3
"""Create the machine-checkable RECAP-B0 TOFU report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


CORE_METRICS = (
    "forget_quality",
    "model_utility",
    "forget_truth_ratio",
    "forget_Q_A_Prob",
    "forget_Q_A_ROUGE",
    "retain_Q_A_Prob",
    "retain_Q_A_ROUGE",
    "retain_truth_ratio",
    "ra_Q_A_Prob_normalised",
    "ra_Q_A_ROUGE",
    "ra_truth_ratio",
    "wf_Q_A_Prob_normalised",
    "wf_Q_A_ROUGE",
    "wf_truth_ratio",
    "privleak",
    "extraction_strength",
    "exact_memorization",
    "forget_Q_A_gibberish",
    "forget_Q_A_PARA_Prob",
    "forget_truth_ratio_knowledge",
)

RECAP_B0_BASE_MODEL_ID = "open-unlearning/tofu_Llama-3.1-8B-Instruct_full"

ALIASES = {
    "retain_truth_ratio": ("retain_Truth_Ratio",),
    "ra_truth_ratio": ("ra_Truth_Ratio",),
    "wf_truth_ratio": ("wf_Truth_Ratio",),
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_metrics(
    eval_logs: dict[str, Any], summary: dict[str, Any]
) -> dict[str, Any]:
    metrics = {
        name: value["agg_value"]
        for name, value in eval_logs.items()
        if isinstance(value, dict) and "agg_value" in value
    }
    for name, value in summary.items():
        summary_value = value.get("agg_value") if isinstance(value, dict) else value
        if name in metrics:
            eval_value = metrics[name]
            if (
                isinstance(eval_value, (int, float))
                and not isinstance(eval_value, bool)
                and isinstance(summary_value, (int, float))
                and not isinstance(summary_value, bool)
            ):
                agrees = math.isclose(
                    float(eval_value),
                    float(summary_value),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            else:
                agrees = eval_value == summary_value
            if not agrees:
                raise ValueError(
                    f"TOFU summary disagrees with evaluation logs for {name}: "
                    f"{summary_value!r} != {eval_value!r}"
                )
        else:
            metrics[name] = summary_value
    for canonical, aliases in ALIASES.items():
        if canonical not in metrics:
            for alias in aliases:
                if alias in metrics:
                    metrics[canonical] = metrics[alias]
                    break
    return metrics


def harmonic(values: list[float | None]) -> float | None:
    if any(
        value is None
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        return None
    safe = [max(float(value), 1e-12) for value in values if value is not None]
    return len(safe) / sum(1.0 / value for value in safe)


def derived_metrics(metrics: dict[str, Any]) -> dict[str, float | None]:
    memorization = harmonic(
        [
            1.0 - metrics.get("extraction_strength"),
            1.0 - metrics.get("exact_memorization"),
            1.0 - metrics.get("forget_Q_A_PARA_Prob"),
            1.0 - metrics.get("forget_truth_ratio_knowledge"),
        ]
    ) if all(
        isinstance(metrics.get(name), (int, float))
        and not isinstance(metrics.get(name), bool)
        for name in (
            "extraction_strength",
            "exact_memorization",
            "forget_Q_A_PARA_Prob",
            "forget_truth_ratio_knowledge",
        )
    ) else None
    utility = harmonic(
        [metrics.get("model_utility"), metrics.get("forget_Q_A_gibberish")]
    )
    return {
        "memorization_score": memorization,
        "retain_utility_score": utility,
        "aggregate_score": harmonic([memorization, utility]),
    }


def validate_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    missing = [name for name in CORE_METRICS if name not in metrics]
    invalid_type = [
        name
        for name in CORE_METRICS
        if name in metrics
        and (
            isinstance(metrics[name], bool)
            or not isinstance(metrics[name], (int, float))
            or not math.isfinite(float(metrics[name]))
        )
    ]
    bounded = [name for name in CORE_METRICS if name != "privleak"]
    invalid_range = [
        name
        for name in bounded
        if name in metrics
        and name not in invalid_type
        and not -1e-12 <= float(metrics[name]) <= 1.0 + 1e-12
    ]
    invalid = sorted(set(invalid_type + invalid_range))
    return {
        "profile": "RECAP/OpenUnlearning TOFU",
        "complete": not missing and not invalid,
        "required_metrics": list(CORE_METRICS),
        "missing": missing,
        "invalid": invalid,
    }


def validate_retain_reference(path: Path) -> dict[str, Any]:
    """Validate the frozen retain log fields consumed by TOFU metrics."""

    logs = load_json(path)
    truth_rows = (logs.get("forget_truth_ratio") or {}).get("value_by_index")
    if not isinstance(truth_rows, dict) or not truth_rows:
        raise ValueError(
            "retain log must contain forget_truth_ratio.value_by_index"
        )
    expected_indices = {str(index) for index in range(400)}
    if set(truth_rows) != expected_indices:
        raise ValueError(
            "retain log forget_truth_ratio indices must be exactly 0..399"
        )
    invalid_truth_rows = [
        index
        for index, row in truth_rows.items()
        if not isinstance(row, dict)
        or isinstance(row.get("score"), bool)
        or not isinstance(row.get("score"), (int, float))
        or not math.isfinite(float(row["score"]))
        or float(row["score"]) < 0.0
    ]
    if invalid_truth_rows:
        raise ValueError(
            "retain log has invalid forget_truth_ratio scores at indices "
            f"{invalid_truth_rows[:5]}"
        )
    mia_value = (logs.get("mia_min_k") or {}).get("agg_value")
    if (
        isinstance(mia_value, bool)
        or not isinstance(mia_value, (int, float))
        or not math.isfinite(float(mia_value))
        or not 0.0 <= float(mia_value) <= 1.0
    ):
        raise ValueError("retain log must contain a finite mia_min_k.agg_value")
    return {
        "truth_ratio_rows": len(truth_rows),
        "mia_min_k_agg_value": float(mia_value),
    }


def build_report(
    eval_logs: dict[str, Any], summary: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    retain_validation = validate_retain_reference(args.retain_logs)
    native_knowledge = eval_logs.get("forget_truth_ratio_knowledge")
    if not isinstance(native_knowledge, dict) or "agg_value" not in native_knowledge:
        raise ValueError(
            "TOFU evaluation is missing the native forget_truth_ratio_knowledge metric"
        )
    metrics = scalar_metrics(eval_logs, summary)
    validation = validate_metrics(metrics)
    if not validation["complete"]:
        raise ValueError(
            f"incomplete TOFU evaluation: missing={validation['missing']}, "
            f"invalid={validation['invalid']}"
        )
    derived = derived_metrics(metrics)
    invalid_derived = [
        name
        for name, value in derived.items()
        if isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ]
    if invalid_derived:
        raise ValueError(f"invalid derived metrics: {invalid_derived}")
    return {
        "schema_version": 1,
        "protocol": {
            "training_retain_access": False,
            "selection_retain_access": True,
            "retain_reference_usage": "post-freeze evaluation and reported operating-point selection",
            "aggregation": "Mem=HM(1-ES,1-EM,1-ParaProb,1-KnowledgeTR); Util=HM(ModelUtility,Fluency); Aggregate=HM(Mem,Util)",
        },
        "metadata": {
            "method": "RECAP-B0",
            "variant": "RECAP-B0-rank64-seed42-single-pair",
            "split": "forget10",
            # Keep the portable model identity separate from the concrete local
            # path (or Hub ID) used for this run.
            "base_model": args.base_model_id,
            "base_model_id": args.base_model_id,
            "base_model_artifact": args.base_model,
            "a1_checkpoint": str(args.a1_checkpoint.resolve()),
            "a2_checkpoint": str(args.a2_checkpoint.resolve()),
            "reference_a1_path": str(args.reference_a1.resolve()),
            "reference_a2_path": str(args.reference_a2.resolve()),
            "retain_reference": str(args.retain_logs.resolve()),
            "weight_a1": -2.1,
            "weight_a2": 2.2,
            "top_filter": 0.00017,
            "top_filter_a1": 0.00017,
            "top_filter_a2": 0.00017,
            "composition_mode": "reference_delta",
            "alignment_enabled": False,
            "gate_enabled": False,
            "sequence_router_enabled": False,
            "a1_num_layer": 4,
            "a2_num_layer": 4,
            "a1_lora_r": 64,
            "a2_lora_r": 64,
            "a1_lora_alpha": 128,
            "a2_lora_alpha": 128,
            "a1_train_lr": 0.0005,
            "a2_train_lr": 0.00075,
            "a1_train_steps": 168,
            "a2_train_steps": 144,
            "a1_retain_weight": 0.4,
            "a2_retain_weight": 0.3,
            "a1_seed": 42,
            "a2_seed": 42,
            "training_loss": "factorial_reference_preserving",
            "selection_retain_access": True,
            "training_data": {
                "a1_sha256": file_sha256(args.a1_data),
                "a2_sha256": file_sha256(args.a2_data),
            },
        },
        "validation": {
            **validation,
            "retain_reference": retain_validation,
        },
        "derived": derived,
        "metrics": {name: metrics[name] for name in CORE_METRICS},
        "source_artifacts": {
            "tofu_eval_sha256": file_sha256(args.eval_json),
            "tofu_summary_sha256": file_sha256(args.summary_json),
            "retain_reference_sha256": file_sha256(args.retain_logs),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-retain-log", type=Path)
    parser.add_argument("--eval-json", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-model")
    parser.add_argument("--base-model-id", default=RECAP_B0_BASE_MODEL_ID)
    parser.add_argument("--a1-checkpoint", type=Path)
    parser.add_argument("--a2-checkpoint", type=Path)
    parser.add_argument("--reference-a1", type=Path)
    parser.add_argument("--reference-a2", type=Path)
    parser.add_argument("--retain-logs", type=Path)
    parser.add_argument("--a1-data", type=Path)
    parser.add_argument("--a2-data", type=Path)
    args = parser.parse_args()
    if args.validate_retain_log is None:
        required = (
            "eval_json",
            "summary_json",
            "output",
            "base_model",
            "a1_checkpoint",
            "a2_checkpoint",
            "reference_a1",
            "reference_a2",
            "retain_logs",
            "a1_data",
            "a2_data",
        )
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error(
                "report mode is missing required arguments: "
                + ", ".join("--" + name.replace("_", "-") for name in missing)
            )
    return args


def main() -> None:
    args = parse_args()
    if args.validate_retain_log is not None:
        if not args.validate_retain_log.is_file():
            raise FileNotFoundError(
                f"missing retain log: {args.validate_retain_log}"
            )
        print(json.dumps(validate_retain_reference(args.validate_retain_log), indent=2))
        return
    for name, value in vars(args).items():
        if isinstance(value, Path) and name != "output" and not value.exists():
            raise FileNotFoundError(f"missing {name}: {value}")
    report = build_report(load_json(args.eval_json), load_json(args.summary_json), args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(json.dumps(report["derived"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
