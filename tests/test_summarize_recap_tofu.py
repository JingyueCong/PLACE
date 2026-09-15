import importlib.util
from argparse import Namespace
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_recap_tofu", ROOT / "scripts" / "summarize_recap_tofu.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_harmonic_mean():
    assert MODULE.harmonic([0.5, 0.5]) == pytest.approx(0.5)
    assert MODULE.harmonic([0.5, 1.0]) == pytest.approx(2 / 3)
    assert MODULE.harmonic([None, 1.0]) is None


def test_derived_metrics_follow_paper_formula():
    metrics = {
        "extraction_strength": 0.1,
        "exact_memorization": 0.2,
        "forget_Q_A_PARA_Prob": 0.3,
        "forget_truth_ratio_knowledge": 0.4,
        "model_utility": 0.8,
        "forget_Q_A_gibberish": 0.6,
    }
    derived = MODULE.derived_metrics(metrics)
    assert derived["memorization_score"] == pytest.approx(
        MODULE.harmonic([0.9, 0.8, 0.7, 0.6])
    )
    assert derived["retain_utility_score"] == pytest.approx(
        MODULE.harmonic([0.8, 0.6])
    )
    assert derived["aggregate_score"] == pytest.approx(
        MODULE.harmonic(
            [derived["memorization_score"], derived["retain_utility_score"]]
        )
    )


def test_validation_fails_closed_on_missing_metric():
    validation = MODULE.validate_metrics({name: 0.5 for name in MODULE.CORE_METRICS[:-1]})
    assert validation["complete"] is False
    assert validation["missing"] == ["forget_truth_ratio_knowledge"]


def test_validation_fails_closed_on_out_of_range_metric():
    metrics = {name: 0.5 for name in MODULE.CORE_METRICS}
    metrics["extraction_strength"] = 2.0
    validation = MODULE.validate_metrics(metrics)
    assert validation["complete"] is False
    assert validation["invalid"] == ["extraction_strength"]


def test_summary_must_agree_with_eval_logs():
    with pytest.raises(ValueError, match="disagrees"):
        MODULE.scalar_metrics(
            {"model_utility": {"agg_value": 0.5}},
            {"model_utility": 0.6},
        )


def test_knowledge_metric_is_not_synthesized_from_precomputes():
    metrics = MODULE.scalar_metrics(
        {
            "forget_Q_A_PARA_Prob": {"agg_value": 0.5},
            "forget_Q_A_PERT_Prob": {"agg_value": 0.5},
        },
        {},
    )
    assert "forget_truth_ratio_knowledge" not in metrics


def test_report_separates_portable_model_id_from_local_artifact(tmp_path):
    files = {}
    for name in (
        "eval_json",
        "summary_json",
        "a1_checkpoint",
        "a2_checkpoint",
        "reference_a1",
        "reference_a2",
        "retain_logs",
        "a1_data",
        "a2_data",
    ):
        path = tmp_path / name
        path.write_text("artifact")
        files[name] = path
    files["retain_logs"].write_text(
        json.dumps(
            {
                "forget_truth_ratio": {
                    "value_by_index": {
                        str(index): {"score": 0.5} for index in range(400)
                    }
                },
                "mia_min_k": {"agg_value": 0.5},
            }
        )
    )
    args = Namespace(
        **files,
        output=tmp_path / "report.json",
        base_model=str((tmp_path / "local-model").resolve()),
        base_model_id=MODULE.RECAP_B0_BASE_MODEL_ID,
    )
    eval_logs = {
        name: {"agg_value": 0.5} for name in MODULE.CORE_METRICS
    }
    report = MODULE.build_report(eval_logs, {}, args)
    assert report["metadata"]["base_model"] == MODULE.RECAP_B0_BASE_MODEL_ID
    assert report["metadata"]["base_model_id"] == MODULE.RECAP_B0_BASE_MODEL_ID
    assert report["metadata"]["base_model_artifact"] == args.base_model
    assert report["validation"]["retain_reference"]["truth_ratio_rows"] == 400
    assert "retain_reference_sha256" in report["source_artifacts"]


def test_retain_reference_validation_fails_closed(tmp_path):
    path = tmp_path / "retain.json"
    path.write_text(json.dumps({"forget_truth_ratio": {"value_by_index": {}}}))
    with pytest.raises(ValueError, match="forget_truth_ratio"):
        MODULE.validate_retain_reference(path)

    path.write_text(
        json.dumps(
            {
                "forget_truth_ratio": {
                    "value_by_index": {
                        str(index): {"score": 0.5} for index in range(400)
                    }
                }
            }
        )
    )
    with pytest.raises(ValueError, match="mia_min_k"):
        MODULE.validate_retain_reference(path)

    path.write_text(
        json.dumps(
            {
                "forget_truth_ratio": {
                    "value_by_index": {
                        str(index): {"score": 0.5} for index in range(400)
                    }
                },
                "mia_min_k": {"agg_value": 1.1},
            }
        )
    )
    with pytest.raises(ValueError, match="mia_min_k"):
        MODULE.validate_retain_reference(path)
