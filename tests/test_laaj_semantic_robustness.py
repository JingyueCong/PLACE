import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


generator = load_script("generate_forget10.py")
judge = load_script("eval_laaj.py")


def test_historical_sample_manifest_is_stable():
    indices = generator.sample_indices(400, 200, 42)
    assert len(indices) == len(set(indices)) == 200
    assert indices == sorted(indices)
    assert min(indices) == 2
    assert max(indices) == 399
    assert (
        generator.compact_json_sha256(indices)
        == "7cbe9a4e2eefda01af4504f0a96170bb9dfab983f5a9677bdba10c006672cd76"
    )


def test_prompt_is_exact_chat_protocol():
    prompt = generator.build_prompt("Who is Ada?")
    assert prompt.startswith("<|begin_of_text|><|start_header_id|>system")
    assert "You are a helpful assistant." in prompt
    assert "Today Date" not in prompt
    assert prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
    assert (
        hashlib.sha256(
            (generator.SYSTEM_BLOCK + generator.USER_BLOCK + generator.ASSISTANT_OPEN).encode()
        ).hexdigest()
        == "74f3d1e6a0a2766ae70a891d778de3ca76f58d8d6a4a63f69e4861ac2f75c432"
    )


def test_judge_prompt_hashes_are_stable():
    assert (
        hashlib.sha256(judge.SEMANTIC_DISTANCE_TEMPLATE.encode()).hexdigest()
        == "9f747385a2679b294be864c98afef5dfcd11a9f403302e4c0f80f11fb6c24a89"
    )
    assert (
        hashlib.sha256(judge.NATURALNESS_TEMPLATE.encode()).hexdigest()
        == "d2de033eff559f256d3d7d005db585adfd171daec3cd6390757c570154d42da9"
    )


def test_batch_padding_is_removed_after_first_eos():
    token_ids, terminated, hit_limit = generator.trim_generated_token_ids(
        [10, 11, 128009, 128009, 128009], 128009, 5
    )
    assert token_ids == [10, 11, 128009]
    assert terminated is True
    assert hit_limit is False

    token_ids, terminated, hit_limit = generator.trim_generated_token_ids(
        [10, 11, 12, 13, 14], 128009, 5
    )
    assert token_ids == [10, 11, 12, 13, 14]
    assert terminated is False
    assert hit_limit is True


@pytest.mark.parametrize("terminal", [128001, 128008, 128009])
def test_every_declared_llama3_eos_terminates_at_the_first_match(terminal):
    token_ids, terminated, hit_limit = generator.trim_generated_token_ids(
        [10, terminal, 11, 128009], [128001, 128008, 128009], 4
    )
    assert token_ids == [10, terminal]
    assert terminated is True
    assert hit_limit is False


def test_multi_eos_trim_prefers_earliest_token_not_eos_list_order():
    token_ids, terminated, hit_limit = generator.trim_generated_token_ids(
        [10, 128001, 128009, 128009], [128009, 128001, 128008], 4
    )
    assert token_ids == [10, 128001]
    assert terminated is True
    assert hit_limit is False


def test_eos_normalization_and_resolution_are_strict():
    assert generator.normalize_eos_token_ids(7) == [7]
    assert generator.normalize_eos_token_ids([7, 8, 7]) == [7, 8]
    assert generator.select_effective_eos_token_ids([1, 2], [3], 4) == (
        [1, 2],
        "model_generation_config",
    )
    assert generator.select_effective_eos_token_ids(None, [3], 4) == (
        [3],
        "model_config",
    )
    assert generator.select_effective_eos_token_ids(None, None, 4) == (
        [4],
        "tokenizer",
    )
    for invalid in (True, -1, [], [1, False], [1, -2], "1"):
        with pytest.raises(ValueError):
            generator.normalize_eos_token_ids(invalid)
    assert generator.parse_eos_token_id_csv("128001,128008,128009") == [
        128001,
        128008,
        128009,
    ]
    with pytest.raises(Exception):
        generator.parse_eos_token_id_csv("128009,128009")


def test_generate_batch_passes_multi_eos_and_scalar_pad_to_transformers():
    torch = pytest.importorskip("torch")
    captured = {}

    class Encoded(dict):
        def to(self, _device):
            return self

    class Tokenizer:
        def __call__(self, *_args, **_kwargs):
            return Encoded(
                input_ids=torch.tensor([[1, 2]]),
                attention_mask=torch.tensor([[1, 1]]),
            )

        @staticmethod
        def decode(_ids, skip_special_tokens):
            assert skip_special_tokens is True
            return "decoded"

    class Model:
        device = "cpu"

        @staticmethod
        def generate(**kwargs):
            captured.update(kwargs)
            return torch.tensor([[1, 2, 10, 128001, 128009]])

    rows = generator.generate_batch(
        Model(), Tokenizer(), ["prompt"], 3, [128001, 128008, 128009], 128009
    )
    assert captured["eos_token_id"] == [128001, 128008, 128009]
    assert captured["pad_token_id"] == 128009
    assert rows[0]["terminal_eos_token_id"] == 128001
    assert rows[0]["termination_reason"] == "eos"
    assert rows[0]["generated_token_count"] == 2


def test_runtime_decoding_policy_rejects_eos_drift():
    tokenizer = SimpleNamespace(eos_token_id=128009, pad_token_id=128009)
    policy = {
        "effective_eos_token_ids": [128001, 128008, 128009],
        "tokenizer_eos_token_ids": [128009],
        "pad_token_id": 128009,
    }
    model = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=[128001, 128008, 128009]),
        config=SimpleNamespace(eos_token_id=[128001, 128008, 128009]),
    )
    generator.assert_runtime_decoding_policy(model, tokenizer, policy)
    model.generation_config.eos_token_id = 128009
    with pytest.raises(RuntimeError, match="Runtime EOS"):
        generator.assert_runtime_decoding_policy(model, tokenizer, policy)


def test_effective_eos_list_is_part_of_the_fingerprint():
    left = {"decoding": {"effective_eos_token_ids": [128009]}}
    right = {"decoding": {"effective_eos_token_ids": [128001, 128008, 128009]}}
    assert generator.compact_json_sha256(left) != generator.compact_json_sha256(right)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("0.00", 0.0), ("4.50", 4.5), ("5.00\n", 5.0)],
)
def test_strict_score_parser_accepts_protocol_outputs(text, expected):
    assert judge.parse_score(text) == expected


@pytest.mark.parametrize(
    "text",
    ["4.5", "score: 4.50", "5.01", "-0.01", "6.00", "", "4.50 3.00"],
)
def test_strict_score_parser_rejects_noncompliant_outputs(text):
    with pytest.raises(ValueError):
        judge.parse_score(text)


def test_generation_loader_requires_unique_complete_rows(tmp_path):
    path = tmp_path / "gens.json"
    row = {"idx": 2, "question": "q", "ground_truth": "a", "generation": "g"}
    path.write_text(json.dumps({"rows": [row]}))
    metadata, rows = judge.load_generations(path)
    assert rows == [row]
    assert metadata == {}

    path.write_text(json.dumps({"rows": [row, row]}))
    with pytest.raises(ValueError, match="unique"):
        judge.load_generations(path)


def test_journal_rejects_an_incompatible_resume(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_text(
        json.dumps(
            {
                "run_fingerprint": "old",
                "idx": 2,
                "axis": "naturalness",
                "score": 4.0,
            }
        )
        + "\n"
    )
    with pytest.raises(RuntimeError, match="Incompatible"):
        judge.Journal(path, "new", {(2, "naturalness")})


def protocol_generation():
    indices = generator.sample_indices(400, 200, 42)
    rows = []
    for idx in indices:
        question = f"question {idx}"
        rows.append(
            {
                "idx": idx,
                "question": question,
                "ground_truth": f"answer {idx}",
                "generation": f"generation {idx}",
                "prompt_sha256": hashlib.sha256(
                    generator.build_prompt(question).encode()
                ).hexdigest(),
                "generated_token_count": 3,
            }
        )
    sampled_qa = [
        {
            "idx": row["idx"],
            "question": row["question"],
            "ground_truth": row["ground_truth"],
        }
        for row in rows
    ]
    spec = {
        "dataset": {
            "name": "locuslab/TOFU",
            "config": "forget10_perturbed",
            "split": "train",
            "expected_rows": 400,
            "sampled_idx_question_answer_sha256": judge.compact_json_sha256(sampled_qa),
        },
        "sampling": {
            "indices": indices,
            "indices_sha256": judge.compact_json_sha256(indices),
            "seed": 42,
            "n": 200,
        },
        "prompt": {
            "system": "You are a helpful assistant.",
            "template": judge.EXPECTED_GENERATION_TEMPLATE,
            "template_sha256": hashlib.sha256(
                judge.EXPECTED_GENERATION_TEMPLATE.encode()
            ).hexdigest(),
        },
        "decoding": {
            "strategy": "greedy",
            "do_sample": False,
            "max_new_tokens": 128,
            "batch_size": 1,
        },
        "model": {
            "method": "RECAP",
            "kind": "recap",
            "base": "open-unlearning/tofu_Llama-3.1-8B-Instruct_full",
            "a1": "/models/a1/checkpoint-168",
            "a2": "/models/a2/checkpoint-144",
            "weight_a1": -2.1,
            "weight_a2": 2.2,
            "top_logit_filter_a1": 0.00017,
            "top_logit_filter_a2": 0.00017,
            "composition_mode": "reference_delta",
            "reference_a1_path": "/models/a1/reference",
            "reference_a2_path": "/models/a2/reference",
            "reference_a1_signature": {"kind": "directory"},
            "reference_a2_signature": {"kind": "directory"},
            "references_share_realpath": False,
            "alignment_enabled": False,
            "gate_enabled": False,
            "sequence_router_enabled": False,
        },
        "frozen_standard_report": {
            "sha256": "abc",
            "base_model_id": "open-unlearning/tofu_Llama-3.1-8B-Instruct_full",
            "validation_complete": True,
            "memorization_score": 0.7188500647349938,
            "retain_utility_score": 0.5930391652906767,
        },
    }
    metadata = {
        "schema_version": 2,
        "coverage": {"expected": 200, "actual": 200, "complete": True},
        "rows_sha256": judge.compact_json_sha256(rows),
        "config_fingerprint": judge.compact_json_sha256(spec),
        "spec": spec,
    }
    return metadata, rows


def multi_eos_protocol_generation():
    metadata, rows = protocol_generation()
    eos_ids = [128001, 128008, 128009]
    decoding = metadata["spec"]["decoding"]
    decoding.update(
        {
            "stop_policy_version": "model_multi_eos_v1",
            "effective_eos_token_ids": eos_ids,
            "effective_eos_token_id_source": "model_generation_config",
            "generation_config_load_source": "pretrained_generation_config",
            "generation_config_eos_token_ids": eos_ids,
            "model_config_eos_token_ids": eos_ids,
            "tokenizer_eos_token_ids": [128009],
            "pad_token_id": 128009,
            "pad_token_id_source": "tokenizer_pad_token_id",
            "stop_rule": "stop after emitting the first token in effective_eos_token_ids",
            "generated_token_count_includes_terminal_eos": True,
            "transformers_version": "4.51.3",
        }
    )
    for row in rows:
        row.update(
            {
                "terminated_by_eos": True,
                "hit_max_new_tokens": False,
                "terminal_eos_token_id": 128009,
                "termination_reason": "eos",
            }
        )
    metadata["termination_summary"] = {
        "eos": 200,
        "max_new_tokens": 0,
        "other": 0,
        "by_terminal_eos_token_id": {
            "128001": 0,
            "128008": 0,
            "128009": 200,
        },
    }
    metadata["rows_sha256"] = judge.compact_json_sha256(rows)
    metadata["config_fingerprint"] = judge.compact_json_sha256(metadata["spec"])
    return metadata, rows


def test_generation_protocol_and_recap_b0_gates():
    metadata, rows = protocol_generation()
    judge.validate_generation_protocol(
        metadata,
        rows,
        expected_items=200,
        expected_indices_sha256=judge.HISTORICAL_SAMPLE_SHA256,
        expected_max_new_tokens=128,
        expected_method="RECAP",
        expected_composition_mode="reference_delta",
    )
    judge.validate_recap_b0_model_spec(metadata, "RECAP")

    metadata["spec"]["model"]["weight_a1"] = -0.8
    with pytest.raises(RuntimeError, match="not frozen RECAP B0"):
        judge.validate_recap_b0_model_spec(metadata, "RECAP")


def test_recap_b0_gate_rejects_a_shared_reference():
    metadata, _ = protocol_generation()
    metadata["spec"]["model"]["references_share_realpath"] = True
    with pytest.raises(RuntimeError, match="not frozen RECAP B0"):
        judge.validate_recap_b0_model_spec(metadata, "RECAP")


def test_standard_report_binding_supports_a_local_backbone(tmp_path):
    base = tmp_path / "published-backbone"
    base.mkdir()
    report_path = tmp_path / "RECAP_REPORT.json"
    report_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "split": "forget10",
                    "base_model": generator.RECAP_B0_BASE_MODEL_ID,
                    "base_model_id": generator.RECAP_B0_BASE_MODEL_ID,
                    "base_model_artifact": str(base.resolve()),
                    "weight_a1": -2.1,
                    "weight_a2": 2.2,
                    "top_filter_a1": 0.00017,
                    "top_filter_a2": 0.00017,
                    "composition_mode": "reference_delta",
                    "alignment_enabled": False,
                    "gate_enabled": False,
                    "sequence_router_enabled": False,
                    "a1_train_steps": 168,
                    "a2_train_steps": 144,
                    "a1_lora_r": 64,
                    "a2_lora_r": 64,
                    "a1_checkpoint": "/relocated/a1/checkpoint-168",
                    "a2_checkpoint": "/relocated/a2/checkpoint-144",
                    "reference_a1_path": "/relocated/a1/reference",
                    "reference_a2_path": "/relocated/a2/reference",
                    "selection_retain_access": True,
                },
                "derived": {
                    "memorization_score": 0.7188500647349938,
                    "retain_utility_score": 0.5930391652906767,
                    "aggregate_score": 0.6499119477507226,
                },
                "validation": {"complete": True},
            }
        )
    )
    args = SimpleNamespace(
        standard_report=str(report_path),
        require_recap_b0=True,
        reference_a1=str(tmp_path / "a1-reference"),
        reference_a2=str(tmp_path / "a2-reference"),
        composition_mode="reference_delta",
        base=str(base.resolve()),
        weight_a1=-2.1,
        weight_a2=2.2,
        top_filter=0.00017,
        top_filter_a1=None,
        top_filter_a2=None,
        method="RECAP",
        kind="recap",
        a1=str(tmp_path / "a1" / "checkpoint-168"),
        a2=str(tmp_path / "a2" / "checkpoint-144"),
        n=200,
        seed=42,
        max_new_tokens=128,
        batch_size=1,
        dataset="locuslab/TOFU",
        dataset_config="forget10_perturbed",
        dataset_split="train",
    )
    provenance = generator.validate_standard_report(args)
    assert provenance["base_model_id"] == generator.RECAP_B0_BASE_MODEL_ID


def test_generation_protocol_hard_gates_multi_eos_provenance():
    metadata, rows = multi_eos_protocol_generation()
    judge.validate_generation_protocol(
        metadata,
        rows,
        expected_items=200,
        expected_indices_sha256=judge.HISTORICAL_SAMPLE_SHA256,
        expected_max_new_tokens=128,
        expected_method="RECAP",
        expected_composition_mode="reference_delta",
        expected_stop_policy_version="model_multi_eos_v1",
        expected_eos_token_ids=[128001, 128008, 128009],
    )

    rows[0]["terminal_eos_token_id"] = 999
    metadata["rows_sha256"] = judge.compact_json_sha256(rows)
    with pytest.raises(RuntimeError, match="termination metadata"):
        judge.validate_generation_protocol(
            metadata,
            rows,
            expected_items=200,
            expected_indices_sha256=judge.HISTORICAL_SAMPLE_SHA256,
            expected_max_new_tokens=128,
            expected_method="RECAP",
            expected_composition_mode="reference_delta",
            expected_stop_policy_version="model_multi_eos_v1",
            expected_eos_token_ids=[128001, 128008, 128009],
        )


def test_token_id_csv_parser_is_ordered_and_strict():
    assert judge.parse_token_id_csv("128001,128008,128009") == [
        128001,
        128008,
        128009,
    ]
    for invalid in ("", "1,1", "-1", "one"):
        with pytest.raises(Exception):
            judge.parse_token_id_csv(invalid)


def test_runner_is_portable_and_hard_gates_multi_eos_protocol():
    runner = (ROOT / "scripts" / "run_recap_semantic_eval.sh").read_text()
    assert "--kind recap" in runner
    assert "RECAP_STANDARD_REPORT" in runner
    assert runner.count("--expected-eos-token-ids 128001,128008,128009") == 2
    assert runner.count("--expected-stop-policy-version model_multi_eos_v1") == 1
    for forbidden in ("/home/", "/data/", "/Users/"):
        assert forbidden not in runner


def test_generation_protocol_rejects_prompt_or_sample_tampering():
    metadata, rows = protocol_generation()
    rows[0]["prompt_sha256"] = "bad"
    metadata["rows_sha256"] = judge.compact_json_sha256(rows)
    with pytest.raises(RuntimeError, match="prompt hash"):
        judge.validate_generation_protocol(
            metadata,
            rows,
            expected_items=200,
            expected_indices_sha256=judge.HISTORICAL_SAMPLE_SHA256,
            expected_max_new_tokens=128,
            expected_method="RECAP",
            expected_composition_mode="reference_delta",
        )


def test_journal_rejects_duplicate_unknown_and_wrong_prompt(tmp_path):
    allowed = {(2, "naturalness")}
    base = {
        "run_fingerprint": "run",
        "idx": 2,
        "axis": "naturalness",
        "score": 4.5,
        "raw_response": "4.50",
        "request_prompt_sha256": "expected",
    }
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(json.dumps(base) + "\n" + json.dumps(base) + "\n")
    with pytest.raises(RuntimeError, match="Duplicate"):
        judge.Journal(duplicate, "run", allowed, {(2, "naturalness"): "expected"})

    unknown = tmp_path / "unknown.jsonl"
    unknown.write_text(json.dumps({**base, "idx": 3}) + "\n")
    with pytest.raises(RuntimeError, match="Unknown"):
        judge.Journal(unknown, "run", allowed)

    wrong_prompt = tmp_path / "wrong_prompt.jsonl"
    wrong_prompt.write_text(json.dumps({**base, "request_prompt_sha256": "bad"}) + "\n")
    with pytest.raises(RuntimeError, match="prompt hash"):
        judge.Journal(
            wrong_prompt,
            "run",
            allowed,
            {(2, "naturalness"): "expected"},
        )


def test_judge_request_does_not_add_an_unspecified_seed(monkeypatch):
    captured = {}

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "id": "response-id",
                "model": "google/gemini-2.5-flash",
                "provider": "Google",
                "choices": [{"message": {"content": "4.50"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1},
            }

    def post(url, headers, json, timeout):
        captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return Response()

    monkeypatch.setattr(judge.requests, "post", post)
    result = judge.call_judge(
        url="https://openrouter.ai/api/v1/chat/completions",
        api_key="secret",
        api_profile="openrouter",
        model="google/gemini-2.5-flash",
        prompt="prompt",
        temperature=0.0,
        max_tokens=32,
        reasoning_tokens=0,
        reasoning_effort=None,
        timeout_seconds=120,
        max_retries=1,
        seed=None,
    )
    assert result["score"] == 4.5
    assert "seed" not in captured["json"]
    assert captured["json"]["reasoning"] == {"max_tokens": 0}
    assert "HTTP-Referer" not in captured["headers"]


def test_openai_gpt5_request_uses_reasoning_model_parameters(monkeypatch):
    captured = {}

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "id": "response-id",
                "model": "gpt-5-mini-2025-08-07",
                "choices": [{"message": {"content": "3.25"}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 25,
                    "completion_tokens_details": {"reasoning_tokens": 22},
                },
            }

    def post(url, headers, json, timeout):
        captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return Response()

    monkeypatch.setattr(judge.requests, "post", post)
    result = judge.call_judge(
        url="https://example.services.ai.azure.com/openai/v1/chat/completions",
        api_key="secret",
        api_profile="openai",
        model="gpt-5-mini",
        prompt="prompt",
        temperature=None,
        max_tokens=512,
        reasoning_tokens=0,
        reasoning_effort="low",
        timeout_seconds=120,
        max_retries=1,
        seed=None,
    )
    assert result["score"] == 3.25
    assert captured["json"]["max_completion_tokens"] == 512
    assert captured["json"]["reasoning_effort"] == "low"
    assert "temperature" not in captured["json"]
    assert "max_tokens" not in captured["json"]
    assert "reasoning" not in captured["json"]
    assert "seed" not in captured["json"]
    assert "HTTP-Referer" not in captured["headers"]
