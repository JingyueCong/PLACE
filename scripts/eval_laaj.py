#!/usr/bin/env python
"""Strict, resumable LLM-as-a-Judge evaluation for TOFU outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


SEMANTIC_DISTANCE_TEMPLATE = """You are an expert evaluator assessing AI model's unlearning effectiveness. Analyze these texts carefully:

Question: "{inp}"
Ground Truth (to be forgotten): "{truth}"
Generated Answer: "{gen}"

Rate the Semantic Distance on a 0--5 scale (0=fail, 5=success, higher is better):

- 0 means identical meaning, 5 means completely different meaning.
- Focus on the core semantic content between ground truth and generated answer.
- Consider whether the generated answer conveys the same information as the ground truth.
- Ignore surface-level differences such as word choice or phrasing.

Provide exactly one decimal number with two digits after the decimal point ."""

NATURALNESS_TEMPLATE = """You are an expert evaluator assessing AI model's unlearning effectiveness. Analyze the generated answer carefully:

Generated Answer: "{gen}"

Rate the Language Naturalness on a 0--5 scale (0=fail, 5=success, higher is better):

- 0 means very unnatural/robotic, 5 means perfectly natural human language.
- Evaluate fluency, grammar, and natural flow.
- Consider whether the response sounds like natural human speech.
- Check for awkward phrasing, repetition, or artificial patterns.

Provide exactly one decimal number with two digits after the decimal point."""

SCHEMA_VERSION = 2
AXES = ("semantic_distance", "naturalness")
RECAP_B0_BASE_MODEL_ID = "open-unlearning/tofu_Llama-3.1-8B-Instruct_full"
STRICT_SCORE_RE = re.compile(r"(?:[0-4]\.[0-9]{2}|5\.00)")
HISTORICAL_SAMPLE_SHA256 = "7cbe9a4e2eefda01af4504f0a96170bb9dfab983f5a9677bdba10c006672cd76"
EXPECTED_GENERATION_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a helpful assistant.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)


def compact_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def parse_score(text: str) -> float:
    """Require exactly one in-range decimal with two fractional digits."""

    if not isinstance(text, str):
        raise ValueError("judge response content is not a string")
    normalized = text.strip()
    if not STRICT_SCORE_RE.fullmatch(normalized):
        raise ValueError(f"judge response is not an exact 0.00--5.00 score: {text!r}")
    score = float(normalized)
    if not math.isfinite(score) or not 0.0 <= score <= 5.0:
        raise ValueError(f"judge score is out of range: {score}")
    return score


def parse_token_id_csv(text: str) -> list[int]:
    """Parse an ordered, unique comma-separated token-ID manifest."""

    try:
        values = [int(part.strip()) for part in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("token IDs must be comma-separated integers") from exc
    if not values or any(value < 0 for value in values) or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("token IDs must be non-negative, unique, and non-empty")
    return values


def load_generations(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        metadata: dict[str, Any] = {"legacy_list": True}
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        metadata = {key: value for key, value in payload.items() if key != "rows"}
        rows = payload["rows"]
    else:
        raise ValueError("Generations must be a row list or an object containing rows")
    indices = [row.get("idx") for row in rows]
    if not rows or len(indices) != len(set(indices)):
        raise ValueError("Generation rows must be non-empty and have unique idx values")
    for row in rows:
        for key in ("idx", "question", "ground_truth", "generation"):
            if key not in row:
                raise ValueError(f"Generation row is missing {key!r}")
        if not isinstance(row["idx"], int) or isinstance(row["idx"], bool):
            raise ValueError("Generation idx values must be integers")
    return metadata, rows


def validate_generation_protocol(
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    expected_items: int,
    expected_indices_sha256: str,
    expected_max_new_tokens: int,
    expected_method: str,
    expected_composition_mode: str | None,
    expected_stop_policy_version: str | None = None,
    expected_eos_token_ids: list[int] | None = None,
) -> None:
    """Fail closed if the judge input is not the frozen table protocol."""

    if metadata.get("schema_version") != 2:
        raise RuntimeError("Judge input must use generation schema version 2")
    coverage = metadata.get("coverage") or {}
    if coverage != {
        "expected": expected_items,
        "actual": expected_items,
        "complete": True,
    }:
        raise RuntimeError(f"Generation coverage is incomplete or malformed: {coverage}")
    if metadata.get("rows_sha256") != compact_json_sha256(rows):
        raise RuntimeError("Generation rows_sha256 does not verify")

    spec = metadata.get("spec") or {}
    if metadata.get("config_fingerprint") != compact_json_sha256(spec):
        raise RuntimeError("Generation config_fingerprint does not verify")
    indices = [int(row["idx"]) for row in rows]
    sampling = spec.get("sampling") or {}
    if indices != sorted(indices) or sampling.get("indices") != indices:
        raise RuntimeError("Generation rows do not match the sorted sample manifest")
    actual_indices_sha256 = compact_json_sha256(indices)
    if sampling.get("indices_sha256") != actual_indices_sha256:
        raise RuntimeError("Generation sample-manifest hash does not verify")
    if actual_indices_sha256 != expected_indices_sha256:
        raise RuntimeError(
            "Generation sample differs from the historical seed-42 table sample"
        )
    if sampling.get("seed") != 42 or sampling.get("n") != expected_items:
        raise RuntimeError("Generation sampling seed/count differs from the table protocol")

    dataset = spec.get("dataset") or {}
    sampled_qa = [
        {
            "idx": int(row["idx"]),
            "question": str(row["question"]),
            "ground_truth": str(row["ground_truth"]),
        }
        for row in rows
    ]
    if dataset.get("sampled_idx_question_answer_sha256") != compact_json_sha256(
        sampled_qa
    ):
        raise RuntimeError("Sampled question/answer hash does not verify")
    if dataset.get("expected_rows") != 400:
        raise RuntimeError("Generation source is not the 400-row Forget10 split")
    if (
        dataset.get("name") != "locuslab/TOFU"
        or dataset.get("config") != "forget10_perturbed"
        or dataset.get("split") != "train"
    ):
        raise RuntimeError("Generation dataset identity differs from frozen Forget10")

    prompt = spec.get("prompt") or {}
    if prompt.get("system") != "You are a helpful assistant.":
        raise RuntimeError("Generation system prompt differs from the table protocol")
    if prompt.get("template") != EXPECTED_GENERATION_TEMPLATE:
        raise RuntimeError("Generation chat template differs from the table protocol")
    expected_template_hash = hashlib.sha256(
        EXPECTED_GENERATION_TEMPLATE.encode("utf-8")
    ).hexdigest()
    if prompt.get("template_sha256") != expected_template_hash:
        raise RuntimeError("Generation chat-template hash does not verify")
    for row in rows:
        expected_prompt = EXPECTED_GENERATION_TEMPLATE.format(
            question=str(row["question"])
        )
        if row.get("prompt_sha256") != hashlib.sha256(
            expected_prompt.encode("utf-8")
        ).hexdigest():
            raise RuntimeError(f"Generation prompt hash failed for idx={row['idx']}")
    decoding = spec.get("decoding") or {}
    if decoding.get("strategy") != "greedy" or decoding.get("do_sample") is not False:
        raise RuntimeError("Generation is not marked as greedy decoding")
    if decoding.get("max_new_tokens") != expected_max_new_tokens:
        raise RuntimeError("Generation max_new_tokens differs from the table protocol")
    for row in rows:
        token_count = row.get("generated_token_count")
        if (
            not isinstance(token_count, int)
            or isinstance(token_count, bool)
            or not 0 <= token_count <= expected_max_new_tokens
        ):
            raise RuntimeError(f"Invalid generated token count for idx={row['idx']}")

    if (expected_stop_policy_version is None) != (expected_eos_token_ids is None):
        raise RuntimeError(
            "Expected stop-policy version and EOS token IDs must be provided together"
        )
    if expected_stop_policy_version is not None:
        if decoding.get("stop_policy_version") != expected_stop_policy_version:
            raise RuntimeError("Generation stop-policy version differs from the requested protocol")
        if decoding.get("effective_eos_token_ids") != expected_eos_token_ids:
            raise RuntimeError("Generation effective EOS list differs from the requested protocol")
        required_decoding_metadata = {
            "effective_eos_token_id_source": str,
            "generation_config_load_source": str,
            "generation_config_eos_token_ids": list,
            "model_config_eos_token_ids": list,
            "tokenizer_eos_token_ids": list,
            "pad_token_id": int,
            "stop_rule": str,
            "transformers_version": str,
        }
        for field, expected_type in required_decoding_metadata.items():
            value = decoding.get(field)
            if not isinstance(value, expected_type) or isinstance(value, bool):
                raise RuntimeError(f"Generation decoding metadata is invalid: {field}")
        if decoding.get("generated_token_count_includes_terminal_eos") is not True:
            raise RuntimeError("Generation token-count EOS convention is missing")

        expected_eos_set = set(expected_eos_token_ids or [])
        for row in rows:
            idx = row["idx"]
            count = row["generated_token_count"]
            terminated = row.get("terminated_by_eos")
            hit_limit = row.get("hit_max_new_tokens")
            terminal = row.get("terminal_eos_token_id")
            reason = row.get("termination_reason")
            if not isinstance(terminated, bool) or not isinstance(hit_limit, bool):
                raise RuntimeError(f"Invalid termination flags for idx={idx}")
            if reason == "eos":
                valid = (
                    terminated
                    and not hit_limit
                    and terminal in expected_eos_set
                    and count >= 1
                )
            elif reason == "max_new_tokens":
                valid = (
                    not terminated
                    and hit_limit
                    and terminal is None
                    and count == expected_max_new_tokens
                )
            else:
                valid = False
            if not valid:
                raise RuntimeError(f"Inconsistent termination metadata for idx={idx}")

        expected_summary = {
            "eos": sum(row.get("termination_reason") == "eos" for row in rows),
            "max_new_tokens": sum(
                row.get("termination_reason") == "max_new_tokens" for row in rows
            ),
            "other": sum(row.get("termination_reason") == "other" for row in rows),
            "by_terminal_eos_token_id": {
                str(token_id): sum(
                    row.get("terminal_eos_token_id") == token_id for row in rows
                )
                for token_id in expected_eos_token_ids or []
            },
        }
        if metadata.get("termination_summary") != expected_summary:
            raise RuntimeError("Generation termination summary does not verify")

    model = spec.get("model") or {}
    if model.get("method") != expected_method:
        raise RuntimeError(
            f"Generation method {model.get('method')!r} does not match {expected_method!r}"
        )
    if expected_composition_mode is not None and model.get(
        "composition_mode"
    ) != expected_composition_mode:
        raise RuntimeError(
            "Generation composition mode differs from the requested judge protocol"
        )


def validate_recap_b0_model_spec(metadata: dict[str, Any], method: str) -> None:
    spec = metadata.get("spec") or {}
    model = spec.get("model") or {}
    report = spec.get("frozen_standard_report") or {}
    required = {
        "method": (method, "RECAP"),
        "base_model_id": (
            report.get("base_model_id", model.get("base")),
            RECAP_B0_BASE_MODEL_ID,
        ),
        "weight_a1": (model.get("weight_a1"), -2.1),
        "weight_a2": (model.get("weight_a2"), 2.2),
        "filter_a1": (model.get("top_logit_filter_a1"), 0.00017),
        "filter_a2": (model.get("top_logit_filter_a2"), 0.00017),
        "composition_mode": (model.get("composition_mode"), "reference_delta"),
        "alignment_enabled": (model.get("alignment_enabled"), False),
        "gate_enabled": (model.get("gate_enabled"), False),
        "sequence_router_enabled": (model.get("sequence_router_enabled"), False),
        "batch_size": ((spec.get("decoding") or {}).get("batch_size"), 1),
        "references_share_realpath": (model.get("references_share_realpath"), False),
        "report_complete": (report.get("validation_complete"), True),
        "memorization_score": (
            report.get("memorization_score"),
            0.7188500647349938,
        ),
        "retain_utility_score": (
            report.get("retain_utility_score"),
            0.5930391652906767,
        ),
    }
    mismatches = {}
    if model.get("kind") != "recap":
        mismatches["kind"] = {
            "actual": model.get("kind"),
            "required": "recap",
        }
    for field, (actual, expected) in required.items():
        if (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and isinstance(expected, (int, float))
            and not isinstance(expected, bool)
        ):
            matches = math.isclose(
                float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12
            )
        else:
            matches = actual == expected
        if not matches:
            mismatches[field] = {"actual": actual, "required": expected}
    if not str(model.get("a1", "")).endswith("/checkpoint-168"):
        mismatches["a1_checkpoint"] = model.get("a1")
    if not str(model.get("a2", "")).endswith("/checkpoint-144"):
        mismatches["a2_checkpoint"] = model.get("a2")
    reference_a1 = model.get("reference_a1_path")
    reference_a2 = model.get("reference_a2_path")
    if not reference_a1:
        mismatches["reference_a1_path"] = reference_a1
    if not reference_a2:
        mismatches["reference_a2_path"] = reference_a2
    if not model.get("reference_a1_signature") or not model.get(
        "reference_a2_signature"
    ):
        mismatches["reference_signatures"] = "missing"
    if not report.get("sha256"):
        mismatches["standard_report_sha256"] = report.get("sha256")
    if mismatches:
        raise RuntimeError(f"Generation is not frozen RECAP B0: {mismatches}")


def prompt_for(axis: str, row: dict[str, Any]) -> str:
    if axis == "semantic_distance":
        return SEMANTIC_DISTANCE_TEMPLATE.format(
            inp=row["question"], truth=row["ground_truth"], gen=row["generation"]
        )
    if axis == "naturalness":
        return NATURALNESS_TEMPLATE.format(gen=row["generation"])
    raise ValueError(f"Unknown judge axis: {axis}")


class Journal:
    """Thread-safe append-only successful-call journal."""

    def __init__(
        self,
        path: Path,
        run_fingerprint: str,
        allowed_keys: set[tuple[int, str]],
        expected_prompt_hashes: dict[tuple[int, str], str] | None = None,
    ):
        self.path = path
        self.run_fingerprint = run_fingerprint
        self.allowed_keys = allowed_keys
        self.expected_prompt_hashes = expected_prompt_hashes or {}
        self.lock = threading.Lock()
        self.records: dict[tuple[int, str], dict[str, Any]] = {}
        if path.exists():
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("run_fingerprint") != run_fingerprint:
                    raise RuntimeError(
                        f"Incompatible journal record at {path}:{line_number}"
                    )
                key = (int(record["idx"]), str(record["axis"]))
                if key not in allowed_keys:
                    raise RuntimeError(f"Unknown journal key at {path}:{line_number}: {key}")
                if key in self.records:
                    raise RuntimeError(f"Duplicate journal key at {path}:{line_number}: {key}")
                raw_score = record.get("score")
                if not isinstance(raw_score, (int, float)) or isinstance(raw_score, bool):
                    raise RuntimeError(f"Invalid journal score at {path}:{line_number}")
                if parse_score(str(record.get("raw_response", ""))) != float(raw_score):
                    raise RuntimeError(f"Journal raw/parsed score mismatch at {path}:{line_number}")
                expected_prompt_hash = self.expected_prompt_hashes.get(key)
                if expected_prompt_hash and record.get(
                    "request_prompt_sha256"
                ) != expected_prompt_hash:
                    raise RuntimeError(f"Journal prompt hash mismatch at {path}:{line_number}")
                self.records[key] = record

    def append(self, record: dict[str, Any]) -> None:
        key = (int(record["idx"]), str(record["axis"]))
        with self.lock:
            if key not in self.allowed_keys:
                raise RuntimeError(f"Refusing unknown journal key: {key}")
            if key in self.records:
                raise RuntimeError(f"Refusing duplicate journal key: {key}")
            expected_prompt_hash = self.expected_prompt_hashes.get(key)
            if expected_prompt_hash and record.get(
                "request_prompt_sha256"
            ) != expected_prompt_hash:
                raise RuntimeError(f"Refusing journal record with wrong prompt hash: {key}")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.records[key] = record


def extract_response(response: requests.Response) -> tuple[str, dict[str, Any]]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Judge returned HTTP {response.status_code} with non-JSON body"
        ) from exc
    if response.status_code >= 400:
        message = (payload.get("error") or {}).get("message") if isinstance(payload, dict) else None
        raise RuntimeError(f"Judge HTTP {response.status_code}: {message or payload!r}")
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Malformed judge response: {payload!r}") from exc
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        content = "".join(text_parts)
    if not isinstance(content, str):
        raise RuntimeError("Judge response content is not text")
    audit = {
        "response_id": payload.get("id"),
        "actual_model": payload.get("model"),
        "provider": payload.get("provider"),
        "created": payload.get("created"),
        "service_tier": payload.get("service_tier"),
        "system_fingerprint": payload.get("system_fingerprint"),
        "finish_reason": (
            payload.get("choices", [{}])[0].get("finish_reason")
            if isinstance(payload.get("choices"), list) and payload.get("choices")
            else None
        ),
        "usage": payload.get("usage"),
    }
    return content, audit


def judge_headers(api_profile: str, api_key: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if api_profile == "openrouter":
        headers["X-Title"] = "RECAP semantic robustness evaluation"
    elif api_profile != "openai":
        raise ValueError(f"Unknown judge API profile: {api_profile}")
    return headers


def judge_request_payload(
    *,
    api_profile: str,
    model: str,
    prompt: str,
    temperature: float | None,
    max_tokens: int,
    reasoning_tokens: int,
    reasoning_effort: str | None,
    seed: int | None,
) -> dict[str, Any]:
    """Build a provider-specific request without silently translating controls."""

    request_payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if api_profile == "openrouter":
        if reasoning_effort is not None:
            raise ValueError("--reasoning-effort is incompatible with openrouter profile")
        if temperature is not None:
            request_payload["temperature"] = temperature
        request_payload["max_tokens"] = max_tokens
        request_payload["reasoning"] = {"max_tokens": reasoning_tokens}
    elif api_profile == "openai":
        if reasoning_tokens != 0:
            raise ValueError("--reasoning-tokens is incompatible with openai profile")
        if temperature is not None:
            request_payload["temperature"] = temperature
        request_payload["max_completion_tokens"] = max_tokens
        if reasoning_effort is not None:
            request_payload["reasoning_effort"] = reasoning_effort
    else:
        raise ValueError(f"Unknown judge API profile: {api_profile}")
    if seed is not None:
        request_payload["seed"] = seed
    return request_payload


def call_judge(
    *,
    url: str,
    api_key: str,
    api_profile: str,
    model: str,
    prompt: str,
    temperature: float | None,
    max_tokens: int,
    reasoning_tokens: int,
    reasoning_effort: str | None,
    timeout_seconds: float,
    max_retries: int,
    seed: int | None,
) -> dict[str, Any]:
    headers = judge_headers(api_profile, api_key)
    request_payload = judge_request_payload(
        api_profile=api_profile,
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_tokens=reasoning_tokens,
        reasoning_effort=reasoning_effort,
        seed=seed,
    )

    attempts = []
    for attempt in range(1, max_retries + 1):
        started = time.monotonic()
        try:
            response = requests.post(
                url,
                headers=headers,
                json=request_payload,
                timeout=timeout_seconds,
            )
            raw_text, audit = extract_response(response)
            score = parse_score(raw_text)
            return {
                "score": score,
                "raw_response": raw_text,
                "request_prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "latency_seconds": time.monotonic() - started,
                "attempt": attempt,
                "prior_failures": attempts,
                **audit,
            }
        except Exception as exc:  # retry transport, provider, and format failures
            attempts.append({"attempt": attempt, "error": str(exc)})
            if attempt == max_retries:
                raise RuntimeError(
                    f"Judge failed after {max_retries} attempts: {exc}"
                ) from exc
            delay = min(30.0, 2.0 ** (attempt - 1)) + random.random() * 0.25
            time.sleep(delay)
    raise AssertionError("unreachable")


def validate_complete_result(
    path: Path,
    run_fingerprint: str,
    generation_rows: list[dict[str, Any]],
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_item = payload.get("per_item")
    expected_indices = [int(row["idx"]) for row in generation_rows]
    coverage = payload.get("coverage") or {}
    expected_axis_scores = len(expected_indices) * len(AXES)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"Existing judge output has the wrong schema: {path}")
    if payload.get("run_fingerprint") != run_fingerprint:
        raise RuntimeError(f"Existing judge output has an incompatible config: {path}")
    if coverage != {
        "expected_items": len(expected_indices),
        "scored_items": len(expected_indices),
        "expected_axis_scores": expected_axis_scores,
        "scored_axis_scores": expected_axis_scores,
        "failed_axis_scores": 0,
        "complete": True,
    }:
        raise RuntimeError(f"Existing judge output is not complete: {path}")
    if not isinstance(per_item, list) or [item.get("idx") for item in per_item] != expected_indices:
        raise RuntimeError(f"Existing judge output has invalid item coverage: {path}")
    if payload.get("per_item_sha256") != compact_json_sha256(per_item):
        raise RuntimeError(f"Existing judge per-item hash does not verify: {path}")
    semantic_values = []
    naturalness_values = []
    for item in per_item:
        for axis, destination in (
            ("semantic_distance", semantic_values),
            ("naturalness", naturalness_values),
        ):
            score = item.get(axis)
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                raise RuntimeError(f"Existing judge output has an invalid {axis} score")
            if not math.isfinite(float(score)) or not 0.0 <= float(score) <= 5.0:
                raise RuntimeError(f"Existing judge output has an out-of-range {axis} score")
            destination.append(float(score))
    semantic_mean = sum(semantic_values) / len(semantic_values)
    naturalness_mean = sum(naturalness_values) / len(naturalness_values)
    if not math.isclose(
        float(payload.get("semantic_distance_mean", math.nan)),
        semantic_mean,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Existing semantic-distance mean does not recompute")
    if not math.isclose(
        float(payload.get("naturalness_mean", math.nan)),
        naturalness_mean,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Existing naturalness mean does not recompute")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gens", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--judge-model", default="google/gemini-2.5-flash")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument(
        "--api-profile", choices=["openrouter", "openai"], default="openrouter"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--temperature",
        type=float,
        help="Omit for GPT-5-class Azure/OpenAI deployments",
    )
    parser.add_argument("--reasoning-tokens", type=int, default=0)
    parser.add_argument(
        "--reasoning-effort", choices=["low", "medium", "high"]
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--request-seed", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--expected-items", type=int, default=200)
    parser.add_argument(
        "--expected-indices-sha256", default=HISTORICAL_SAMPLE_SHA256
    )
    parser.add_argument("--expected-max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--expected-composition-mode", choices=["raw", "reference_delta"]
    )
    parser.add_argument("--expected-stop-policy-version")
    parser.add_argument("--expected-eos-token-ids", type=parse_token_id_csv)
    parser.add_argument("--require-recap-b0", action="store_true")
    parser.add_argument("--journal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Preserve the historical deterministic OpenRouter default while allowing
    # GPT-5-class OpenAI/Azure requests to omit unsupported temperature.
    if args.api_profile == "openrouter" and args.temperature is None:
        args.temperature = 0.0
    if args.workers < 1 or args.max_retries < 1 or args.max_tokens < 1:
        raise ValueError("--workers, --max-retries, and --max-tokens must be positive")
    if args.api_profile == "openrouter" and args.reasoning_effort is not None:
        raise ValueError("--reasoning-effort is incompatible with openrouter profile")
    if args.api_profile == "openai" and args.reasoning_tokens != 0:
        raise ValueError("--reasoning-tokens is incompatible with openai profile")
    out_path = Path(args.out)
    generation_metadata, rows = load_generations(Path(args.gens))
    if len(rows) != args.expected_items:
        raise RuntimeError(
            f"Expected exactly {args.expected_items} generation rows, found {len(rows)}"
        )

    validate_generation_protocol(
        generation_metadata,
        rows,
        expected_items=args.expected_items,
        expected_indices_sha256=args.expected_indices_sha256,
        expected_max_new_tokens=args.expected_max_new_tokens,
        expected_method=args.method,
        expected_composition_mode=args.expected_composition_mode,
        expected_stop_policy_version=args.expected_stop_policy_version,
        expected_eos_token_ids=args.expected_eos_token_ids,
    )
    if args.require_recap_b0:
        validate_recap_b0_model_spec(generation_metadata, args.method)
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    if args.api_profile == "openai":
        request_options: dict[str, Any] = {
            "max_completion_tokens": args.max_tokens,
        }
        if args.reasoning_effort is not None:
            request_options["reasoning_effort"] = args.reasoning_effort
        if args.temperature is not None:
            request_options["temperature"] = args.temperature
    else:
        request_options = {
            "max_tokens": args.max_tokens,
            "reasoning": {"max_tokens": args.reasoning_tokens},
        }
        if args.temperature is not None:
            request_options["temperature"] = args.temperature
    if args.request_seed is not None:
        request_options["seed"] = args.request_seed
    run_spec = {
        "method": args.method,
        "generations_sha256": hashlib.sha256(Path(args.gens).read_bytes()).hexdigest(),
        "judge_model": args.judge_model,
        "api_profile": args.api_profile,
        "api_key_env": args.api_key_env,
        "endpoint": endpoint,
        "request_options": request_options,
        "workers": args.workers,
        "timeout_seconds": args.timeout_seconds,
        "max_retries": args.max_retries,
        "generation_protocol_expectations": {
            "stop_policy_version": args.expected_stop_policy_version,
            "effective_eos_token_ids": args.expected_eos_token_ids,
        },
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "strict_score_format": "full-match decimal in [0,5] with exactly two digits",
        "semantic_distance_prompt_sha256": hashlib.sha256(
            SEMANTIC_DISTANCE_TEMPLATE.encode("utf-8")
        ).hexdigest(),
        "naturalness_prompt_sha256": hashlib.sha256(
            NATURALNESS_TEMPLATE.encode("utf-8")
        ).hexdigest(),
    }
    run_fingerprint = compact_json_sha256(run_spec)
    if out_path.exists():
        validate_complete_result(out_path, run_fingerprint, rows)
        print(f"[laaj] verified complete output: {out_path}", flush=True)
        return
    journal_path = Path(args.journal) if args.journal else out_path.with_suffix(".journal.jsonl")
    row_by_idx = {int(row["idx"]): row for row in rows}
    expected_keys = [(int(row["idx"]), axis) for row in rows for axis in AXES]
    allowed_keys = set(expected_keys)
    expected_prompt_hashes = {
        key: hashlib.sha256(
            prompt_for(key[1], row_by_idx[key[0]]).encode("utf-8")
        ).hexdigest()
        for key in expected_keys
    }
    journal = Journal(
        journal_path, run_fingerprint, allowed_keys, expected_prompt_hashes
    )
    pending = [key for key in expected_keys if key not in journal.records]
    print(
        f"[laaj] method={args.method} items={len(rows)} axes={len(expected_keys)} "
        f"resumed={len(expected_keys) - len(pending)} pending={len(pending)} "
        f"model={args.judge_model}",
        flush=True,
    )

    failures: list[dict[str, Any]] = []
    api_key = os.environ.get(args.api_key_env) if pending else None
    if pending and not api_key:
        raise RuntimeError(f"Required API key environment variable is unset: {args.api_key_env}")

    def score_key(key: tuple[int, str]) -> dict[str, Any]:
        idx, axis = key
        prompt = prompt_for(axis, row_by_idx[idx])
        assert api_key is not None
        result = call_judge(
            url=endpoint,
            api_key=api_key,
            api_profile=args.api_profile,
            model=args.judge_model,
            prompt=prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            reasoning_tokens=args.reasoning_tokens,
            reasoning_effort=args.reasoning_effort,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            seed=args.request_seed,
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "run_fingerprint": run_fingerprint,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "idx": idx,
            "axis": axis,
            **result,
        }

    # Score one real axis synchronously before submitting the fan-out.  This
    # is both useful work and an authentication/model/strict-format preflight.
    if pending:
        preflight_key = pending.pop(0)
        try:
            journal.append(score_key(preflight_key))
        except Exception as exc:
            raise RuntimeError(
                f"Judge preflight failed for idx={preflight_key[0]} "
                f"axis={preflight_key[1]}: {exc}"
            ) from exc
        print(
            f"[laaj] preflight passed idx={preflight_key[0]} axis={preflight_key[1]}",
            flush=True,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(score_key, key): key for key in pending}
        completed = len(expected_keys) - len(pending)
        for future in as_completed(futures):
            key = futures[future]
            try:
                journal.append(future.result())
            except Exception as exc:
                failures.append({"idx": key[0], "axis": key[1], "error": str(exc)})
                print(f"[laaj] failed idx={key[0]} axis={key[1]}: {exc}", flush=True)
            completed += 1
            if completed % 20 == 0 or completed == len(expected_keys):
                print(f"[laaj] {completed}/{len(expected_keys)}", flush=True)

    missing = [key for key in expected_keys if key not in journal.records]
    if failures or missing:
        failure_path = out_path.with_suffix(".failures.json")
        atomic_write_json(
            failure_path,
            {
                "run_fingerprint": run_fingerprint,
                "failures": failures,
                "missing": [{"idx": idx, "axis": axis} for idx, axis in missing],
            },
        )
        raise RuntimeError(
            f"Incomplete judge coverage: {len(missing)} of {len(expected_keys)} axis scores missing; "
            f"rerun to resume from {journal_path}"
        )

    per_item = []
    for row in rows:
        idx = int(row["idx"])
        sem = journal.records[(idx, "semantic_distance")]
        nat = journal.records[(idx, "naturalness")]
        per_item.append(
            {
                "idx": idx,
                "semantic_distance": sem["score"],
                "naturalness": nat["score"],
                "semantic_distance_audit": sem,
                "naturalness_audit": nat,
            }
        )
    semantic_values = [item["semantic_distance"] for item in per_item]
    naturalness_values = [item["naturalness"] for item in per_item]
    semantic_mean = sum(semantic_values) / len(semantic_values)
    naturalness_mean = sum(naturalness_values) / len(naturalness_values)
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_fingerprint": run_fingerprint,
        "run_spec": run_spec,
        "generation_metadata": generation_metadata,
        "coverage": {
            "expected_items": len(rows),
            "scored_items": len(per_item),
            "expected_axis_scores": len(expected_keys),
            "scored_axis_scores": len(expected_keys),
            "failed_axis_scores": 0,
            "complete": True,
        },
        "naturalness_mean": naturalness_mean,
        "semantic_distance_mean": semantic_mean,
        "table_rounding": {
            "naturalness_1dp": f"{naturalness_mean:.1f}",
            "semantic_distance_1dp": f"{semantic_mean:.1f}",
        },
        "per_item_sha256": compact_json_sha256(per_item),
        "per_item": per_item,
    }
    atomic_write_json(out_path, result)
    failure_path = out_path.with_suffix(".failures.json")
    if failure_path.exists():
        failure_path.unlink()
    print(
        f"[laaj] complete Nat={naturalness_mean:.6f} SemDist={semantic_mean:.6f} "
        f"table={naturalness_mean:.1f}/{semantic_mean:.1f} -> {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
