#!/usr/bin/env python
"""Generate the fixed TOFU Forget10 sample used by the robustness table.

The frozen robustness protocol samples 200 of the 400 Forget10 rows
with ``np.random.default_rng(42).choice(..., replace=False)``.  This version
keeps that protocol and adds the reference-delta arguments required by RECAP.

The output is a versioned JSON object with complete prompt, sampling, model,
and decoding provenance.  A ``.partial.json`` sibling is written atomically
after each batch so interrupted GPU runs can resume safely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Iterable

import numpy as np


SYSTEM_PROMPT = "You are a helpful assistant."
SYSTEM_BLOCK = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    f"{SYSTEM_PROMPT}<|eot_id|>"
)
USER_BLOCK = "<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>"
ASSISTANT_OPEN = "<|start_header_id|>assistant<|end_header_id|>\n\n"
SCHEMA_VERSION = 2
STOP_POLICY_VERSION = "model_multi_eos_v1"
RECAP_B0_EOS_TOKEN_IDS = [128001, 128008, 128009]
RECAP_B0_BASE_MODEL_ID = "open-unlearning/tofu_Llama-3.1-8B-Instruct_full"


def build_prompt(question: str) -> str:
    """Render the exact chat prompt used by the historical table runner."""

    return SYSTEM_BLOCK + USER_BLOCK.format(question=question) + ASSISTANT_OPEN


def sample_indices(
    n_total: int = 400, n_sample: int = 200, seed: int = 42
) -> list[int]:
    """Return the table's deterministic uniform-without-replacement sample."""

    if not 0 < n_sample <= n_total:
        raise ValueError(f"n_sample must be in [1, {n_total}], got {n_sample}")
    rng = np.random.default_rng(seed)
    return sorted(int(index) for index in rng.choice(n_total, n_sample, replace=False))


def compact_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_signature(raw_path: str | None) -> dict[str, Any] | None:
    """Fingerprint local artifact identity without hashing multi-GB tensors.

    Tensor names, sizes, and nanosecond mtimes detect in-place replacement;
    small JSON configs are content-hashed.  Hub IDs are recorded verbatim and
    are subsequently resolved by ``from_pretrained`` from the frozen cache.
    """

    if raw_path is None:
        return None
    path = Path(raw_path)
    if not path.exists():
        return {"requested": raw_path, "kind": "hub_or_unresolved_id"}
    resolved = path.resolve()
    if resolved.is_file():
        stat = resolved.stat()
        return {
            "requested": raw_path,
            "resolved": str(resolved),
            "kind": "file",
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": file_sha256(resolved),
        }
    entries = []
    for child in sorted(item for item in resolved.rglob("*") if item.is_file()):
        stat = child.stat()
        entry: dict[str, Any] = {
            "path": str(child.relative_to(resolved)),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if child.suffix.lower() in {".json", ".yaml", ".yml"} and stat.st_size <= 10_000_000:
            entry["sha256"] = file_sha256(child)
        entries.append(entry)
    return {
        "requested": raw_path,
        "resolved": str(resolved),
        "kind": "directory",
        "entry_count": len(entries),
        "tree_stat_sha256": compact_json_sha256(entries),
    }


def model_artifact_signature(
    raw_path: str | None, *, offline: bool
) -> dict[str, Any] | None:
    signature = artifact_signature(raw_path)
    if signature is None or signature.get("kind") != "hub_or_unresolved_id":
        return signature
    try:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(repo_id=raw_path, local_files_only=offline)
    except Exception as exc:
        return {**signature, "cache_resolution_error": type(exc).__name__}
    return {
        "requested": raw_path,
        "kind": "huggingface_snapshot",
        "snapshot": artifact_signature(snapshot),
        "revision": Path(snapshot).name,
    }


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def current_git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _same_value(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
    return actual == expected


def _same_artifact_reference(actual: Any, expected: Any) -> bool:
    """Compare Hub IDs literally and existing local paths canonically."""

    if not isinstance(actual, str) or not isinstance(expected, str):
        return actual == expected
    actual_path = Path(actual).expanduser()
    expected_path = Path(expected).expanduser()
    if actual_path.exists() and expected_path.exists():
        return actual_path.resolve() == expected_path.resolve()
    return actual == expected


def validate_standard_report(args: argparse.Namespace) -> dict[str, Any] | None:
    """Bind a generation invocation to the frozen standard-evaluation report."""

    if not args.standard_report:
        if args.require_recap_b0:
            raise ValueError("--require-recap-b0 requires --standard-report")
        return None
    report_path = Path(args.standard_report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metadata = report.get("metadata") or {}
    reference_a1, reference_a2 = resolved_reference_paths(args)
    report_reference_a1 = metadata.get("reference_a1_path")
    report_reference_a2 = metadata.get("reference_a2_path")
    display_references = metadata.get("reference_path")
    if (
        (not report_reference_a1 or not report_reference_a2)
        and isinstance(display_references, str)
        and display_references.startswith("A1=")
        and ";A2=" in display_references
    ):
        report_reference_a1, report_reference_a2 = display_references[3:].split(
            ";A2=", 1
        )
    report_base_artifact = metadata.get(
        "base_model_artifact", metadata.get("base_model")
    )
    comparisons = {
        "weight_a1": (args.weight_a1, metadata.get("weight_a1")),
        "weight_a2": (args.weight_a2, metadata.get("weight_a2")),
        "top_filter_a1": (
            args.top_filter if args.top_filter_a1 is None else args.top_filter_a1,
            metadata.get("top_filter_a1"),
        ),
        "top_filter_a2": (
            args.top_filter if args.top_filter_a2 is None else args.top_filter_a2,
            metadata.get("top_filter_a2"),
        ),
        "composition_mode": (args.composition_mode, metadata.get("composition_mode")),
    }
    # The published adapters may be relocated from the original experiment
    # machine.  For the frozen B0 profile, validate step suffixes and model
    # structure below instead of comparing machine-specific absolute paths.
    if not args.require_recap_b0:
        comparisons.update(
            {
                "a1_checkpoint": (args.a1, metadata.get("a1_checkpoint")),
                "a2_checkpoint": (args.a2, metadata.get("a2_checkpoint")),
                "reference_a1_path": (reference_a1, report_reference_a1),
                "reference_a2_path": (reference_a2, report_reference_a2),
            }
        )
    mismatches = {
        field: {"command": actual, "report": expected}
        for field, (actual, expected) in comparisons.items()
        if not _same_value(actual, expected)
    }
    if not _same_artifact_reference(args.base, report_base_artifact):
        mismatches["base_model_artifact"] = {
            "command": args.base,
            "report": report_base_artifact,
        }
    if mismatches:
        raise ValueError(
            f"Generation arguments disagree with the standard report: {mismatches}"
        )
    for field in ("alignment_enabled", "gate_enabled", "sequence_router_enabled"):
        if metadata.get(field) is not False:
            raise ValueError(f"Frozen report unexpectedly enables {field}")

    if args.require_recap_b0:
        required = {
            "method_label": (args.method, "RECAP"),
            "split": (metadata.get("split"), "forget10"),
            "base": (
                metadata.get("base_model_id", metadata.get("base_model")),
                RECAP_B0_BASE_MODEL_ID,
            ),
            "composition": (metadata.get("composition_mode"), "reference_delta"),
            "weight_a1": (metadata.get("weight_a1"), -2.1),
            "weight_a2": (metadata.get("weight_a2"), 2.2),
            "filter_a1": (metadata.get("top_filter_a1"), 0.00017),
            "filter_a2": (metadata.get("top_filter_a2"), 0.00017),
            "a1_steps": (metadata.get("a1_train_steps"), 168),
            "a2_steps": (metadata.get("a2_train_steps"), 144),
            "a1_rank": (metadata.get("a1_lora_r"), 64),
            "a2_rank": (metadata.get("a2_lora_r"), 64),
            "sample_n": (args.n, 200),
            "sample_seed": (args.seed, 42),
            "max_new_tokens": (args.max_new_tokens, 128),
            "batch_size": (args.batch_size, 1),
            "dataset": (args.dataset, "locuslab/TOFU"),
            "dataset_config": (args.dataset_config, "forget10_perturbed"),
            "dataset_split": (args.dataset_split, "train"),
            "selection_retain_access": (
                metadata.get("selection_retain_access"),
                True,
            ),
        }
        required_mismatches = {
            field: {"actual": actual, "required": expected}
            for field, (actual, expected) in required.items()
            if not _same_value(actual, expected)
        }
        if args.kind != "recap":
            required_mismatches["kind"] = {
                "actual": args.kind,
                "required": "recap",
            }
        if not str(metadata.get("a1_checkpoint", "")).endswith("/checkpoint-168"):
            required_mismatches["a1_checkpoint_suffix"] = metadata.get("a1_checkpoint")
        if not str(metadata.get("a2_checkpoint", "")).endswith("/checkpoint-144"):
            required_mismatches["a2_checkpoint_suffix"] = metadata.get("a2_checkpoint")
        derived = report.get("derived") or {}
        if not _same_value(derived.get("memorization_score"), 0.7188500647349938):
            required_mismatches["memorization_score"] = derived.get("memorization_score")
        if not _same_value(derived.get("retain_utility_score"), 0.5930391652906767):
            required_mismatches["retain_utility_score"] = derived.get("retain_utility_score")
        if not _same_value(derived.get("aggregate_score"), 0.6499119477507226):
            required_mismatches["aggregate_score"] = derived.get("aggregate_score")
        validation = report.get("validation") or {}
        if validation.get("complete") is not True:
            required_mismatches["report_complete"] = validation.get("complete")
        if required_mismatches:
            raise ValueError(f"Report/invocation is not frozen RECAP B0: {required_mismatches}")

    return {
        "path": str(report_path.resolve()),
        "sha256": file_sha256(report_path),
        "validation_complete": (report.get("validation") or {}).get("complete"),
        "memorization_score": (report.get("derived") or {}).get("memorization_score"),
        "retain_utility_score": (report.get("derived") or {}).get("retain_utility_score"),
        "aggregate_score": (report.get("derived") or {}).get("aggregate_score"),
        "selection_retain_access": metadata.get("selection_retain_access"),
        "base_model_id": metadata.get("base_model_id", metadata.get("base_model")),
    }


def resolved_reference_paths(args: argparse.Namespace) -> tuple[str | None, str | None]:
    if args.composition_mode != "reference_delta":
        return args.reference_a1, args.reference_a2
    reference_a1 = args.reference_a1
    reference_a2 = args.reference_a2
    if reference_a1 in {None, "null", "auto"}:
        reference_a1 = str(Path(args.a1).parent / "fullmodel")
    if reference_a2 in {None, "null", "auto"}:
        reference_a2 = str(Path(args.a2).parent / "fullmodel")
    return reference_a1, reference_a2


def semantic_spec(
    args: argparse.Namespace,
    indices: list[int],
    sampled_qa_sha256: str,
    decoding_policy: dict[str, Any],
) -> dict[str, Any]:
    top_a1 = args.top_filter if args.top_filter_a1 is None else args.top_filter_a1
    top_a2 = args.top_filter if args.top_filter_a2 is None else args.top_filter_a2
    reference_a1, reference_a2 = resolved_reference_paths(args)
    script_path = Path(__file__).resolve()
    wrapper_path = Path(args.ou_repo).resolve() / "src" / "model" / "recap.py"
    return {
        "implementation": {
            "generator_path": str(script_path),
            "generator_sha256": file_sha256(script_path),
            "model_wrapper": artifact_signature(str(wrapper_path)),
        },
        "frozen_standard_report": getattr(
            args, "standard_report_provenance", None
        ),
        "dataset": {
            "name": args.dataset,
            "config": args.dataset_config,
            "split": args.dataset_split,
            "expected_rows": args.expected_rows,
            "question_field": args.question_field,
            "answer_field": args.answer_field,
            "sampled_idx_question_answer_sha256": sampled_qa_sha256,
        },
        "sampling": {
            "algorithm": "numpy.random.Generator(PCG64).choice_without_replacement_sorted",
            "numpy_version": np.__version__,
            "seed": args.seed,
            "n": args.n,
            "indices": indices,
            "indices_sha256": compact_json_sha256(indices),
        },
        "prompt": {
            "system": SYSTEM_PROMPT,
            "template": SYSTEM_BLOCK + USER_BLOCK + ASSISTANT_OPEN,
            "template_sha256": hashlib.sha256(
                (SYSTEM_BLOCK + USER_BLOCK + ASSISTANT_OPEN).encode("utf-8")
            ).hexdigest(),
        },
        "decoding": {
            "strategy": "greedy",
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            **decoding_policy,
        },
        "model": {
            "method": args.method,
            "kind": args.kind,
            "base": args.base,
            "base_signature": model_artifact_signature(args.base, offline=args.offline),
            "tokenizer": args.tokenizer or args.base,
            "tokenizer_signature": model_artifact_signature(
                args.tokenizer or args.base, offline=args.offline
            ),
            "a1": args.a1,
            "a1_signature": artifact_signature(args.a1),
            "a2": args.a2,
            "a2_signature": artifact_signature(args.a2),
            "weight_a1": args.weight_a1,
            "weight_a2": args.weight_a2,
            "top_logit_filter": args.top_filter,
            "top_logit_filter_a1": top_a1,
            "top_logit_filter_a2": top_a2,
            "composition_mode": args.composition_mode,
            "reference_a1_path": reference_a1,
            "reference_a1_signature": artifact_signature(reference_a1),
            "reference_a2_path": reference_a2,
            "reference_a2_signature": artifact_signature(reference_a2),
            "alignment_enabled": False,
            "gate_enabled": False,
            "sequence_router_enabled": False,
            "torch_dtype": args.dtype,
            "attention_implementation": args.attn_implementation,
            "offline_cache_only": args.offline,
            "references_share_realpath": bool(
                reference_a1
                and reference_a2
                and os.path.realpath(reference_a1) == os.path.realpath(reference_a2)
            ),
        },
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.kind != "recap":
        raise ValueError("--kind must be recap")
    if not args.a1 or not args.a2:
        raise ValueError("--a1 and --a2 are required for RECAP")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")


def _torch_dtype(torch_module: Any, name: str) -> Any:
    try:
        return getattr(torch_module, name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported torch dtype: {name}") from exc


def normalize_eos_token_ids(value: Any, *, source: str = "eos_token_id") -> list[int]:
    """Normalize a Transformers EOS value without losing multi-token stops."""

    if isinstance(value, bool):
        raise ValueError(f"{source} must contain token IDs, not booleans")
    if isinstance(value, int):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        raise ValueError(f"{source} must be an int, list, or tuple, got {value!r}")
    if not values:
        raise ValueError(f"{source} must not be empty")
    normalized: list[int] = []
    seen: set[int] = set()
    for token_id in values:
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise ValueError(f"{source} contains an invalid token ID: {token_id!r}")
        if token_id not in seen:
            normalized.append(token_id)
            seen.add(token_id)
    return normalized


def parse_eos_token_id_csv(text: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in text.split(",")]
        normalized = normalize_eos_token_ids(
            values, source="--expected-eos-token-ids"
        )
        if len(normalized) != len(values):
            raise ValueError("--expected-eos-token-ids must not contain duplicates")
        return normalized
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _optional_eos_token_ids(value: Any, *, source: str) -> list[int] | None:
    if value is None:
        return None
    return normalize_eos_token_ids(value, source=source)


def select_effective_eos_token_ids(
    generation_config_eos: Any,
    model_config_eos: Any,
    tokenizer_eos: Any,
) -> tuple[list[int], str]:
    """Apply the same EOS precedence used by a loaded generation model."""

    candidates = (
        ("model_generation_config", generation_config_eos),
        ("model_config", model_config_eos),
        ("tokenizer", tokenizer_eos),
    )
    for source, value in candidates:
        if value is not None:
            return normalize_eos_token_ids(value, source=source), source
    raise ValueError("No EOS token ID is declared by the model or tokenizer")


def resolve_decoding_policy(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve EOS/padding metadata before fingerprinting or loading weights."""

    from transformers import AutoConfig, AutoTokenizer, GenerationConfig, __version__

    model_config = AutoConfig.from_pretrained(
        args.base, local_files_only=args.offline
    )
    try:
        generation_config = GenerationConfig.from_pretrained(
            args.base, local_files_only=args.offline
        )
        generation_config_load_source = "pretrained_generation_config"
    except OSError:
        generation_config = GenerationConfig.from_model_config(model_config)
        generation_config_load_source = "derived_from_model_config"
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.base, local_files_only=args.offline
    )

    generation_eos = getattr(generation_config, "eos_token_id", None)
    model_eos = getattr(model_config, "eos_token_id", None)
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    effective_eos, effective_source = select_effective_eos_token_ids(
        generation_eos, model_eos, tokenizer_eos
    )
    tokenizer_eos_ids = _optional_eos_token_ids(
        tokenizer_eos, source="tokenizer.eos_token_id"
    )
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        if not isinstance(tokenizer_eos, int) or isinstance(tokenizer_eos, bool):
            raise ValueError(
                "Tokenizer has no scalar pad_token_id or scalar eos_token_id fallback"
            )
        pad_token_id = tokenizer_eos
        pad_source = "tokenizer_eos_fallback"
    else:
        pad_source = "tokenizer_pad_token_id"
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int) or pad_token_id < 0:
        raise ValueError(f"Tokenizer has an invalid pad token ID: {pad_token_id!r}")

    return {
        "stop_policy_version": STOP_POLICY_VERSION,
        "effective_eos_token_ids": effective_eos,
        "effective_eos_token_id_source": effective_source,
        "generation_config_load_source": generation_config_load_source,
        "generation_config_eos_token_ids": _optional_eos_token_ids(
            generation_eos, source="generation_config.eos_token_id"
        ),
        "model_config_eos_token_ids": _optional_eos_token_ids(
            model_eos, source="model_config.eos_token_id"
        ),
        "tokenizer_eos_token_ids": tokenizer_eos_ids,
        "pad_token_id": pad_token_id,
        "pad_token_id_source": pad_source,
        "stop_rule": "stop after emitting the first token in effective_eos_token_ids",
        "generated_token_count_includes_terminal_eos": True,
        "transformers_version": __version__,
    }


def assert_runtime_decoding_policy(
    model: Any, tokenizer: Any, decoding_policy: dict[str, Any]
) -> None:
    """Fail closed if loaded runtime objects differ from preflight provenance."""

    generation_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    model_eos = getattr(getattr(model, "config", None), "eos_token_id", None)
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    actual_eos, actual_source = select_effective_eos_token_ids(
        generation_eos, model_eos, tokenizer_eos
    )
    expected_eos = decoding_policy["effective_eos_token_ids"]
    if actual_eos != expected_eos:
        raise RuntimeError(
            "Runtime EOS configuration differs from the fingerprinted preflight: "
            f"expected={expected_eos}, actual={actual_eos}, source={actual_source}"
        )
    actual_tokenizer_eos = _optional_eos_token_ids(
        tokenizer_eos, source="runtime tokenizer.eos_token_id"
    )
    if actual_tokenizer_eos != decoding_policy["tokenizer_eos_token_ids"]:
        raise RuntimeError("Runtime tokenizer EOS differs from preflight provenance")
    if getattr(tokenizer, "pad_token_id", None) != decoding_policy["pad_token_id"]:
        raise RuntimeError("Runtime tokenizer padding differs from preflight provenance")


def load_model(args: argparse.Namespace):
    """Load the deployment model and every stashed assistant/reference."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = _torch_dtype(torch, args.dtype)
    common = {
        "torch_dtype": dtype,
        "attn_implementation": args.attn_implementation,
        "local_files_only": args.offline,
    }

    sys.path.insert(0, str(Path(args.ou_repo) / "src"))
    from model.recap import RECAPForCausalLM

    model = RECAPForCausalLM.from_pretrained(
        args.base,
        a1_path=args.a1,
        a2_path=args.a2,
        weight_a1=args.weight_a1,
        weight_a2=args.weight_a2,
        top_logit_filter=args.top_filter,
        top_logit_filter_a1=args.top_filter_a1,
        top_logit_filter_a2=args.top_filter_a2,
        composition_mode=args.composition_mode,
        reference_a1_path=args.reference_a1,
        reference_a2_path=args.reference_a2,
        **common,
    )

    model.to(args.device)
    # These modules are deliberately stashed with object.__setattr__, so a
    # parent .to(device) does not visit them.  Move each unique object once.
    seen: set[int] = set()
    for attr in (
        "_recap_a1",
        "_recap_a2",
        "_recap_reference_a1",
        "_recap_reference_a2",
    ):
        child = getattr(model, attr, None)
        if child is not None and id(child) not in seen:
            child.to(args.device)
            child.eval()
            seen.add(id(child))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.base, local_files_only=args.offline
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return model, tokenizer


def batched(values: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def trim_generated_token_ids(
    padded_token_ids: list[int],
    eos_token_ids: int | Sequence[int],
    max_new_tokens: int,
) -> tuple[list[int], bool, bool]:
    """Remove batch padding after the first declared EOS and derive flags."""

    normalized_eos = normalize_eos_token_ids(
        eos_token_ids, source="trim eos_token_ids"
    )
    eos_set = set(normalized_eos)
    eos_position = next(
        (position for position, token_id in enumerate(padded_token_ids) if token_id in eos_set),
        None,
    )
    if eos_position is not None:
        token_ids = padded_token_ids[: eos_position + 1]
        terminated_by_eos = True
    else:
        token_ids = padded_token_ids
        terminated_by_eos = False
    hit_max_new_tokens = not terminated_by_eos and len(token_ids) == max_new_tokens
    return token_ids, terminated_by_eos, hit_max_new_tokens


def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
    eos_token_ids: Sequence[int],
    pad_token_id: int,
) -> list[dict[str, Any]]:
    import torch

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
    ).to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=False,
            pad_token_id=pad_token_id,
            eos_token_id=list(eos_token_ids),
        )
    prompt_width = encoded["input_ids"].shape[1]
    results = []
    for sequence in output_ids:
        padded_generated_ids = sequence[prompt_width:]
        padded_token_ids = padded_generated_ids.detach().cpu().tolist()
        token_ids, terminated_by_eos, hit_max_new_tokens = trim_generated_token_ids(
            padded_token_ids, eos_token_ids, max_new_tokens
        )
        generated_ids = padded_generated_ids[: len(token_ids)]
        terminal_eos_token_id = token_ids[-1] if terminated_by_eos else None
        if terminated_by_eos:
            termination_reason = "eos"
        elif hit_max_new_tokens:
            termination_reason = "max_new_tokens"
        else:
            termination_reason = "other"
        results.append(
            {
                "generation": tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                ).strip(),
                "generated_token_count": len(token_ids),
                "terminated_by_eos": terminated_by_eos,
                "hit_max_new_tokens": hit_max_new_tokens,
                "terminal_eos_token_id": terminal_eos_token_id,
                "termination_reason": termination_reason,
                "generated_token_ids_sha256": compact_json_sha256(token_ids),
            }
        )
    return results


def termination_summary(
    rows: list[dict[str, Any]], eos_token_ids: Sequence[int]
) -> dict[str, Any]:
    return {
        "eos": sum(row.get("termination_reason") == "eos" for row in rows),
        "max_new_tokens": sum(
            row.get("termination_reason") == "max_new_tokens" for row in rows
        ),
        "other": sum(row.get("termination_reason") == "other" for row in rows),
        "by_terminal_eos_token_id": {
            str(token_id): sum(
                row.get("terminal_eos_token_id") == token_id for row in rows
            )
            for token_id in eos_token_ids
        },
    }


def validate_termination_rows(
    rows: list[dict[str, Any]], decoding: dict[str, Any]
) -> None:
    eos_token_ids = normalize_eos_token_ids(
        decoding.get("effective_eos_token_ids"), source="spec.decoding effective EOS"
    )
    max_new_tokens = decoding.get("max_new_tokens")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
        raise RuntimeError("Invalid max_new_tokens in generation spec")
    for row in rows:
        idx = row.get("idx")
        count = row.get("generated_token_count")
        terminated = row.get("terminated_by_eos")
        hit_limit = row.get("hit_max_new_tokens")
        terminal = row.get("terminal_eos_token_id")
        reason = row.get("termination_reason")
        if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= max_new_tokens:
            raise RuntimeError(f"Invalid generated token count for idx={idx}")
        if not isinstance(terminated, bool) or not isinstance(hit_limit, bool):
            raise RuntimeError(f"Invalid termination flags for idx={idx}")
        if reason == "eos":
            valid = terminated and not hit_limit and terminal in eos_token_ids and count >= 1
        elif reason == "max_new_tokens":
            valid = not terminated and hit_limit and terminal is None and count == max_new_tokens
        else:
            valid = False
        if not valid:
            raise RuntimeError(f"Inconsistent termination metadata for idx={idx}")


def load_partial(path: Path, fingerprint: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("config_fingerprint") != fingerprint:
        raise RuntimeError(
            f"Refusing to resume incompatible partial output: {path}"
        )
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"Malformed partial output: {path}")
    return rows


def validate_complete_output(
    path: Path,
    fingerprint: str,
    expected_indices: list[int],
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    coverage = payload.get("coverage") or {}
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"Existing output has the wrong schema version: {path}")
    if payload.get("config_fingerprint") != fingerprint:
        raise RuntimeError(f"Existing output has an incompatible config: {path}")
    if not isinstance(rows, list) or [row.get("idx") for row in rows] != expected_indices:
        raise RuntimeError(f"Existing output has invalid sampled-row coverage: {path}")
    if coverage != {
        "expected": len(expected_indices),
        "actual": len(expected_indices),
        "complete": True,
    }:
        raise RuntimeError(f"Existing output is not marked complete: {path}")
    if payload.get("rows_sha256") != compact_json_sha256(rows):
        raise RuntimeError(f"Existing output row hash does not verify: {path}")
    decoding = ((payload.get("spec") or {}).get("decoding") or {})
    if decoding.get("stop_policy_version") != STOP_POLICY_VERSION:
        raise RuntimeError(f"Existing output has the wrong stop policy: {path}")
    validate_termination_rows(rows, decoding)
    expected_summary = termination_summary(
        rows, decoding["effective_eos_token_ids"]
    )
    if payload.get("termination_summary") != expected_summary:
        raise RuntimeError(f"Existing output termination summary does not verify: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=["recap"])
    parser.add_argument("--method", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--a1")
    parser.add_argument("--a2")
    parser.add_argument("--weight-a1", type=float, default=-0.8)
    parser.add_argument("--weight-a2", type=float, default=0.5)
    parser.add_argument("--top-filter", type=float, default=0.01)
    parser.add_argument("--top-filter-a1", type=float)
    parser.add_argument("--top-filter-a2", type=float)
    parser.add_argument(
        "--composition-mode", choices=["reference_delta"], default="reference_delta"
    )
    parser.add_argument("--reference-a1")
    parser.add_argument("--reference-a2")
    parser.add_argument("--ou-repo", default="open-unlearning")
    parser.add_argument(
        "--standard-report",
        help="frozen RECAP standard-evaluation report",
    )
    parser.add_argument("--require-recap-b0", action="store_true")
    parser.add_argument("--dataset", default="locuslab/TOFU")
    parser.add_argument("--dataset-config", default="forget10_perturbed")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--expected-rows", type=int, default=400)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--expected-eos-token-ids", type=parse_eos_token_id_csv)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    args.standard_report_provenance = validate_standard_report(args)
    out_path = Path(args.out)

    from datasets import load_dataset
    import torch

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = load_dataset(args.dataset, args.dataset_config, split=args.dataset_split)
    if len(dataset) != args.expected_rows:
        raise RuntimeError(
            f"Expected {args.expected_rows} dataset rows, found {len(dataset)}"
        )
    indices = sample_indices(len(dataset), args.n, args.seed)
    sampled_qa = [
        {
            "idx": index,
            "question": str(dataset[index][args.question_field]),
            "ground_truth": str(dataset[index][args.answer_field]),
        }
        for index in indices
    ]
    sampled_qa_sha256 = compact_json_sha256(sampled_qa)
    decoding_policy = resolve_decoding_policy(args)
    if (
        args.expected_eos_token_ids is not None
        and decoding_policy["effective_eos_token_ids"] != args.expected_eos_token_ids
    ):
        raise RuntimeError(
            "Resolved EOS list differs from --expected-eos-token-ids: "
            f"expected={args.expected_eos_token_ids}, "
            f"actual={decoding_policy['effective_eos_token_ids']}"
        )
    if (
        args.require_recap_b0
        and decoding_policy["effective_eos_token_ids"] != RECAP_B0_EOS_TOKEN_IDS
    ):
        raise RuntimeError(
            "Frozen RECAP B0 requires the Llama-3 multi-EOS list "
            f"{RECAP_B0_EOS_TOKEN_IDS}, found "
            f"{decoding_policy['effective_eos_token_ids']}"
        )
    spec = semantic_spec(args, indices, sampled_qa_sha256, decoding_policy)
    fingerprint = compact_json_sha256(spec)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "config_fingerprint": fingerprint,
                    "indices_sha256": spec["sampling"]["indices_sha256"],
                    "sampled_qa_sha256": sampled_qa_sha256,
                    "frozen_standard_report": spec["frozen_standard_report"],
                    "decoding": spec["decoding"],
                    "model": spec["model"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return
    if out_path.exists():
        validate_complete_output(out_path, fingerprint, indices)
        print(f"[generate] verified complete output: {out_path}", flush=True)
        return
    partial_path = out_path.with_suffix(".partial.json")
    rows = load_partial(partial_path, fingerprint)
    expected_prefix = indices[: len(rows)]
    if [row.get("idx") for row in rows] != expected_prefix:
        raise RuntimeError("Partial rows are not the expected sampled-index prefix")
    if rows:
        validate_termination_rows(rows, spec["decoding"])

    print(
        f"[generate] method={args.method} sample={len(indices)}/{len(dataset)} "
        f"indices_sha256={spec['sampling']['indices_sha256']} resumed={len(rows)}",
        flush=True,
    )
    pending = indices[len(rows) :]
    model = tokenizer = None
    if pending:
        model, tokenizer = load_model(args)
        assert_runtime_decoding_policy(model, tokenizer, decoding_policy)
    for index_batch in batched(pending, args.batch_size):
        prompts = [
            build_prompt(str(dataset[index][args.question_field])) for index in index_batch
        ]
        generated = generate_batch(
            model,
            tokenizer,
            prompts,
            args.max_new_tokens,
            decoding_policy["effective_eos_token_ids"],
            decoding_policy["pad_token_id"],
        )
        for index, prompt, result in zip(index_batch, prompts, generated):
            rows.append(
                {
                    "idx": index,
                    "question": str(dataset[index][args.question_field]),
                    "ground_truth": str(dataset[index][args.answer_field]),
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    **result,
                }
            )
        validate_termination_rows(rows, spec["decoding"])
        atomic_write_json(
            partial_path,
            {
                "schema_version": SCHEMA_VERSION,
                "config_fingerprint": fingerprint,
                "spec": spec,
                "termination_summary": termination_summary(
                    rows, decoding_policy["effective_eos_token_ids"]
                ),
                "rows": rows,
            },
        )
        print(f"[generate] {len(rows)}/{len(indices)}", flush=True)

    if len(rows) != len(indices) or [row["idx"] for row in rows] != indices:
        raise RuntimeError("Generation coverage invariant failed")
    validate_termination_rows(rows, spec["decoding"])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": fingerprint,
        "git_commit": current_git_commit(Path(args.ou_repo).resolve().parent),
        "spec": spec,
        "coverage": {
            "expected": len(indices),
            "actual": len(rows),
            "complete": True,
        },
        "termination_summary": termination_summary(
            rows, decoding_policy["effective_eos_token_ids"]
        ),
        "rows_sha256": compact_json_sha256(rows),
        "rows": rows,
    }
    atomic_write_json(out_path, payload)
    print(f"[generate] complete: {out_path}", flush=True)


if __name__ == "__main__":
    main()
