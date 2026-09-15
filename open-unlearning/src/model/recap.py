"""RECAP inference wrapper for TOFU.

RECAP composes two independently trained assistants through their changes
relative to the frozen initialization used for each role::

    logits = base
           + w1 * mask_1 * (assistant_1 - reference_1)
           + w2 * mask_2 * (assistant_2 - reference_2)

This paper-facing implementation intentionally supports only reference-delta
composition.  It does not include calibration, alignment, gates, or routers.
KV caching is disabled because the base, assistants, and references can have
different transformer depths.
"""

import logging
import math
import os
from typing import Optional

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger("model.recap")

_AUTO_REFERENCE_VALUES = {None, "", "auto", "null"}


def _validate_finite_float(name: str, value: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number (got {value!r}).") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number (got {value!r}).")
    return value


def _validate_filter(name: str, value: float) -> float:
    value = _validate_finite_float(name, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1] (got {value}).")
    return value


def _relative_top_keep_mask(
    base_logits: torch.Tensor,
    relative_top: float,
) -> torch.Tensor:
    """Return the base-vocabulary mask used for an assistant contribution.

    The threshold matches the established ULD filter: it retains at least
    ``relative_top * vocabulary_size`` candidates and also admits candidates
    whose probability is sufficiently close to the base model's top token.
    A value of zero disables filtering.
    """

    relative_top = _validate_filter("relative_top", relative_top)
    if base_logits.shape[-1] < 1:
        raise ValueError("base_logits must have a non-empty vocabulary axis.")
    if relative_top == 0.0:
        return torch.ones_like(base_logits, dtype=torch.bool)

    min_tokens_to_keep = max(int(relative_top * base_logits.shape[-1]), 1)
    normalized = base_logits.log_softmax(dim=-1)
    kth_score = torch.topk(
        normalized,
        min_tokens_to_keep,
        dim=-1,
    ).values[..., -1]
    relative_score = normalized.amax(dim=-1) + math.log(relative_top)
    threshold = torch.minimum(kth_score, relative_score).unsqueeze(-1)
    return normalized >= threshold


def compose_reference_delta(
    base_logits: torch.Tensor,
    a1_logits: torch.Tensor,
    reference_a1_logits: torch.Tensor,
    a2_logits: torch.Tensor,
    reference_a2_logits: torch.Tensor,
    *,
    weight_a1: float,
    weight_a2: float,
    top_logit_filter_a1: float,
    top_logit_filter_a2: float,
) -> torch.Tensor:
    """Compose RECAP logits without mutating any input tensor."""

    named_logits = {
        "a1_logits": a1_logits,
        "reference_a1_logits": reference_a1_logits,
        "a2_logits": a2_logits,
        "reference_a2_logits": reference_a2_logits,
    }
    for name, logits in named_logits.items():
        if logits.shape != base_logits.shape:
            raise ValueError(
                f"{name} shape {tuple(logits.shape)} does not match "
                f"base_logits shape {tuple(base_logits.shape)}."
            )

    weight_a1 = _validate_finite_float("weight_a1", weight_a1)
    weight_a2 = _validate_finite_float("weight_a2", weight_a2)
    top_logit_filter_a1 = _validate_filter(
        "top_logit_filter_a1", top_logit_filter_a1
    )
    top_logit_filter_a2 = _validate_filter(
        "top_logit_filter_a2", top_logit_filter_a2
    )

    keep_a1 = _relative_top_keep_mask(base_logits, top_logit_filter_a1)
    keep_a2 = _relative_top_keep_mask(base_logits, top_logit_filter_a2)
    delta_a1 = torch.where(
        keep_a1,
        a1_logits - reference_a1_logits,
        torch.zeros((), device=base_logits.device, dtype=base_logits.dtype),
    )
    delta_a2 = torch.where(
        keep_a2,
        a2_logits - reference_a2_logits,
        torch.zeros((), device=base_logits.device, dtype=base_logits.dtype),
    )
    return base_logits + weight_a1 * delta_a1 + weight_a2 * delta_a2


def _model_load_kwargs(
    torch_dtype: Optional[torch.dtype],
    attn_implementation: Optional[str],
) -> dict:
    kwargs = {}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    return kwargs


def _sibling_fullmodel(assistant_path: str) -> str:
    return os.path.normpath(os.path.join(assistant_path, "..", "fullmodel"))


def _load_assistant(
    assistant_path: str,
    torch_dtype: Optional[torch.dtype],
    attn_implementation: Optional[str],
    initialization_path: Optional[str] = None,
) -> torch.nn.Module:
    """Load a merged assistant or merge LoRA into its frozen initialization."""

    load_kwargs = _model_load_kwargs(torch_dtype, attn_implementation)
    adapter_config = os.path.join(assistant_path, "adapter_config.json")
    if os.path.isfile(adapter_config):
        fullmodel_path = initialization_path or _sibling_fullmodel(assistant_path)
        if not os.path.isdir(fullmodel_path):
            raise FileNotFoundError(
                f"Found LoRA adapter at {assistant_path!r}, but its frozen "
                f"assistant initialization is missing: {fullmodel_path!r}."
            )

        from peft import PeftModel  # PEFT is optional for merged checkpoints.

        logger.info("RECAP: loading assistant initialization from %s", fullmodel_path)
        initialization = AutoModelForCausalLM.from_pretrained(
            fullmodel_path,
            **load_kwargs,
        )
        logger.info("RECAP: merging LoRA adapter from %s", assistant_path)
        return PeftModel.from_pretrained(
            initialization,
            assistant_path,
        ).merge_and_unload()

    logger.info("RECAP: loading merged assistant from %s", assistant_path)
    return AutoModelForCausalLM.from_pretrained(assistant_path, **load_kwargs)


def _resolve_reference_path(
    role: str,
    assistant_path: str,
    reference_path: Optional[str],
) -> str:
    """Resolve a role's frozen reference, failing closed when none exists."""

    if reference_path not in _AUTO_REFERENCE_VALUES:
        if reference_path == "???":
            raise ValueError(f"reference_{role}_path must be set or use auto/null.")
        return str(reference_path)

    inferred = _sibling_fullmodel(assistant_path)
    if not os.path.isdir(inferred):
        raise FileNotFoundError(
            f"RECAP requires an independent frozen reference for {role}. "
            f"No explicit reference_{role}_path was set and the inferred "
            f"sibling directory does not exist: {inferred!r}."
        )
    return inferred


def _freeze_for_inference(module: torch.nn.Module) -> torch.nn.Module:
    module.requires_grad_(False)
    module.eval()
    return module


def _validate_reference_pair(
    role: str,
    base: torch.nn.Module,
    assistant: torch.nn.Module,
    reference: torch.nn.Module,
) -> None:
    """Reject incompatible logits or assistant/reference architectures."""

    for field in ("vocab_size", "hidden_size"):
        base_value = getattr(base.config, field, None)
        assistant_value = getattr(assistant.config, field, None)
        reference_value = getattr(reference.config, field, None)
        if base_value != assistant_value or assistant_value != reference_value:
            raise ValueError(
                f"RECAP {role} config mismatch for {field}: "
                f"base={base_value}, assistant={assistant_value}, "
                f"reference={reference_value}."
            )

    assistant_layers = getattr(assistant.config, "num_hidden_layers", None)
    reference_layers = getattr(reference.config, "num_hidden_layers", None)
    if assistant_layers != reference_layers:
        raise ValueError(
            f"RECAP {role} assistant/reference mismatch for num_hidden_layers: "
            f"assistant={assistant_layers}, reference={reference_layers}."
        )


class RECAPForCausalLM(LlamaForCausalLM):
    """Llama causal LM with the paper's two-role reference-delta inference."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        a1_path: Optional[str] = None,
        a2_path: Optional[str] = None,
        reference_a1_path: Optional[str] = None,
        reference_a2_path: Optional[str] = None,
        weight_a1: float = -2.1,
        weight_a2: float = 2.2,
        top_logit_filter: float = 0.00017,
        top_logit_filter_a1: Optional[float] = None,
        top_logit_filter_a2: Optional[float] = None,
        composition_mode: str = "reference_delta",
        **kwargs,
    ):
        for name, path in (("a1_path", a1_path), ("a2_path", a2_path)):
            if path is None or str(path).strip() in {"", "???", "null", "auto"}:
                raise ValueError(f"RECAPForCausalLM requires `{name}`.")

        if composition_mode != "reference_delta":
            raise ValueError(
                "RECAPForCausalLM only supports fail-closed "
                f"reference_delta composition (got {composition_mode!r})."
            )

        weight_a1 = _validate_finite_float("weight_a1", weight_a1)
        weight_a2 = _validate_finite_float("weight_a2", weight_a2)
        top_logit_filter = _validate_filter("top_logit_filter", top_logit_filter)
        filter_a1 = _validate_filter(
            "top_logit_filter_a1",
            top_logit_filter if top_logit_filter_a1 is None else top_logit_filter_a1,
        )
        filter_a2 = _validate_filter(
            "top_logit_filter_a2",
            top_logit_filter if top_logit_filter_a2 is None else top_logit_filter_a2,
        )

        # Resolve references before loading the 8B base so configuration errors
        # fail cheaply.  Each role is then loaded into a distinct frozen module.
        a1_path = str(a1_path)
        a2_path = str(a2_path)
        reference_a1_path = _resolve_reference_path(
            "a1", a1_path, reference_a1_path
        )
        reference_a2_path = _resolve_reference_path(
            "a2", a2_path, reference_a2_path
        )
        if os.path.realpath(reference_a1_path) == os.path.realpath(reference_a2_path):
            raise ValueError(
                "RECAP-B0 requires distinct role-specific A1 and A2 references."
            )

        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        device = next(model.parameters()).device
        torch_dtype = kwargs.get("torch_dtype")
        attn_implementation = kwargs.get("attn_implementation")
        load_kwargs = _model_load_kwargs(torch_dtype, attn_implementation)

        a1 = _load_assistant(
            a1_path,
            torch_dtype,
            attn_implementation,
            initialization_path=reference_a1_path,
        )
        a2 = _load_assistant(
            a2_path,
            torch_dtype,
            attn_implementation,
            initialization_path=reference_a2_path,
        )
        # Never alias references across roles, even when callers explicitly use
        # the same path: provenance and freezing remain role-local.
        reference_a1 = AutoModelForCausalLM.from_pretrained(
            reference_a1_path,
            **load_kwargs,
        )
        reference_a2 = AutoModelForCausalLM.from_pretrained(
            reference_a2_path,
            **load_kwargs,
        )

        for module in (a1, a2, reference_a1, reference_a2):
            module.to(device)
            _freeze_for_inference(module)

        _validate_reference_pair("a1", model, a1, reference_a1)
        _validate_reference_pair("a2", model, a2, reference_a2)

        # Keep auxiliary modules out of the wrapper's module tree.  This avoids
        # model.save_pretrained() treating them as part of the base checkpoint.
        object.__setattr__(model, "_recap_a1", a1)
        object.__setattr__(model, "_recap_a2", a2)
        object.__setattr__(model, "_recap_reference_a1", reference_a1)
        object.__setattr__(model, "_recap_reference_a2", reference_a2)
        model._recap_weight_a1 = weight_a1
        model._recap_weight_a2 = weight_a2
        model._recap_top_logit_filter_a1 = filter_a1
        model._recap_top_logit_filter_a2 = filter_a2
        model._recap_reference_a1_path = reference_a1_path
        model._recap_reference_a2_path = reference_a2_path

        model.config.use_cache = False
        model.generation_config.use_cache = False
        logger.info(
            "RECAP ready: base=%s a1=%s a2=%s reference_a1=%s "
            "reference_a2=%s weight_a1=%s weight_a2=%s filter_a1=%s filter_a2=%s",
            pretrained_model_name_or_path,
            a1_path,
            a2_path,
            reference_a1_path,
            reference_a2_path,
            weight_a1,
            weight_a2,
            filter_a1,
            filter_a2,
        )
        return model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
        **kwargs,
    ):
        common = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": None,
            "inputs_embeds": inputs_embeds,
            "labels": None,
            "use_cache": False,
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict": True,
        }

        base_output = super().forward(**common)
        with torch.no_grad():
            a1_logits = self._recap_a1(**common).logits
            a2_logits = self._recap_a2(**common).logits
            reference_a1_logits = self._recap_reference_a1(**common).logits
            reference_a2_logits = self._recap_reference_a2(**common).logits

        base_logits = base_output.logits
        target = {"device": base_logits.device, "dtype": base_logits.dtype}
        logits = compose_reference_delta(
            base_logits,
            a1_logits.to(**target),
            reference_a1_logits.to(**target),
            a2_logits.to(**target),
            reference_a2_logits.to(**target),
            weight_a1=self._recap_weight_a1,
            weight_a2=self._recap_weight_a2,
            top_logit_filter_a1=self._recap_top_logit_filter_a1,
            top_logit_filter_a2=self._recap_top_logit_filter_a2,
        )

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            loss = CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        kwargs.pop("past_key_values", None)
        kwargs["use_cache"] = False
        prepared = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=None,
            **kwargs,
        )
        prepared["input_ids"] = input_ids
        if kwargs.get("attention_mask") is not None:
            prepared["attention_mask"] = kwargs["attention_mask"]
        prepared["past_key_values"] = None
        prepared["use_cache"] = False
        return prepared
