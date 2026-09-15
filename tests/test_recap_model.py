import importlib.util
from pathlib import Path

import pytest
import torch


RECAP_PATH = (
    Path(__file__).resolve().parents[1]
    / "open-unlearning"
    / "src"
    / "model"
    / "recap.py"
)
SPEC = importlib.util.spec_from_file_location("recap_under_test", RECAP_PATH)
recap = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(recap)


def test_reference_delta_formula_without_filtering():
    base = torch.tensor([[[1.0, 2.0, 3.0]]])
    a1 = torch.tensor([[[4.0, 6.0, 8.0]]])
    ref1 = torch.tensor([[[1.0, 2.0, 3.0]]])
    a2 = torch.tensor([[[7.0, 8.0, 9.0]]])
    ref2 = torch.tensor([[[6.0, 6.0, 6.0]]])

    actual = recap.compose_reference_delta(
        base,
        a1,
        ref1,
        a2,
        ref2,
        weight_a1=-2.0,
        weight_a2=0.5,
        top_logit_filter_a1=0.0,
        top_logit_filter_a2=0.0,
    )
    expected = base - 2.0 * (a1 - ref1) + 0.5 * (a2 - ref2)
    torch.testing.assert_close(actual, expected)


def test_role_specific_filters_mask_the_reference_deltas():
    base = torch.tensor([[[3.0, 2.0, 0.0, -2.0]]])
    zeros = torch.zeros_like(base)
    a1 = torch.ones_like(base)
    a2 = torch.full_like(base, 2.0)

    actual = recap.compose_reference_delta(
        base,
        a1,
        zeros,
        a2,
        zeros,
        weight_a1=1.0,
        weight_a2=1.0,
        top_logit_filter_a1=0.5,
        top_logit_filter_a2=0.75,
    )
    # 0.5 retains the top two base candidates; 0.75 retains the top three.
    expected = base + torch.tensor([[[3.0, 3.0, 2.0, 0.0]]])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("bad_filter", [-0.01, 1.01, float("nan")])
def test_filter_parameter_validation(bad_filter):
    with pytest.raises(ValueError, match="must be"):
        recap._relative_top_keep_mask(torch.zeros(1, 1, 4), bad_filter)


def test_composition_rejects_mismatched_shapes():
    base = torch.zeros(1, 1, 4)
    with pytest.raises(ValueError, match="does not match"):
        recap.compose_reference_delta(
            base,
            torch.zeros(1, 1, 3),
            base,
            base,
            base,
            weight_a1=1.0,
            weight_a2=1.0,
            top_logit_filter_a1=0.0,
            top_logit_filter_a2=0.0,
        )


def test_model_rejects_missing_assistants_before_loading_weights():
    with pytest.raises(ValueError, match="a1_path"):
        recap.RECAPForCausalLM.from_pretrained("unused", a2_path="unused")


def test_model_rejects_raw_composition_before_loading_weights(tmp_path):
    a1 = tmp_path / "a1" / "checkpoint"
    a2 = tmp_path / "a2" / "checkpoint"
    (a1.parent / "fullmodel").mkdir(parents=True)
    (a2.parent / "fullmodel").mkdir(parents=True)
    with pytest.raises(ValueError, match="only supports.*reference_delta"):
        recap.RECAPForCausalLM.from_pretrained(
            "unused",
            a1_path=str(a1),
            a2_path=str(a2),
            composition_mode="raw",
        )


def test_auto_reference_fails_closed_when_sibling_is_missing(tmp_path):
    assistant = tmp_path / "a1" / "checkpoint"
    assistant.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="independent frozen reference"):
        recap._resolve_reference_path("a1", str(assistant), None)


def test_model_rejects_a_shared_reference_before_loading_weights(tmp_path):
    reference = tmp_path / "shared-reference"
    reference.mkdir()
    with pytest.raises(ValueError, match="distinct role-specific"):
        recap.RECAPForCausalLM.from_pretrained(
            "unused",
            a1_path=str(tmp_path / "a1" / "checkpoint-168"),
            a2_path=str(tmp_path / "a2" / "checkpoint-144"),
            reference_a1_path=str(reference),
            reference_a2_path=str(reference),
        )


def test_lora_requires_sibling_fullmodel(tmp_path):
    assistant = tmp_path / "a1" / "checkpoint"
    assistant.mkdir(parents=True)
    (assistant / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="assistant initialization"):
        recap._load_assistant(str(assistant), None, None)


def test_lora_can_use_an_explicitly_named_initialization(tmp_path, monkeypatch):
    assistant = tmp_path / "a1" / "checkpoint"
    reference = tmp_path / "published" / "a1-reference"
    assistant.mkdir(parents=True)
    reference.mkdir(parents=True)
    (assistant / "adapter_config.json").write_text("{}", encoding="utf-8")

    loaded = []

    class FakeModel:
        pass

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **_kwargs):
            loaded.append(path)
            return FakeModel()

    class FakePeft:
        @classmethod
        def from_pretrained(cls, initialization, path):
            assert isinstance(initialization, FakeModel)
            loaded.append(path)
            return cls()

        @staticmethod
        def merge_and_unload():
            return "merged"

    monkeypatch.setattr(recap, "AutoModelForCausalLM", FakeAutoModel)
    monkeypatch.setitem(
        __import__("sys").modules,
        "peft",
        type("FakePeftModule", (), {"PeftModel": FakePeft}),
    )
    merged = recap._load_assistant(
        str(assistant), None, None, initialization_path=str(reference)
    )
    assert merged == "merged"
    assert loaded == [str(reference), str(assistant)]


def test_freeze_for_inference_disables_gradients():
    module = torch.nn.Linear(2, 2)
    module.train()
    recap._freeze_for_inference(module)
    assert not module.training
    assert all(not parameter.requires_grad for parameter in module.parameters())
