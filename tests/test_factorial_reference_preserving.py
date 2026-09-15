import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ULD" / "uld" / "model" / "forget_losses.py"
spec = importlib.util.spec_from_file_location("reference_losses", MODULE_PATH)
losses = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(losses)


def import_model_utils():
    peft_module = types.ModuleType("peft")
    peft_module.LoraConfig = object
    peft_module.get_peft_model = lambda model, config: model

    uld_module = types.ModuleType("uld")
    uld_module.__path__ = []
    model_module = types.ModuleType("uld.model")
    model_module.__path__ = []
    root_utils = types.ModuleType("uld.utils")
    root_utils.NameTimer = object
    peft_util = types.ModuleType("uld.model.peft_util")
    peft_util.find_all_linear_names = lambda model: []

    module_path = ROOT / "ULD" / "uld" / "model" / "utils.py"
    module_spec = importlib.util.spec_from_file_location(
        "uld.model.utils", module_path
    )
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader is not None
    with mock.patch.dict(
        sys.modules,
        {
            "peft": peft_module,
            "uld": uld_module,
            "uld.model": model_module,
            "uld.utils": root_utils,
            "uld.model.peft_util": peft_util,
        },
    ):
        module_spec.loader.exec_module(module)
    return module


class TinyModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(logits.clone())

    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch = input_ids.shape[0]
        return type("Output", (), {"logits": self.logits.expand(batch, -1, -1)})


class FactorialReferencePreservingTest(unittest.TestCase):
    def batch(self):
        return {
            "input_ids": torch.tensor([[1, 2, 3, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
            "labels": torch.tensor([[-100, -100, 3, -100]]),
        }

    def test_equal_reference_has_zero_kl(self):
        logits = torch.randn(1, 4, 5)
        loss = losses.AnswerMaskedKLLossFunc(
            TinyModel(logits), oracle_model=TinyModel(logits), **self.batch()
        )
        self.assertAlmostEqual(float(loss.detach()), 0.0, places=6)

    def test_only_answer_positions_affect_kl(self):
        reference = torch.zeros(1, 4, 5)
        changed = reference.clone()
        changed[:, 0, 0] = 20.0
        changed[:, 2:, 1] = 20.0
        loss = losses.AnswerMaskedKLLossFunc(
            TinyModel(changed), oracle_model=TinyModel(reference), **self.batch()
        )
        self.assertAlmostEqual(float(loss.detach()), 0.0, places=6)

        changed[:, 1, 2] = 20.0
        model = TinyModel(changed)
        loss = losses.AnswerMaskedKLLossFunc(
            model, oracle_model=TinyModel(reference), **self.batch()
        )
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertGreater(float(model.logits.grad[:, 1].abs().sum()), 0.0)

    def test_config_uses_reference_oracle(self):
        config = {
            "forget_loss": "GradDescentLossFunc",
            "retain_loss": "AnswerMaskedKLLossFunc",
            "retain_weight": 0.3,
        }
        objective = losses.create_unlearn_loss(config)
        self.assertIs(objective.retain_loss_func, losses.AnswerMaskedKLLossFunc)
        self.assertTrue(losses.loss_requries_oracle(config))

        loss_config = (
            ROOT / "ULD/configs/unlearn_loss/factorial_reference_preserving.yaml"
        ).read_text(encoding="utf-8")
        trainer = (ROOT / "ULD/scripts/hf_forget_train.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("oracle_source: assistant_initialization", loss_config)
        self.assertIn("os.path.join(baseoutdir, 'fullmodel')", trainer)

    def test_exact_steps_and_optimizer_are_configurable(self):
        tune_config = (ROOT / "ULD/configs/tune_config.yaml").read_text(
            encoding="utf-8"
        )
        trainer = (ROOT / "ULD/scripts/hf_forget_train.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("max_steps: null", tune_config)
        self.assertIn("optim: adamw_torch", tune_config)
        self.assertIn("report_to: none", tune_config)
        self.assertIn('trainer_config.get("max_steps")', trainer)
        self.assertIn("trainer_config.get('optim'", trainer)

    def test_training_sequence_limit_matches_the_executed_b0_path(self):
        data_config = (ROOT / "ULD/configs/data/tofu_chat3.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("max_len: 256", data_config)
        self.assertNotIn("max_length: 350", data_config)

    def test_attention_backend_can_be_frozen_to_sdpa(self):
        model_utils = import_model_utils()
        with mock.patch.dict(
            os.environ, {"ULD_ATTN_IMPLEMENTATION": "sdpa"}, clear=False
        ):
            self.assertEqual(
                model_utils._attn_kwargs(), {"attn_implementation": "sdpa"}
            )

    def test_unset_attention_backend_preserves_auto_detection(self):
        model_utils = import_model_utils()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ULD_ATTN_IMPLEMENTATION", None)
            with mock.patch.object(
                model_utils, "_flash_attn_available", return_value=True
            ):
                self.assertEqual(
                    model_utils._attn_kwargs(), {"use_flash_attention_2": True}
                )
            with mock.patch.object(
                model_utils, "_flash_attn_available", return_value=False
            ):
                self.assertEqual(model_utils._attn_kwargs(), {})

    def test_unknown_attention_backend_fails_closed(self):
        model_utils = import_model_utils()
        with mock.patch.dict(
            os.environ, {"ULD_ATTN_IMPLEMENTATION": "automatic"}, clear=False
        ):
            with self.assertRaisesRegex(ValueError, "ULD_ATTN_IMPLEMENTATION"):
                model_utils._attn_kwargs()


if __name__ == "__main__":
    unittest.main()
