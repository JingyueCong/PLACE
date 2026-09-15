import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ULD" / "uld" / "data" / "recap.py"
spec = importlib.util.spec_from_file_location("recap_data", MODULE_PATH)
recap = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(recap)


def valid_unit(source_id="forget10-00000"):
    return {
        "source_id": source_id,
        "source_question": "Which award did Basil Hart win?",
        "source_answer": "The Riverdale Book Award.",
        "cells": {
            "C11": {
                "question": "Which award did Basil Hart win?",
                "answer": "The Riverdale Book Award.",
            },
            "C01": {
                "question": "Which award did Elian Mercer win?",
                "answer": "The Northbridge Literary Medal.",
            },
            "C10": {
                "question": "Where does Basil Hart live?",
                "answer": "Grayhaven.",
            },
            "C00": {
                "question": "Where does Elian Mercer live?",
                "answer": "Westhaven.",
            },
        },
    }


class FakeDataset(list):
    @property
    def column_names(self):
        return list(self[0]) if self else []

    def remove_columns(self, columns):
        return FakeDataset(
            [{key: value for key, value in row.items() if key not in columns} for row in self]
        )

    def rename_column(self, old, new):
        return FakeDataset(
            [
                {new if key == old else key: value for key, value in row.items()}
                for row in self
            ]
        )

    def select(self, indices):
        return FakeDataset([self[index] for index in indices])

    @classmethod
    def from_dict(cls, values):
        if not values:
            return cls()
        size = len(next(iter(values.values())))
        return cls([{key: column[index] for key, column in values.items()} for index in range(size)])

    @classmethod
    def from_list(cls, rows):
        return cls(rows)

    @classmethod
    def from_generator(cls, generator, gen_kwargs):
        return cls(generator(**gen_kwargs))


def import_tofu_with_fake_dependencies(load_dataset):
    datasets_module = types.ModuleType("datasets")
    datasets_module.Dataset = FakeDataset
    datasets_module.concatenate_datasets = lambda parts: FakeDataset(
        [row for part in parts for row in part]
    )
    datasets_module.load_dataset = load_dataset

    uld_module = types.ModuleType("uld")
    uld_module.__path__ = []
    data_module = types.ModuleType("uld.data")
    data_module.__path__ = []

    conv_module = types.ModuleType("uld.data.conv_util")
    conv_module.create_template = lambda config, tokenizer=None: object()
    datamodule_module = types.ModuleType("uld.data.datamodule")
    datamodule_module.TrainDataModule = object
    datamodule_module.TorchDataset = object

    recap_spec = importlib.util.spec_from_file_location(
        "uld.data.recap", MODULE_PATH
    )
    recap_module = importlib.util.module_from_spec(recap_spec)
    assert recap_spec.loader is not None

    tofu_path = ROOT / "ULD" / "uld" / "data" / "tofu.py"
    tofu_spec = importlib.util.spec_from_file_location("uld.data.tofu", tofu_path)
    tofu_module = importlib.util.module_from_spec(tofu_spec)
    assert tofu_spec.loader is not None

    modules = {
        "datasets": datasets_module,
        "uld": uld_module,
        "uld.data": data_module,
        "uld.data.conv_util": conv_module,
        "uld.data.datamodule": datamodule_module,
        "uld.data.recap": recap_module,
    }
    with mock.patch.dict(sys.modules, modules):
        recap_spec.loader.exec_module(recap_module)
        tofu_spec.loader.exec_module(tofu_module)
    return tofu_module


class RecapDataTest(unittest.TestCase):
    def test_valid_unit_and_dual_role_mapping(self):
        unit = valid_unit()
        self.assertEqual(recap.validate_recap_unit(unit), [])
        a1_fit, a1_control = recap.recap_dual_roles([unit], "recap_a1")
        a2_fit, a2_control = recap.recap_dual_roles([unit], "recap_a2")
        self.assertEqual(a1_fit, [unit["cells"]["C11"]])
        self.assertEqual(a1_control, [unit["cells"]["C01"]])
        self.assertEqual(a2_fit, [unit["cells"]["C10"]])
        self.assertEqual(a2_control, [unit["cells"]["C00"]])

    def test_schema_fails_closed(self):
        unit = valid_unit()
        unit["cells"].pop("C00")
        self.assertIn("missing cells: C00", recap.validate_recap_unit(unit))

        unit = valid_unit()
        unit["cells"]["C11"]["answer"] += " changed"
        self.assertTrue(
            any("exactly equal" in error for error in recap.validate_recap_unit(unit))
        )

        unit = valid_unit()
        unit["cells"]["extra"] = {"question": "q", "answer": "a"}
        self.assertIn("unexpected cells: extra", recap.validate_recap_unit(unit))

    def test_loader_rejects_duplicate_source_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recap.jsonl"
            row = json.dumps(valid_unit())
            path.write_text(f"{row}\n{row}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate source_id"):
                recap.load_recap_units(path)

    def test_strict_retain_free_never_opens_a_retain_split(self):
        requested_splits = []

        def fake_load_dataset(name, split):
            requested_splits.append(split)
            if str(split).startswith("retain"):
                raise AssertionError("retain split was accessed")
            return {
                "train": FakeDataset(
                    [
                        {
                            "question": "q",
                            "answer": "a",
                            "paraphrased_question": "pq",
                            "paraphrased_answer": "pa",
                            "perturbed_answer": ["xa"],
                        }
                    ]
                )
            }

        tofu = import_tofu_with_fake_dependencies(fake_load_dataset)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recap.jsonl"
            path.write_text(json.dumps(valid_unit()) + "\n", encoding="utf-8")
            module = tofu.ToFU_DataModule(
                split="forget10_perturbed",
                tokenizer=object(),
                conv_template_config={},
                data_role="recap_a1",
                counterfactual_path=str(path),
                strict_retain_free=True,
            )

        self.assertNotIn("retain", module.eval_sets)
        self.assertEqual(module.max_len, 256)
        self.assertEqual((module.forget_length, module.retain_length), (1, 1))
        self.assertEqual(module.forget_data[0], valid_unit()["cells"]["C11"])
        self.assertEqual(module.forget_data[1], valid_unit()["cells"]["C01"])
        self.assertTrue(requested_splits)
        self.assertFalse(any(str(split).startswith("retain") for split in requested_splits))

    def test_recap_mode_requires_strict_retain_free(self):
        unit = valid_unit()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recap.jsonl"
            path.write_text(json.dumps(unit) + "\n", encoding="utf-8")

            requested_splits = []

            def fake_load_dataset(name, split):
                requested_splits.append(split)
                return {
                    "train": FakeDataset(
                        [
                            {
                                "question": "q",
                                "answer": "a",
                                "paraphrased_question": "pq",
                                "paraphrased_answer": "pa",
                                "perturbed_answer": ["xa"],
                            }
                        ]
                    )
                }

            tofu = import_tofu_with_fake_dependencies(fake_load_dataset)
            with self.assertRaisesRegex(ValueError, "requires strict_retain_free"):
                tofu.ToFU_DataModule(
                    split="forget10_perturbed",
                    tokenizer=object(),
                    conv_template_config={},
                    data_role="recap_a1",
                    counterfactual_path=str(path),
                )
            self.assertEqual(requested_splits, [])

    def test_missing_recap_role_is_rejected_before_dataset_access(self):
        requested_splits = []

        def fake_load_dataset(name, split):
            requested_splits.append(split)
            return {
                "train": FakeDataset(
                    [
                        {
                            "question": "q",
                            "answer": "a",
                            "paraphrased_question": "pq",
                            "paraphrased_answer": "pa",
                            "perturbed_answer": ["xa"],
                        }
                    ]
                )
            }

        tofu = import_tofu_with_fake_dependencies(fake_load_dataset)
        with self.assertRaisesRegex(ValueError, "Unknown RECAP data role"):
            tofu.ToFU_DataModule(
                split="forget10_perturbed",
                tokenizer=object(),
                conv_template_config={},
            )
        self.assertEqual(requested_splits, [])


if __name__ == "__main__":
    unittest.main()
