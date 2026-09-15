import datasets
from datasets import load_dataset

from .conv_util import create_template
from .datamodule import TrainDataModule
from .recap import load_recap_units, recap_dual_roles


class ToFU_DataModule(TrainDataModule):
    def __init__(
        self,
        split,
        tokenizer,
        conv_template_config,
        name="locuslab/TOFU",
        max_len=256,
        batch_size=8,
        with_retain=False,
        with_dpo=False,
        expand_forget=False,
        with_perturb=False,
        data_role=None,
        counterfactual_path=None,
        strict_retain_free=False,
        **kwargs,
    ):
        super().__init__()

        if "max_length" in kwargs:
            raise ValueError(
                "Use data.dataset.max_len; max_length is not a supported "
                "ToFU training field"
            )

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.dpo_mode = with_dpo
        self.conv_template = create_template(conv_template_config, tokenizer=tokenizer)

        if data_role not in {"recap_a1", "recap_a2"}:
            raise ValueError(f"Unknown RECAP data role: {data_role!r}")
        if not strict_retain_free:
            raise ValueError(f"{data_role} requires strict_retain_free=True")
        if counterfactual_path is None:
            raise ValueError(f"{data_role} requires counterfactual_path")
        if with_retain or expand_forget or with_perturb or with_dpo:
            raise ValueError(
                f"{data_role} accepts only retain-free frozen 2x2 JSONL supervision"
            )

        def flatten_perturb(perturb_dataset):
            for sample in perturb_dataset:
                perturb_answer_list = sample.get("perturbed_answer", [])
                for perturb_ans in perturb_answer_list[:1]:
                    yield {"question": sample["question"], "answer": perturb_ans}

        source_eval = load_dataset(name, split)["train"]

        forget_eval = source_eval
        cols_to_drop = [
            column
            for column in (
                "paraphrased_answer",
                "paraphrased_question",
                "perturbed_answer",
            )
            if column in forget_eval.column_names
        ]
        if cols_to_drop:
            forget_eval = forget_eval.remove_columns(cols_to_drop)
        self.forget_eval = forget_eval

        perturb_eval = source_eval
        if "perturbed_answer" in perturb_eval.column_names:
            perturb_eval = datasets.Dataset.from_generator(
                flatten_perturb,
                gen_kwargs={"perturb_dataset": perturb_eval},
            )
        self.perturb_eval = perturb_eval

        paraphrase_eval = source_eval
        if "paraphrased_answer" in paraphrase_eval.column_names:
            columns = [
                column
                for column in ("answer", "perturbed_answer", "paraphrased_question")
                if column in paraphrase_eval.column_names
            ]
            paraphrase_eval = paraphrase_eval.remove_columns(columns)
            paraphrase_eval = paraphrase_eval.rename_column(
                "paraphrased_answer", "answer"
            )
        self.paraphrase_eval = paraphrase_eval

        units = load_recap_units(counterfactual_path)
        fit_rows, control_rows = recap_dual_roles(units, data_role)
        fit_data = datasets.Dataset.from_list(fit_rows)
        control_data = datasets.Dataset.from_list(control_rows)

        self.forget_length = len(fit_data)
        # The inherited sampler/loss APIs call the control partition `retain`;
        # these rows are matched counterfactual controls, not TOFU retain data.
        self.retain_length = len(control_data)
        self.forget_data = datasets.concatenate_datasets([fit_data, control_data])
        self.eval_sets = {
            "forget": self.forget_eval,
            "perturb": self.perturb_eval,
            "paraphrase": self.paraphrase_eval,
        }

        fit_cell, control_cell = (
            ("C11", "C01")
            if data_role == "recap_a1"
            else ("C10", "C00")
        )
        print(
            f"Loaded RECAP {data_role}: "
            f"fit={fit_cell}({self.forget_length}), "
            f"control={control_cell}({self.retain_length}) from "
            f"{counterfactual_path}"
        )
