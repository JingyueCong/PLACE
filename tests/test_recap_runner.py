import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_recap_tofu_8b.sh"
SEMANTIC_RUNNER = ROOT / "scripts" / "run_recap_semantic_eval.sh"


def test_shell_entrypoints_parse():
    for script in (RUNNER, SEMANTIC_RUNNER):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_paper_runner_freezes_b0_and_has_no_machine_paths():
    source = RUNNER.read_text(encoding="utf-8")
    for expected in (
        "trainer.max_steps=\"$steps\"",
        "trainer.optim=adamw_torch",
        "trainer.report_to=none",
        "ULD_ATTN_IMPLEMENTATION=sdpa",
        "model.model_args.weight_a1=-2.1",
        "model.model_args.weight_a2=2.2",
        "model.model_args.top_logit_filter=0.00017",
        "eval=tofu_recap",
        "eval.tofu.overwrite=true",
        "--validate-retain-log",
        "selection_retain_access=true",
        "distinct role-specific reference directories",
    ):
        assert expected in source
    for forbidden in ("/home/", "/data/", "/Users/"):
        assert forbidden not in source


def test_knowledge_truth_ratio_is_in_the_recap_suite():
    recap_suite = (
        ROOT / "open-unlearning" / "configs" / "eval" / "tofu_recap.yaml"
    ).read_text(encoding="utf-8")
    assert "forget_Truth_Ratio_Knowledge" in recap_suite

    eval_config = (
        ROOT / "open-unlearning" / "configs" / "eval.yaml"
    ).read_text(encoding="utf-8")
    assert "eval: tofu_recap" in eval_config
