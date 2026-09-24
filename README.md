# PLACE

PLACE is an anonymous research implementation for machine unlearning
experiments on TOFU with Llama-3.1-8B-Instruct. The repository includes
training, evaluation, validation, and semantic-robustness utilities.

## Anonymous review

This repository is prepared for anonymous review. Author names, affiliations,
contact details, and identifying project links are intentionally omitted. Some
internal script names and environment variables retain the `RECAP` prefix so
that they continue to match the released code.

## Installation

Install a CUDA-compatible PyTorch build for the host, then install the project
dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e ULD
```

The reference workflow used Python 3.10 for training and Python 3.11 for
evaluation. Separate environments can be selected with `TRAIN_PYTHON` and
`EVAL_PYTHON`.

## Quick start

Commands should be run from the repository root. The main runner requires the
external artifacts documented in [ARTIFACTS.md](ARTIFACTS.md):

```bash
export RECAP_A1_DATA="$PWD/artifacts/recap-b0/a1.jsonl"
export RECAP_A2_DATA="$PWD/artifacts/recap-b0/a2.jsonl"
export RECAP_RETAIN_LOGS="$PWD/artifacts/recap-b0/retain90/TOFU_EVAL.json"
export RECAP_MODEL_ROOT="$PWD/outputs/recap-b0"
```

Validate the inputs without training:

```bash
MODE=preflight bash scripts/run_recap_tofu_8b.sh
```

Run training and standard evaluation:

```bash
MODE=all bash scripts/run_recap_tofu_8b.sh
```

Use `MODE=train` or `MODE=eval` to run either stage separately. Additional
checkpoint, reference, model, GPU, and interpreter settings are documented in
the runner. Semantic evaluation is available through
`scripts/run_recap_semantic_eval.sh`.

## Artifacts

This is a code-only release. Generated data, model checkpoints, frozen
references, evaluation caches, reports, raw generations, and judge records are
not stored in Git. See [ARTIFACTS.md](ARTIFACTS.md) for required artifact
identities, integrity checks, and publication status.

The selected operating point is development and retain-informed rather than a
test-blind confirmation.

## Repository structure

- `scripts/`: validation, training, evaluation, and report utilities.
- `ULD/`: assistant training implementation derived from ULD.
- `open-unlearning/`: deployment and TOFU evaluation implementation derived
  from OpenUnlearning.
- `tests/`: repository-level regression tests.
- `ARTIFACTS.md`: external artifact contract and availability.
- `NOTICE`: upstream attribution.

## License and citation

The upstream MIT license texts are preserved in `ULD/LICENSE` and
`open-unlearning/LICENSE`; see [NOTICE](NOTICE) for attribution. A root-level
license for PLACE-specific additions and citation metadata will be added after
anonymous review.
