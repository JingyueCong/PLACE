# PLACE: Code for TOFU Experiments

PLACE is an anonymous research implementation for machine unlearning
experiments on TOFU with Llama-3.1-8B-Instruct. The repository includes
training, evaluation, validation, and semantic-robustness utilities.

## Anonymous review

This repository is prepared for anonymous review. Author names, affiliations,
contact details, and identifying project links are intentionally omitted.

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

## Usage

Commands should be run from the repository root. Obtain the external artifacts
documented in [ARTIFACTS.md](ARTIFACTS.md), then use the entry points in
`scripts/` for data validation, training, standard evaluation, and semantic
evaluation. Required paths and runtime options are documented in each runner.

Run the repository-level checks with:

```bash
python -m pytest -q
```

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
