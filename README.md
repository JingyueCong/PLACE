# RECAP: Code for TOFU Experiments

This repository contains the official implementation for the RECAP
Llama-3.1-8B / TOFU Forget10 experiment. It provides the selected B0
configuration, deterministic data checks, standard TOFU evaluation, and the
semantic-robustness evaluation utilities.

## Scope

This is a **code-only reproduction repository**. Generated factorial data,
model checkpoints, frozen assistant references, benchmark caches, retain
evaluation logs, standard TOFU reports, raw generations, and judge records are
not included, and their publication is pending. See
[ARTIFACTS.md](ARTIFACTS.md) for the external artifact contract and status.

The two incorporated projects retain their upstream MIT licenses. A separate
root-level license for the RECAP-specific additions has not yet been selected;
the maintainers must add one before presenting this repository as a reusable
archival artifact.

## Frozen B0 configuration

B0 is one Llama-3.1-8B-Instruct Forget10 training run (400 questions from 20
author blocks) with seed 42 and one A1/A2 assistant pair.

| Item | A1 (deletion) | A2 (compensation) |
|---|---:|---:|
| Training cells | fit `C11`; control `C01` | fit `C10`; control `C00` |
| Selected checkpoint | step 168 | step 144 |
| Learning rate | `5e-4` | `7.5e-4` |
| Reference-KL coefficient | `0.4` | `0.3` |
| Training seed | 42 | 42 |
| Selected epoch | 3.36 | 2.88 |
| Assistant depth | first 4 transformer layers | first 4 transformer layers |
| LoRA | rank 64, alpha 128, dropout 0.05 | rank 64, alpha 128, dropout 0.05 |

Both LoRA adapters target `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
`up_proj`, and `down_proj`. The effective maximum training sequence length is
256. Training uses per-device batch size 1 with gradient accumulation 16, `adamw_torch`
(`weight_decay = 0.01`), BF16, gradient clipping at 1.0, and linear decay
after 10% warmup. The launch configuration sets `max_epochs = 1`, but the
explicit `max_steps` budget controls training; Trainer records the selected
checkpoints at approximately 3.36 and 2.88 epochs. The backbone is frozen.

The loss for each role is answer cross-entropy on its fit cell plus
answer-masked KL to that role's frozen assistant initialization on its control
cell. Configuration fields named `retain_weight` are the coefficients of these
counterfactual control losses; they do **not** mean that retain examples enter
assistant training.

At inference, B0 uses role-specific reference deltas:

```text
logits = base_logits
       - 2.1 * filter(A1_logits - A1_reference_logits, 0.00017)
       + 2.2 * filter(A2_logits - A2_reference_logits, 0.00017)
```

Routing, learned gates, and alignment calibration are disabled. Evaluation
uses seed 0, greedy decoding, batch size 1, at most 200 new tokens, and no KV
cache. Because the final A1 and A2 assistants have different role-specific
frozen references, a decoding step executes five forward passes: deployed
model, two trained assistants, and two references.

## Data and protocol disclosure

The two roles use separate 400-row factorial JSONL artifacts. They have the
same ordered source IDs and the same `C11` content, but were constructed with
different replacement identities and therefore represent **different
replacement worlds**. The selected B0 result must not be described as a
strict shared-replacement-world 2x2 experiment.

No original TOFU retain example is used to optimize either assistant. However,
checkpoint and inference-coefficient selection used retained-utility metrics:

```text
training_retain_access = false
selection_retain_access = true
```

B0 is therefore a development selection, not a test-blind or independently
held-out confirmation. Multiple operating-point evaluations of this assistant
pair are not independent training repetitions.

## Installation

The audited B0 run used separate training and evaluation environments:

| | Training | Evaluation |
|---|---|---|
| Python | 3.10.20 | 3.11.15 |
| PyTorch | 2.1.1+cu118 | 2.4.1 |
| NumPy | 1.26.4 | 2.2.3 |
| Transformers | 4.51.3 | 4.51.3 |
| PEFT | 0.15.2 | 0.15.2 |
| Accelerate | 0.34.2 | 0.34.2 |
| Tokenizers | 0.21.4 | 0.21.4 |

Install a CUDA-compatible PyTorch build for the host first. The root
requirements file uses Python-version markers to approximate the two audited
environments:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e ULD
```

For the closest environment match, create separate Python 3.10 training and
Python 3.11 evaluation environments and pass their interpreter paths to the
runner. Hardware- and CUDA-specific packages are intentionally not vendored.

## Running B0

For a paper run, set every artifact and output location explicitly. The
examples below use ignored, repository-relative directories; replace them with
the actual published artifact locations.

```bash
python scripts/validate_recap_tofu_data.py \
  --a1 "$PWD/artifacts/recap-b0/a1.jsonl" \
  --a2 "$PWD/artifacts/recap-b0/a2.jsonl" \
  --profile recap-b0

RECAP_A1_DATA="$PWD/artifacts/recap-b0/a1.jsonl" \
RECAP_A2_DATA="$PWD/artifacts/recap-b0/a2.jsonl" \
RECAP_MODEL_ROOT="$PWD/outputs/recap-b0" \
MODE=preflight bash scripts/run_recap_tofu_8b.sh

RECAP_A1_DATA="$PWD/artifacts/recap-b0/a1.jsonl" \
RECAP_A2_DATA="$PWD/artifacts/recap-b0/a2.jsonl" \
RECAP_MODEL_ROOT="$PWD/outputs/recap-b0" \
MODE=train bash scripts/run_recap_tofu_8b.sh
```

`preflight` verifies the two training-data paths and their paired-data contract
without starting training. `train` produces the two role-specific assistants. For an
evaluation of published checkpoints, supply every dependency explicitly:

```bash
RECAP_A1_DATA="$PWD/artifacts/recap-b0/a1.jsonl" \
RECAP_A2_DATA="$PWD/artifacts/recap-b0/a2.jsonl" \
RECAP_A1_CHECKPOINT="$PWD/artifacts/recap-b0/a1/checkpoint-168" \
RECAP_A2_CHECKPOINT="$PWD/artifacts/recap-b0/a2/checkpoint-144" \
RECAP_A1_REFERENCE="$PWD/artifacts/recap-b0/a1/reference" \
RECAP_A2_REFERENCE="$PWD/artifacts/recap-b0/a2/reference" \
RECAP_RETAIN_LOGS="$PWD/artifacts/recap-b0/retain90/TOFU_EVAL.json" \
RECAP_MODEL_ROOT="$PWD/outputs/recap-b0" \
RECAP_TASK_NAME=recap_tofu_llama31_8b_forget10_b0 \
MODE=eval bash scripts/run_recap_tofu_8b.sh
```

The final standard report is written to
`open-unlearning/saves/eval/${RECAP_TASK_NAME}/RECAP_REPORT.json`. `MODE=all`
runs validation, training, and evaluation in order. The exact artifact
identities are recorded in [ARTIFACTS.md](ARTIFACTS.md); the script's `usage`
function documents the complete environment-variable contract. The report
stores the portable backbone ID separately from the concrete local path used
by a run, so relocating a published copy does not relabel the model.

## Semantic-robustness evaluation

The semantic evaluation is deliberately separate from the standard TOFU
metrics. It deterministically samples 200 of the 400 Forget10 items with seed
42, generates with the frozen B0 protocol (including the corrected multi-EOS
stopping rule), and then applies a strict LLM-as-a-judge scorer:

```bash
python scripts/generate_forget10.py --help
python scripts/eval_laaj.py --help

RECAP_A1_CHECKPOINT="$PWD/artifacts/recap-b0/a1/checkpoint-168" \
RECAP_A2_CHECKPOINT="$PWD/artifacts/recap-b0/a2/checkpoint-144" \
RECAP_A1_REFERENCE="$PWD/artifacts/recap-b0/a1/reference" \
RECAP_A2_REFERENCE="$PWD/artifacts/recap-b0/a2/reference" \
RECAP_STANDARD_REPORT="$PWD/open-unlearning/saves/eval/recap_tofu_llama31_8b_forget10_b0/RECAP_REPORT.json" \
RECAP_SEMANTIC_OUTPUT_DIR="$PWD/outputs/semantic-recap-b0" \
MODE=preflight bash scripts/run_recap_semantic_eval.sh
```

`scripts/run_recap_semantic_eval.sh` wires these two steps together while still
requiring explicit checkpoint and frozen standard-report paths; the example
also freezes the output path. Semantic generation uses at most 128 new tokens,
distinct from the 200-token cap in the standard TOFU evaluation. Judge
credentials must be provided through environment variables and must never be
committed.

Judge outputs are not committed because provider outputs are nondeterministic
and may contain per-item generations. Publish paper-facing aggregates and their
provenance as a separately reviewed artifact.

## Repository map

- `scripts/run_recap_tofu_8b.sh`: frozen B0 preflight/train/eval entry point.
- `scripts/validate_recap_tofu_data.py`: deterministic paired-data validator.
- `scripts/summarize_recap_tofu.py`: fail-closed standard-report builder.
- `scripts/generate_forget10.py`: fixed-sample, resumable generation.
- `scripts/eval_laaj.py`: strict, resumable LLM-as-a-judge evaluation.
- `scripts/run_recap_semantic_eval.sh`: frozen semantic protocol wrapper.
- `ULD/`: assistant training implementation derived from ULD.
- `open-unlearning/`: deployment and TOFU evaluation implementation derived
  from OpenUnlearning.
- `tests/`: repository-surface regression tests.

## Licensing and citation

This repository preserves the upstream MIT license texts in `ULD/LICENSE` and
`open-unlearning/LICENSE`; attribution details are in [NOTICE](NOTICE). A paper
citation and archival artifact identifier will be added when the public ICLR
artifact is finalized. Do not cite an experiment-directory name as if it were
a separate method or a seed-averaged result.
