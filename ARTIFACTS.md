# External artifact manifest

This repository intentionally keeps generated data, model weights, evaluation
caches, API credentials, standard reports, raw generations, and per-item judge
records out of Git. The B0 runner requires explicit paths so that a missing
artifact cannot silently fall back to a local experiment directory.

## Required artifacts

| Logical artifact | Required identity | Included |
|---|---|---|
| Deployed model | `open-unlearning/tofu_Llama-3.1-8B-Instruct_full` | No; obtain under the model provider's terms |
| A1 factorial data | 400 ordered Forget10 rows; SHA-256 below | No; publication pending |
| A2 FullAnswer factorial data | 400 ordered Forget10 rows; SHA-256 below | No; publication pending |
| A1 assistant | seed 42, rank 64, selected `checkpoint-168` | No; publication pending |
| A1 frozen reference | depth-matched initialization for the A1 assistant | No; publication pending |
| A2 assistant | seed 42, rank 64, selected `checkpoint-144` | No; publication pending |
| A2 frozen reference | depth-matched initialization for the A2 assistant | No; publication pending |
| TOFU retain90 evaluation log | frozen reference log used by the standard evaluator | No; required to reproduce the selected development metrics |
| Standard B0 report | `RECAP_REPORT.json` under the caller-selected task directory | No; generated locally |

The supplied SHA-256 values identify the **JSONL data files**, not the model
checkpoints:

```text
A1  3040d25d6c3799705b01707eb0e53a057982a2919b3aa51685eac6ab02fd82f0
A2  b8b8403ef99f531448ee683c0b37d30cc9bc705112b737ecd8b22984965e405e
```

Checkpoint tree digests and stable download locations have not yet been
published. Until they are added here, the repository should be described as a
code release, not as a self-contained reproduction artifact.

## Paired-data contract

Each non-empty JSONL line is one author-linked factorial unit with a unique
`source_id` and exactly four cells: `C11`, `C01`, `C10`, and `C00`. For the
frozen artifacts:

- each file has exactly 400 rows in canonical Forget10 order;
- A1 trains its fit term on `C11` and control term on `C01`;
- A2 trains its fit term on `C10` and control term on `C00`;
- source IDs and `C11` match row by row across the two files;
- the A1 and A2 replacement identities differ for all 400 paired rows;
- no A1 replacement identity appears in its paired A2 `C00` cell.

Thus A1 and A2 use different replacement worlds. The validator checks
structural and cross-file invariants and, in `recap-b0` mode, enforces the two
published file hashes. The identity-in-`C00` statement above comes from the
offline evidence audit; a structural pass is not a substitute for semantic or
human review of generated counterfactuals.

Before training, verify both files with the release validator and with the
platform SHA-256 utility:

```bash
python scripts/validate_recap_tofu_data.py \
  --a1 "$PWD/artifacts/recap-b0/a1.jsonl" \
  --a2 "$PWD/artifacts/recap-b0/a2.jsonl" \
  --profile recap-b0
sha256sum "$PWD/artifacts/recap-b0/a1.jsonl" "$PWD/artifacts/recap-b0/a2.jsonl"
```

On macOS, use `shasum -a 256` in place of `sha256sum`.

## Checkpoint contract

Each role needs both its selected LoRA checkpoint and the role-specific frozen
assistant initialization used for reference-delta inference. A checkpoint is
not sufficient by itself. The expected adapter configuration for both roles is:

```text
r = 64
lora_alpha = 128
lora_dropout = 0.05
target_modules = q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
bias = none
A1 selected step = 168
A2 selected step = 144
```

The A1 and A2 reference paths must not be collapsed to one shared reference:
the verified B0 deployment used separate role-specific references. The release
runner also checks the selected step names (`checkpoint-168` and
`checkpoint-144`) when B0-strict mode is enabled.

## Publication checklist

Before labeling this repository a complete ICLR artifact, the maintainers must:

1. publish the two immutable JSONL files at the hashes above;
2. publish A1/A2 checkpoints and both reference snapshots with tree digests;
3. publish or identify the exact frozen retain90 evaluation log;
4. publish the standard B0 report and a machine-readable run manifest;
5. add stable, preferably archival, download identifiers to this file;
6. verify the repository from a clean checkout without private paths or caches;
7. select and add a root-level license for the RECAP-specific additions.
