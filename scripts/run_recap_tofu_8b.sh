#!/usr/bin/env bash
# Paper recipe for RECAP-B0 on TOFU Forget10 / Llama-3.1-8B.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODE="${MODE:-all}" # preflight | train | eval | all
TRAIN_PYTHON="${TRAIN_PYTHON:-python}"
EVAL_PYTHON="${EVAL_PYTHON:-python}"
GPU="${GPU:-0}"
BASE_MODEL_ID="${RECAP_BASE_MODEL_ID:-open-unlearning/tofu_Llama-3.1-8B-Instruct_full}"
BASE_MODEL="${BASE_MODEL:-$BASE_MODEL_ID}"

A1_DATA="${RECAP_A1_DATA:-}"
A2_DATA="${RECAP_A2_DATA:-}"
MODEL_ROOT="${RECAP_MODEL_ROOT:-${REPO_ROOT}/outputs/recap_tofu_8b_b0}"
RETAIN_LOGS="${RECAP_RETAIN_LOGS:-}"
TASK_NAME="${RECAP_TASK_NAME:-recap_tofu_llama31_8b_forget10_b0}"

A1_CHECKPOINT="${RECAP_A1_CHECKPOINT:-}"
A2_CHECKPOINT="${RECAP_A2_CHECKPOINT:-}"
A1_REFERENCE="${RECAP_A1_REFERENCE:-}"
A2_REFERENCE="${RECAP_A2_REFERENCE:-}"

usage() {
    cat <<'EOF'
RECAP-B0 / TOFU Forget10 / Llama-3.1-8B

Required in every mode:
  RECAP_A1_DATA       published 400-row hybrid A1 JSONL
  RECAP_A2_DATA       published 400-row FullAnswer A2 JSONL

Required for eval/all:
  RECAP_RETAIN_LOGS   frozen OpenUnlearning retain90 TOFU_EVAL.json

Optional:
  MODE=preflight|train|eval|all       (default: all)
  RECAP_MODEL_ROOT, RECAP_BASE_MODEL_ID, BASE_MODEL
  GPU, TRAIN_PYTHON, EVAL_PYTHON
  RECAP_A1_CHECKPOINT / RECAP_A2_CHECKPOINT (required for eval if not under MODEL_ROOT)
  RECAP_A1_REFERENCE / RECAP_A2_REFERENCE   (default: checkpoint sibling fullmodel)

`train` optimizes assistants without retain examples. `eval` reads the frozen
retain90 log for metrics. The published operating point was retain-informed
during selection (`selection_retain_access=true`).
EOF
}

case "$MODE" in
    preflight|train|eval|all) ;;
    *) echo "Invalid MODE: $MODE" >&2; usage >&2; exit 2 ;;
esac

TRAIN_PYTHON="$(command -v "$TRAIN_PYTHON")" || {
    echo "TRAIN_PYTHON is not executable" >&2
    exit 2
}
EVAL_PYTHON="$(command -v "$EVAL_PYTHON")" || {
    echo "EVAL_PYTHON is not executable" >&2
    exit 2
}

absolute_path() {
    "$TRAIN_PYTHON" -c \
        'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' \
        "$1"
}

for data_path in "$A1_DATA" "$A2_DATA"; do
    if [ -z "$data_path" ] || [ ! -s "$data_path" ]; then
        echo "Missing required RECAP data: ${data_path:-<unset>}" >&2
        usage >&2
        exit 2
    fi
done

# Freeze path interpretation before the runner changes into either framework.
A1_DATA="$(absolute_path "$A1_DATA")"
A2_DATA="$(absolute_path "$A2_DATA")"
MODEL_ROOT="$(absolute_path "$MODEL_ROOT")"
if [ -n "$RETAIN_LOGS" ]; then RETAIN_LOGS="$(absolute_path "$RETAIN_LOGS")"; fi
if [ -n "$A1_CHECKPOINT" ]; then A1_CHECKPOINT="$(absolute_path "$A1_CHECKPOINT")"; fi
if [ -n "$A2_CHECKPOINT" ]; then A2_CHECKPOINT="$(absolute_path "$A2_CHECKPOINT")"; fi
if [ -n "$A1_REFERENCE" ]; then A1_REFERENCE="$(absolute_path "$A1_REFERENCE")"; fi
if [ -n "$A2_REFERENCE" ]; then A2_REFERENCE="$(absolute_path "$A2_REFERENCE")"; fi
if [ -e "$BASE_MODEL" ]; then BASE_MODEL="$(absolute_path "$BASE_MODEL")"; fi

"$TRAIN_PYTHON" "$REPO_ROOT/scripts/validate_recap_tofu_data.py" \
    --a1 "$A1_DATA" \
    --a2 "$A2_DATA" \
    --profile recap-b0 \
    --out "$MODEL_ROOT/data_validation.json"

if [ "$MODE" = preflight ]; then
    echo "RECAP-B0 data preflight passed."
    exit 0
fi

find_exact_checkpoint() {
    local root="$1" step="$2" explicit="$3"
    if [ -n "$explicit" ]; then
        [ -d "$explicit" ] || { echo "Checkpoint does not exist: $explicit" >&2; return 1; }
        [ "$(basename "$explicit")" = "checkpoint-$step" ] || {
            echo "Expected checkpoint-$step, got: $explicit" >&2
            return 1
        }
        printf '%s\n' "$explicit"
        return 0
    fi
    local matches=()
    while IFS= read -r candidate; do
        matches+=("$candidate")
    done < <(find "$root" -type d -name "checkpoint-$step" -print 2>/dev/null | sort)
    if [ "${#matches[@]}" -ne 1 ]; then
        echo "Expected exactly one checkpoint-$step under $root; found ${#matches[@]}." >&2
        echo "Set the corresponding RECAP_A*_CHECKPOINT explicitly." >&2
        return 1
    fi
    printf '%s\n' "${matches[0]}"
}

train_role() {
    local role="$1" data="$2" steps="$3" learning_rate="$4" kl_weight="$5"
    local role_root="$MODEL_ROOT/$role"
    local data_mode="recap_${role}"
    local signature_file="$role_root/RECAP_TRAIN_SIGNATURE.txt"
    local data_sha
    data_sha="$("$TRAIN_PYTHON" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$data")"
    local signature="RECAP-B0|role=$role|data_sha256=$data_sha|base=$BASE_MODEL|max_len=256|layers=4|rank=64|alpha=128|dropout=0.05|lr=$learning_rate|steps=$steps|kl=$kl_weight|batch=1|ga=16|optim=adamw_torch|seed=42|attention=sdpa"

    local existing=()
    while IFS= read -r candidate; do
        existing+=("$candidate")
    done < <(find "$role_root" -type d -name "checkpoint-$steps" -print 2>/dev/null | sort)
    if [ "${#existing[@]}" -gt 0 ]; then
        if [ "${#existing[@]}" -eq 1 ] && [ -s "$signature_file" ] \
            && [ "$(<"$signature_file")" = "$signature" ]; then
            echo "Reusing verified $role checkpoint: ${existing[0]}"
            return 0
        fi
        echo "Refusing ambiguous or unsigned existing $role artifacts under $role_root." >&2
        echo "Use a fresh RECAP_MODEL_ROOT or provide frozen checkpoints for MODE=eval." >&2
        return 1
    fi

    mkdir -p "$role_root"
    echo "Training $role: steps=$steps lr=$learning_rate reference_KL=$kl_weight"
    (
        cd "$REPO_ROOT/ULD"
        CUDA_VISIBLE_DEVICES="$GPU" \
        ULD_ATTN_IMPLEMENTATION=sdpa \
        "$TRAIN_PYTHON" scripts/hf_forget_train.py \
            project="recap_${role}_forget10" \
            data=tofu_chat3 \
            data.dataset.split=forget10_perturbed \
            data_mode="$data_mode" \
            data_mode.counterfactual_path="$data" \
            model=llama-3-8b \
            model.model_path="$BASE_MODEL" \
            model.tokenizer_path="$BASE_MODEL" \
            model_mode=uld \
            model_mode.num_layer=4 \
            model_mode.Lora.r=64 \
            model_mode.Lora.alpha=128 \
            model_mode.Lora.dropout=0.05 \
            unlearn_loss=factorial_reference_preserving \
            unlearn_loss.retain_weight="$kl_weight" \
            trainer.batch_size=1 \
            trainer.gradient_accumulation_steps=16 \
            trainer.learning_rate="$learning_rate" \
            trainer.weight_decay=0.01 \
            trainer.warmup_ratio=0.1 \
            trainer.optim=adamw_torch \
            trainer.report_to=none \
            trainer.max_epochs=1 \
            trainer.max_steps="$steps" \
            +trainer.save_steps="$steps" \
            +trainer.eval_steps="$steps" \
            trainer.seed=42 \
            seed=42 \
            trainer.strategy=gpu \
            OUTPUTMODELDIR="$role_root" \
            postfix="$role" \
            "hydra.run.dir=outputs/tune_log/recap_${role}_forget10/\${now:%Y-%m-%d_%H-%M-%S}"
    )

    find_exact_checkpoint "$role_root" "$steps" "" >/dev/null
    printf '%s\n' "$signature" > "$signature_file"
}

if [ "$MODE" = train ] || [ "$MODE" = all ]; then
    train_role a1 "$A1_DATA" 168 5e-4 0.4
    train_role a2 "$A2_DATA" 144 7.5e-4 0.3
fi

if [ "$MODE" = train ]; then
    echo "RECAP assistant training complete: $MODEL_ROOT"
    exit 0
fi

if [ -z "$RETAIN_LOGS" ] || [ ! -s "$RETAIN_LOGS" ]; then
    echo "RECAP_RETAIN_LOGS is required for the frozen standard evaluation." >&2
    usage >&2
    exit 2
fi

A1_CHECKPOINT="$(find_exact_checkpoint "$MODEL_ROOT/a1" 168 "$A1_CHECKPOINT")"
A2_CHECKPOINT="$(find_exact_checkpoint "$MODEL_ROOT/a2" 144 "$A2_CHECKPOINT")"
if [ -z "$A1_REFERENCE" ]; then A1_REFERENCE="$(cd "$A1_CHECKPOINT/../fullmodel" && pwd)"; fi
if [ -z "$A2_REFERENCE" ]; then A2_REFERENCE="$(cd "$A2_CHECKPOINT/../fullmodel" && pwd)"; fi
for reference in "$A1_REFERENCE" "$A2_REFERENCE"; do
    [ -d "$reference" ] || { echo "Missing assistant reference: $reference" >&2; exit 2; }
done
if [ "$(absolute_path "$A1_REFERENCE")" = "$(absolute_path "$A2_REFERENCE")" ]; then
    echo "A1 and A2 must use distinct role-specific reference directories." >&2
    exit 2
fi

echo "Evaluating RECAP-B0 (retain log is evaluation-only)."
"$EVAL_PYTHON" "$REPO_ROOT/scripts/summarize_recap_tofu.py" \
    --validate-retain-log "$RETAIN_LOGS"
(
    cd "$REPO_ROOT/open-unlearning"
    CUDA_VISIBLE_DEVICES="$GPU" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$EVAL_PYTHON" src/eval.py \
        eval=tofu_recap \
        model=Llama-3.1-8B-Instruct_RECAP \
        model.model_args.pretrained_model_name_or_path="$BASE_MODEL" \
        model.model_args.a1_path="$A1_CHECKPOINT" \
        model.model_args.a2_path="$A2_CHECKPOINT" \
        model.model_args.reference_a1_path="$A1_REFERENCE" \
        model.model_args.reference_a2_path="$A2_REFERENCE" \
        model.model_args.composition_mode=reference_delta \
        model.model_args.weight_a1=-2.1 \
        model.model_args.weight_a2=2.2 \
        model.model_args.top_logit_filter=0.00017 \
        model.model_args.top_logit_filter_a1=0.00017 \
        model.model_args.top_logit_filter_a2=0.00017 \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="$BASE_MODEL" \
        eval.tofu.forget_split=forget10 \
        eval.tofu.holdout_split=holdout10 \
        eval.tofu.retain_logs_path="$RETAIN_LOGS" \
        eval.tofu.batch_size=1 \
        eval.tofu.overwrite=true \
        seed=0 \
        task_name="$TASK_NAME"
)

EVAL_DIR="$REPO_ROOT/open-unlearning/saves/eval/$TASK_NAME"
EVAL_JSON="$EVAL_DIR/TOFU_EVAL.json"
SUMMARY_JSON="$EVAL_DIR/TOFU_SUMMARY.json"
for result in "$EVAL_JSON" "$SUMMARY_JSON"; do
    [ -s "$result" ] || { echo "Evaluation did not create $result" >&2; exit 1; }
done

"$EVAL_PYTHON" "$REPO_ROOT/scripts/summarize_recap_tofu.py" \
    --eval-json "$EVAL_JSON" \
    --summary-json "$SUMMARY_JSON" \
    --output "$EVAL_DIR/RECAP_REPORT.json" \
    --base-model "$BASE_MODEL" \
    --base-model-id "$BASE_MODEL_ID" \
    --a1-checkpoint "$A1_CHECKPOINT" \
    --a2-checkpoint "$A2_CHECKPOINT" \
    --reference-a1 "$A1_REFERENCE" \
    --reference-a2 "$A2_REFERENCE" \
    --retain-logs "$RETAIN_LOGS" \
    --a1-data "$A1_DATA" \
    --a2-data "$A2_DATA"

echo "RECAP-B0 evaluation complete: $EVAL_DIR/RECAP_REPORT.json"
