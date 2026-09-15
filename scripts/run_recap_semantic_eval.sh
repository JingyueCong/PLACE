#!/usr/bin/env bash
# Reproduce the frozen 200-item RECAP-B0 semantic-robustness protocol.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
MODE="${MODE:-all}" # preflight | generate | judge | all

BASE_MODEL_ID="${RECAP_BASE_MODEL_ID:-open-unlearning/tofu_Llama-3.1-8B-Instruct_full}"
BASE_MODEL="${BASE_MODEL:-$BASE_MODEL_ID}"
A1_CHECKPOINT="${RECAP_A1_CHECKPOINT:-}"
A2_CHECKPOINT="${RECAP_A2_CHECKPOINT:-}"
STANDARD_REPORT="${RECAP_STANDARD_REPORT:-}"
A1_REFERENCE="${RECAP_A1_REFERENCE:-}"
A2_REFERENCE="${RECAP_A2_REFERENCE:-}"
OUTPUT_DIR="${RECAP_SEMANTIC_OUTPUT_DIR:-${REPO_ROOT}/outputs/semantic_recap_b0}"
GENERATIONS="${OUTPUT_DIR}/recap_b0_generations.json"
JUDGE_OUTPUT="${OUTPUT_DIR}/recap_b0_laaj.json"

JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-mini}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://api.openai.com/v1}"
JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-OPENAI_API_KEY}"
JUDGE_WORKERS="${JUDGE_WORKERS:-8}"

usage() {
    cat <<'EOF'
Required environment variables:
  RECAP_A1_CHECKPOINT   rank-64 seed-42 A1 checkpoint-168
  RECAP_A2_CHECKPOINT   rank-64 seed-42 A2 checkpoint-144
  RECAP_STANDARD_REPORT frozen standard-evaluation RECAP_REPORT.json

Optional:
  RECAP_A1_REFERENCE / RECAP_A2_REFERENCE (default: checkpoint sibling fullmodel)
  RECAP_BASE_MODEL_ID / BASE_MODEL (BASE_MODEL must match the standard report)
  MODE=preflight|generate|judge|all
  JUDGE_MODEL, JUDGE_BASE_URL, JUDGE_API_KEY_ENV, JUDGE_WORKERS

The judge key is read only from the environment named by JUDGE_API_KEY_ENV.
EOF
}

case "$MODE" in
    preflight|generate|judge|all) ;;
    *) echo "Invalid MODE: $MODE" >&2; usage >&2; exit 2 ;;
esac

for required in "$A1_CHECKPOINT" "$A2_CHECKPOINT" "$STANDARD_REPORT"; do
    if [ -z "$required" ] || [ ! -e "$required" ]; then
        echo "Missing required RECAP artifact: ${required:-<unset>}" >&2
        usage >&2
        exit 2
    fi
done

if [ -z "$A1_REFERENCE" ]; then
    A1_REFERENCE="$(cd "$A1_CHECKPOINT/../fullmodel" && pwd)"
fi
if [ -z "$A2_REFERENCE" ]; then
    A2_REFERENCE="$(cd "$A2_CHECKPOINT/../fullmodel" && pwd)"
fi
for reference in "$A1_REFERENCE" "$A2_REFERENCE"; do
    [ -d "$reference" ] || { echo "Missing assistant reference: $reference" >&2; exit 2; }
done
if [ "$(cd "$A1_REFERENCE" && pwd -P)" = "$(cd "$A2_REFERENCE" && pwd -P)" ]; then
    echo "A1 and A2 must use distinct role-specific reference directories." >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"
generate_args=(
    --kind recap
    --method RECAP
    --base "$BASE_MODEL"
    --a1 "$A1_CHECKPOINT"
    --a2 "$A2_CHECKPOINT"
    --reference-a1 "$A1_REFERENCE"
    --reference-a2 "$A2_REFERENCE"
    --weight-a1 -2.1
    --weight-a2 2.2
    --top-filter 0.00017
    --composition-mode reference_delta
    --standard-report "$STANDARD_REPORT"
    --require-recap-b0
    --expected-eos-token-ids 128001,128008,128009
    --ou-repo "$REPO_ROOT/open-unlearning"
    --n 200
    --seed 42
    --batch-size 1
    --max-new-tokens 128
    --out "$GENERATIONS"
)

if [ "$MODE" = preflight ]; then
    "$PYTHON" "$REPO_ROOT/scripts/generate_forget10.py" \
        "${generate_args[@]}" --preflight-only
    exit 0
fi

if [ "$MODE" = generate ] || [ "$MODE" = all ]; then
    "$PYTHON" "$REPO_ROOT/scripts/generate_forget10.py" "${generate_args[@]}"
fi

if [ "$MODE" = judge ] || [ "$MODE" = all ]; then
    [ -s "$GENERATIONS" ] || { echo "Missing generations: $GENERATIONS" >&2; exit 2; }
    "$PYTHON" "$REPO_ROOT/scripts/eval_laaj.py" \
        --gens "$GENERATIONS" \
        --out "$JUDGE_OUTPUT" \
        --method RECAP \
        --judge-model "$JUDGE_MODEL" \
        --base-url "$JUDGE_BASE_URL" \
        --api-key-env "$JUDGE_API_KEY_ENV" \
        --api-profile openai \
        --workers "$JUDGE_WORKERS" \
        --reasoning-effort low \
        --max-tokens 512 \
        --expected-composition-mode reference_delta \
        --expected-stop-policy-version model_multi_eos_v1 \
        --expected-eos-token-ids 128001,128008,128009 \
        --require-recap-b0
fi

echo "RECAP semantic protocol complete: $OUTPUT_DIR"
