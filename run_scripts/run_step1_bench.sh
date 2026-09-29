#!/bin/bash
# ============================================================
# run_step1_bench.sh — Step 1 Benchmark (AHC/UHC) launcher
#
# Usage:
#   bash run_scripts/run_step1_bench.sh --model Qwen3.5-9B-Base
#   bash run_scripts/run_step1_bench_all.sh
#   bash run_scripts/run_step1_bench.sh --dry-run
#   bash run_scripts/run_step1_bench.sh --temperature 0.8
# ============================================================
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_ROOT="${MODEL_ROOT:-./models}"
DEFAULT_MODEL="Qwen3.5-2B-Base"
MODELS=(
    Qwen3.5-2B-Base
    Qwen3.5-9B-Base
    Qwen3.5-35B-A3B-Base
)

PYTHON_MODULE="src.step1_pipeline"
CONFIGS="--config configs/model.yaml configs/judge.yaml configs/data_bench.yaml"

# ---- Parse --model and --all-models, collect remaining args ----
ALL_MODELS=false
MODEL_NAME=""
USER_TP=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all-models)
            ALL_MODELS=true
            shift
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --tensor-parallel-size)
            USER_TP="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ---- Per-model tensor-parallel size ----
get_tp_size() {
    case "$1" in
        Qwen3.5-2B-Base)       echo 1 ;;
        Qwen3.5-9B-Base)       echo 3 ;;
        Qwen3.5-35B-A3B-Base)  echo 4 ;;
        *)                      echo 1 ;;
    esac
}

# ---- Run function ----
run_one() {
    local model="$1"
    local model_path="$MODEL_ROOT/$model"
    local tp_size
    if [[ -n "$USER_TP" ]]; then
        tp_size="$USER_TP"
    else
        tp_size=$(get_tp_size "$model")
    fi
    echo ""
    echo "================================================================"
    echo "  [bench] model: $model  (TP=$tp_size)"
    echo "================================================================"
    mkdir -p logs outputs
    python -m $PYTHON_MODULE $CONFIGS \
        --model-path "$model_path" \
        --tensor-parallel-size "$tp_size" \
        "${EXTRA_ARGS[@]}"
}

# ---- Execute ----
if $ALL_MODELS; then
    for model in "${MODELS[@]}"; do
        run_one "$model"
    done
    echo ""
    echo "All models finished. Results in outputs/step1_bench/*/"
else
    run_one "${MODEL_NAME:-$DEFAULT_MODEL}"
fi
