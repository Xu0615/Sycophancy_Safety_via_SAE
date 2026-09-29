#!/bin/bash
# ============================================================
# run_step2_syco_feature.sh — Extract SAE sycophancy features
# from paired syco_dataset.jsonl responses.
#
# Usage:
#   bash run_scripts/run_step2_syco_feature.sh
#   bash run_scripts/run_step2_syco_feature.sh --model-name Qwen3.5-9B-Base
#   bash run_scripts/run_step2_syco_feature.sh --model-name Qwen3.5-35B-A3B-Base --overwrite
# ============================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${MODEL_ROOT:-./models}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

MODEL_NAME="Qwen3.5-2B-Base"
MODEL_PATH=""
CONFIG="$PROJECT_ROOT/configs/step2.yaml"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-name|--model_name|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --model-path|--model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$MODEL_PATH" ]]; then
    MODEL_PATH="$MODEL_ROOT/$MODEL_NAME"
fi

DATASET_PATH="$PROJECT_ROOT/outputs/step2/$MODEL_NAME/syco_dataset/syco_dataset.jsonl"
OUTPUT_DIR="$PROJECT_ROOT/outputs/step2/$MODEL_NAME/syco_feature"
LOG_FILE="$PROJECT_ROOT/logs/step2/syco_feature_${MODEL_NAME}.log"

mkdir -p "$OUTPUT_DIR" "$PROJECT_ROOT/logs/step2"
cd "$PROJECT_ROOT"

"$PYTHON_BIN" -m src.step2_syco_feature \
    --config "$CONFIG" \
    --model-name "$MODEL_NAME" \
    --model-path "$MODEL_PATH" \
    --dataset-path "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --log-file "$LOG_FILE" \
    "${EXTRA_ARGS[@]}"
