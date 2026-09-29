#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$PROJECT_ROOT/tests/run_step4_35b_train5000_eval400_seed1234.sh"
QWEN35B_SAE_FEATURE_COUNT="${QWEN35B_SAE_FEATURE_COUNT:-32768}"

cd "$PROJECT_ROOT"

validate_qwen35b_feature_id() {
  local experiment="$1"
  local feature_id="$2"
  if ! [[ "$feature_id" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $experiment FEATURE_ID must be a non-negative integer: $feature_id" >&2
    exit 1
  fi
  if [[ "$feature_id" -ge "$QWEN35B_SAE_FEATURE_COUNT" ]]; then
    echo "ERROR: $experiment FEATURE_ID=$feature_id is invalid for Qwen3.5-35B-A3B-Base SAE w${QWEN35B_SAE_FEATURE_COUNT}; valid ids are 0..$((QWEN35B_SAE_FEATURE_COUNT - 1))." >&2
    echo "       Do not reuse 9B feature ids such as 61718 for the 35B SAE." >&2
    exit 1
  fi
}

validate_qwen35b_feature_id "random6868" "6868"
validate_qwen35b_feature_id "alpaca2000" "${ALPACA2000_FEATURE_ID:-2362}"

run_experiment() {
  local experiment="$1"

  case "$experiment" in
    random6868)
      echo
      echo "============================================================"
      echo "Step4 35B-A3B train experiment: random6868"
      echo "============================================================"
      EXPERIMENT_KEY="random6868" \
      STEP4_DLC_COORD_SCOPE="random6868" \
      DATASET_TAG="train5000_eval400_seed1234_syco4000_alpaca1000" \
      SPLIT_TAG="split_train5000_eval400_seed1234_syco4000_alpaca1000" \
      SYCO_SPLIT_TRAIN_SIZE="5000" \
      SYCO_SPLIT_EVAL_SIZE="400" \
      SYCO_SPLIT_SEED="1234" \
      OUTPUT_PREFIX="random6868_" \
      FEATURE_ID="6868" \
      FEATURE_TAG="f6868" \
      LAYER="${RANDOM6868_LAYER:-${LAYER:-27}}" \
      bash "$SCRIPT"
      ;;
    alpaca2000)
      echo
      echo "============================================================"
      echo "Step4 35B-A3B train experiment: alpaca2000"
      echo "============================================================"
      EXPERIMENT_KEY="alpaca2000" \
      STEP4_DLC_COORD_SCOPE="alpaca2000" \
      DATASET_TAG="train2000_eval400_alpaca2000" \
      SPLIT_TAG="split_train2000_eval400_seed1234_alpaca2000" \
      SYCO_SPLIT_TRAIN_SIZE="2000" \
      SYCO_SPLIT_EVAL_SIZE="400" \
      SYCO_SPLIT_SEED="1234" \
      OUTPUT_PREFIX="syco${ALPACA2000_FEATURE_ID:-2362}_" \
      FEATURE_ID="${ALPACA2000_FEATURE_ID:-2362}" \
      FEATURE_TAG="${ALPACA2000_FEATURE_TAG:-f${ALPACA2000_FEATURE_ID:-2362}}" \
      LAYER="${ALPACA2000_LAYER:-${LAYER:-27}}" \
      bash "$SCRIPT"
      ;;
    *)
      echo "ERROR: unknown 35B experiment: $experiment" >&2
      echo "       valid experiments: random6868, alpaca2000" >&2
      exit 1
      ;;
  esac
}

echo
echo "============================================================"
echo "Step4 35B-A3B two-experiment training only"
echo "order      : random6868, alpaca2000"
echo "script     : $SCRIPT"
echo "eval       : disabled"
echo "============================================================"

run_experiment random6868
run_experiment alpaca2000

echo
echo "All requested 35B-A3B training experiments completed."
