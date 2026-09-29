#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_NAME="Qwen3.5-35B-A3B-Base"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/step4_feature_inject}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FEATURE_SWEEP_VALUES="${FEATURE_SWEEP_VALUES:-16082 666 888}"
SWEEP_TIMESTAMP="${SWEEP_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

BASE_EXPERIMENT_TAG="repro_f2362_train2000_eval400_syco1000_alpaca1000_seed1234"
TARGET_TAG_SUFFIX="train2000_eval400_syco1000_alpaca1000_seed1234"
SFT_RUN_NAME="syco_sft_full_lr8.1e-6_ep1_gbs64"
BASE_TRAIN_ROOT="$OUTPUT_ROOT/train/$MODEL_NAME/$BASE_EXPERIMENT_TAG"
BASE_EVAL_ROOT="$OUTPUT_ROOT/eval/$MODEL_NAME/$BASE_EXPERIMENT_TAG"
BASE_SFT_DIR="$BASE_TRAIN_ROOT/$SFT_RUN_NAME"

require_baseline() {
    local required
    for required in \
        "$BASE_SFT_DIR/training_summary.json" \
        "$BASE_SFT_DIR/loss_history.csv" \
        "$BASE_SFT_DIR/model.safetensors.index.json" \
        "$BASE_EVAL_ROOT/base/run_summary.md" \
        "$BASE_EVAL_ROOT/$SFT_RUN_NAME/run_summary.md"; do
        if [[ ! -s "$required" ]]; then
            echo "ERROR: reusable f2362 baseline artifact is missing: $required" >&2
            exit 1
        fi
    done

    "$PYTHON_BIN" - "$BASE_SFT_DIR/training_summary.json" <<'PY'
import json
import math
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    summary = json.load(handle)

checks = {
    "group": summary.get("group") == "syco_sft",
    "injection": summary.get("injection") == "none",
    "learning_rate": math.isclose(float(summary.get("learning_rate", 0)), 8.1e-6),
    "epochs": math.isclose(float(summary.get("epochs", 0)), 1.0),
    "global_batch_size": int(summary.get("global_effective_batch_size", 0)) == 64,
    "per_device_batch_size": int(summary.get("per_device_batch_size", 0)) == 4,
    "hf_eval_ready": bool((summary.get("full_model_save") or {}).get("full_model_eval_ready")),
}
failed = sorted(name for name, ok in checks.items() if not ok)
if failed:
    raise SystemExit(f"f2362 SFT baseline does not match the requested contract: {failed}")
PY
}

seed_shared_baseline() {
    local feature_id="$1"
    local experiment_tag="repro_f${feature_id}_${TARGET_TAG_SUFFIX}"
    local train_root="$OUTPUT_ROOT/train/$MODEL_NAME/$experiment_tag"
    local eval_root="$OUTPUT_ROOT/eval/$MODEL_NAME/$experiment_tag"
    local target_sft_dir="$train_root/$SFT_RUN_NAME"

    mkdir -p "$train_root" "$eval_root"
    # The SFT group has no feature injection. Hard links preserve the exact
    # f2362 baseline weights without consuming another 65 GiB per feature.
    # _vllm_view contains relative symlinks and is regenerated independently.
    mkdir -p "$target_sft_dir"
    find "$BASE_SFT_DIR" -mindepth 1 -maxdepth 1 ! -name _vllm_view \
        -exec cp -al --no-clobber -t "$target_sft_dir" {} +
    rm -rf "$target_sft_dir/_vllm_view"
    for required in training_summary.json loss_history.csv model.safetensors.index.json; do
        if [[ ! -s "$target_sft_dir/$required" ]]; then
            echo "ERROR: reusable SFT baseline copy is incomplete: $target_sft_dir/$required" >&2
            exit 1
        fi
    done
    if [[ ! -e "$eval_root/base" ]]; then
        cp -a "$BASE_EVAL_ROOT/base" "$eval_root/base"
        rm -f "$eval_root/base/comparison_summary.md"
    fi
    if [[ ! -e "$eval_root/$SFT_RUN_NAME" ]]; then
        cp -a "$BASE_EVAL_ROOT/$SFT_RUN_NAME" "$eval_root/$SFT_RUN_NAME"
        rm -f "$eval_root/$SFT_RUN_NAME/comparison_summary.md"
    fi

    "$PYTHON_BIN" - "$train_root/baseline_reuse.json" "$feature_id" \
        "$BASE_SFT_DIR" "$BASE_EVAL_ROOT" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "target_feature_id": int(sys.argv[2]),
    "reason": "syco_sft uses injection=none; share the exact f2362 comparison baseline",
    "train_artifact_mode": "hardlink_copy",
    "source_sft_train_dir": os.path.abspath(sys.argv[3]),
    "copied_eval_runs": ["base", "syco_sft_full_lr8.1e-6_ep1_gbs64"],
    "source_eval_root": os.path.abspath(sys.argv[4]),
}
tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, path)
PY
}

run_feature() {
    local feature_id="$1"
    local experiment_tag="repro_f${feature_id}_${TARGET_TAG_SUFFIX}"
    local run_timestamp="${SWEEP_TIMESTAMP}_f${feature_id}"

    seed_shared_baseline "$feature_id"
    echo "============================================================"
    echo "Starting 35B feature experiment: f${feature_id}"
    echo "experiment tag: $experiment_tag"
    echo "run timestamp : $run_timestamp"
    echo "============================================================"
    FEATURE_ID="$feature_id" \
    EXPERIMENT_TAG="$experiment_tag" \
    RUN_TIMESTAMP="$run_timestamp" \
        bash "$SCRIPT_DIR/run_step4_syco_train&eval_35a3b.sh"
}

require_baseline
read -ra FEATURES <<< "$FEATURE_SWEEP_VALUES"
if (( ${#FEATURES[@]} == 0 )); then
    echo "ERROR: FEATURE_SWEEP_VALUES is empty" >&2
    exit 1
fi

for feature_id in "${FEATURES[@]}"; do
    if [[ ! "$feature_id" =~ ^[0-9]+$ ]] || (( feature_id >= 32768 )); then
        echo "ERROR: invalid layer27 SAE feature id: $feature_id" >&2
        exit 1
    fi
    run_feature "$feature_id"
done

echo "All requested 35B feature experiments completed: ${FEATURES[*]}"
