#!/usr/bin/env bash
# Clean Step 5 endpoint experiment.
#
# This is intentionally separate from the historical margin-recovery scripts.
# It keeps only the user-facing endpoint metrics: syco%, direct refusal%,
# pressure refusal%, single-turn pressure violation%, and the paired
# difference versus ordinary SFT.  Checkpoint selection is syco/quality blind
# to Step 5 safety results.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_step5_margin_recovery_common.sh"

MODEL_NAME="Qwen3.5-2B-Base"
BASE_MODEL="$MODEL_ROOT/$MODEL_NAME"
ROOT="$PROJECT_ROOT/outputs/step5_syco_safe/$MODEL_NAME/margin_recovery"
DATA_DIR="$ROOT/data/syco_training_800"
TRAINING_SPLIT="$DATA_DIR/training_split"
SYCO_TRAIN="$ROOT/train/syco_only"
SYCO_EVAL="$ROOT/candidate_evaluations/$MODEL_NAME"
SAFETY_ROOT="$ROOT/safety_adjusted"
OLD_DATA="$PROJECT_ROOT/outputs/step4_feature_inject/dataset/$MODEL_NAME"
OLD_SPLIT="$OLD_DATA/split_train2000_eval400_seed1234"
CANONICAL_SYCO_EVAL="$OLD_SPLIT/syco_eval.jsonl"
ANCHOR_SELECTION="$ROOT/anchor_selection.json"
UNIFIED_EVAL="$PROJECT_ROOT/outputs/step5_syco_safe/$MODEL_NAME/syco_evaluation/adjusted"
ANCHOR_SUMMARY="$UNIFIED_EVAL/refusal_anchor/syco/summary.md"
# ordinary_sft is the canonical Step 4 control.  It is not retrained in the
# margin-recovery follow-up, so all treatment selection and contrasts use this
# fixed reference.
ORDINARY_SUMMARY="$PROJECT_ROOT/outputs/step5_syco_safe/$MODEL_NAME/syco_evaluation/original/ordinary_sft/syco/summary.md"

mkdir -p "$SYCO_TRAIN" "$SYCO_EVAL" "$SAFETY_ROOT"

ANCHOR_ARTIFACT="$(mr_json_get "$ANCHOR_SELECTION" selected.artifact)"
ANCHOR_NAME="$(mr_json_get "$ANCHOR_SELECTION" selected.name)"

if [[ ! -s "$DATA_DIR/syco_dataset.jsonl" ]]; then
    "$PYTHON_BIN" src/step5_margin_recovery.py prepare-syco-training \
        --model-name "$MODEL_NAME" \
        --source "$OLD_DATA/syco_dataset.jsonl" \
        --original-split "$OLD_SPLIT" \
        --output-dir "$DATA_DIR" \
        --syco-train-count 800 --instruction-count 1000
fi
if [[ ! -s "$ROOT/step4_2b_top1.yaml" ]]; then
    "$PYTHON_BIN" src/step5_margin_recovery.py make-config \
        --model-name "$MODEL_NAME" --feature-ids 28758 \
        --output "$ROOT/step4_2b_top1.yaml"
fi

NAMES=(treatment_beta4 treatment_beta6 treatment_beta8 treatment_beta10)
BETAS=(4 6 8 10)

echo "[adjusted 2B 1/4] train syco-only candidates from the refusal anchor"
pids=()
for index in "${!NAMES[@]}"; do
    name="${NAMES[$index]}"
    out="$SYCO_TRAIN/$name"
    if mr_train_is_complete "$out"; then
        echo "[adjusted 2B train:skip] $name"
        continue
    fi
    (
        STEP4_LOG_DIR="$ROOT/logs/train_syco_only/$name" \
        bash "$SCRIPT_DIR/run_step4_syco_train.sh" \
            --config "$ROOT/step4_2b_top1.yaml" \
            --model "$MODEL_NAME" --model-path "$ANCHOR_ARTIFACT" \
            --group syco_sft_prevent \
            --output-dir "$out" --run-name "$name" \
            --shared-syco-dataset-path "$DATA_DIR/syco_dataset.jsonl" \
            --syco-split-dir "$TRAINING_SPLIT" \
            --syco-split-train-size 1800 --syco-split-eval-size 400 --syco-split-seed 1234 \
            --injection-targets syco --beta "${BETAS[$index]}" --beta-schedule fixed \
            --epochs 1 --learning-rate 2e-6 \
            --batch-size 1 --global-batch-size 8 --warmup-ratio 0.03 \
            --max-length 512 --full-model-export hf --tuning-mode full \
            --gpu "$index" --num-gpus 1 --no-deepspeed \
            --no-gradient-checkpointing --overwrite --skip-completed
    ) &
    pids+=("$!")
done
(( ${#pids[@]} == 0 )) || mr_wait_jobs "adjusted 2B syco-only training" "${pids[@]}"

echo "[adjusted 2B 2/4] evaluate syco holdout and select without safety"
pids=()
for index in "${!NAMES[@]}"; do
    name="${NAMES[$index]}"
    (
        MR_JUDGE_WORKERS="${MR_GRID_JUDGE_WORKERS:-4}"
        mr_eval_step4 "$index" "$MODEL_NAME" "$BASE_MODEL" "$SYCO_TRAIN/$name" \
            "$name" "$SYCO_EVAL" "$CANONICAL_SYCO_EVAL" syco
    ) &
    pids+=("$!")
done
mr_wait_jobs "adjusted 2B syco-only evaluation" "${pids[@]}"

args=()
for name in "${NAMES[@]}"; do
    args+=(--candidate "$name=$SYCO_TRAIN/$name")
done
"$PYTHON_BIN" src/step5_margin_recovery.py select \
    --model-name "$MODEL_NAME" \
    --base-summary "$ANCHOR_SUMMARY" \
    --ordinary-summary "$ORDINARY_SUMMARY" \
    --eval-root "$SYCO_EVAL" "${args[@]}" \
    --max-anomaly 4.5 --min-drift-pp 20 --allow-empty \
    --output "$ROOT/adjusted_treatment_selection.json"

# The adjusted endpoint requested here is the safety-blind checkpoint whose
# hook-off syco is closest to the refusal anchor.  Keep internal output-quality
# diagnostics in the selection JSON, but do not substitute a checkpoint whose
# syco misses the experimental target.
"$PYTHON_BIN" - "$ROOT/adjusted_treatment_selection.json" <<'PY'
import json, sys

path = sys.argv[1]
data = json.load(open(path, encoding="utf-8"))
base_syco = float(data["base"]["syco"])
chosen = min(
    data["candidates"],
    key=lambda row: (abs(float(row["syco"]) - base_syco), float(row["syco"]), row["name"]),
)
data["chosen"] = chosen
data["rule"] = "minimum absolute syco distance to the refusal anchor; safety-blind"
data["selection_uses_step5_safety"] = False
with open(path, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
PY

echo "[adjusted 2B 3/4] run standard Step 5 safety for selected treatment"
TREATMENT_NAME="$(mr_json_get "$ROOT/adjusted_treatment_selection.json" chosen.name)"
TREATMENT_ARTIFACT="$(mr_json_get "$ROOT/adjusted_treatment_selection.json" chosen.artifact)"
EXISTING_TREATMENT="$SAFETY_ROOT/$MODEL_NAME/treatment"
if [[ -s "$EXISTING_TREATMENT/summary.json" ]]; then
    EXISTING_ARTIFACT="$(mr_json_get "$EXISTING_TREATMENT/summary.json" artifact_dir)"
    if [[ "$EXISTING_ARTIFACT" != "$TREATMENT_ARTIFACT" ]]; then
        rm -rf "$EXISTING_TREATMENT"
    fi
fi
mr_eval_step5 0 "$MODEL_NAME" "$BASE_MODEL" "$TREATMENT_ARTIFACT" treatment "$SAFETY_ROOT"

echo "[adjusted 2B 4/4] write machine-readable selection metadata"
"$PYTHON_BIN" - "$ROOT/adjusted_treatment_selection.json" "$TREATMENT_NAME" <<'PY'
import json, sys
path, chosen_name = sys.argv[1:]
data = json.load(open(path, encoding="utf-8"))
data["report_label"] = "treatment"
data["chosen_name"] = chosen_name
data["selection_uses_step5_safety"] = False
with open(path, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
PY
echo "adjusted 2B treatment: $TREATMENT_NAME"
