#!/usr/bin/env bash
# Add safety-blind reverse endpoints to the adjusted 2B/9B Step 5 experiment.
#
# The reverse branches mirror the final adjusted treatment recipes:
#   * 2B starts from the selected refusal anchor, uses 800 syco + 1000
#     instruction examples, and injects f28758 only on syco rows.
#   * 9B starts from the original Base model, uses 400 syco + 1000 instruction
#     examples, and injects f61718 on all training rows.
#
# Checkpoint selection reads only the canonical 400-row syco holdout and its
# repetition/uncertainty audit. Step 5 safety results are generated only after
# the reverse checkpoint has been frozen.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_step5_margin_recovery_common.sh"

# The release reverse definition is now endpoint recovery against syco_sft,
# rather than the older directional scan under the reduced-dose adjusted
# recipe.  Keep this historical file as a compatibility entry point.
exec "$SCRIPT_DIR/run_step5_full_recovery_reverse.sh" "$@"

MAX_ANOMALY="${MAX_ANOMALY:-4.5}"
GRID_JUDGE_WORKERS="${MR_GRID_JUDGE_WORKERS:-8}"

wait_batch() {
    local label="$1"
    shift
    (( $# == 0 )) || mr_wait_jobs "$label" "$@"
}

select_reverse() {
    local model_name="$1" ordinary_summary="$2" eval_root="$3" output="$4"
    shift 4
    local args=()
    local candidate
    for candidate in "$@"; do
        args+=(--candidate "$candidate")
    done

    # select-reverse writes a fallback choice before returning 2 when no
    # quality-qualified candidate exceeds ordinary SFT. Preserve that artifact
    # for an explicit fallback rather than consulting Step 5 safety.
    set +e
    "$PYTHON_BIN" src/step5_margin_recovery.py select-reverse \
        --model-name "$model_name" \
        --ordinary-summary "$ordinary_summary" \
        --eval-root "$eval_root" \
        "${args[@]}" \
        --max-anomaly "$MAX_ANOMALY" \
        --output "$output"
    local status=$?
    set -e
    if (( status != 0 && status != 2 )); then
        return "$status"
    fi
    "$PYTHON_BIN" - "$output" <<'PY'
import json
import sys

path = sys.argv[1]
data = json.load(open(path, encoding="utf-8"))
chosen = data.get("chosen")
if not isinstance(chosen, dict):
    raise SystemExit("reverse selection produced no chosen checkpoint")
data["selection_uses_step5_safety"] = False
data["report_label"] = "reverse"
if isinstance(data.get("selected"), dict):
    data["selection_status"] = "eligible_reverse_above_ordinary"
elif isinstance(data.get("fallback_selected"), dict):
    data["selection_status"] = "quality_qualified_fallback"
else:
    data["selection_status"] = "diagnostic_no_quality_qualified_candidate"
with open(path, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
PY
}

eval_selected_reverse() {
    local gpu="$1" model_name="$2" base_model="$3" artifact="$4"
    local safety_root="$5"
    local existing="$safety_root/$model_name/reverse"
    if [[ -s "$existing/summary.json" ]]; then
        local existing_artifact
        existing_artifact="$(mr_json_get "$existing/summary.json" artifact_dir)"
        if [[ "$existing_artifact" != "$artifact" ]]; then
            rm -rf "$existing"
        fi
    fi
    mr_eval_step5 "$gpu" "$model_name" "$base_model" "$artifact" reverse "$safety_root"
}

run_2b_reverse() {
    local model_name="Qwen3.5-2B-Base"
    local base_model="$MODEL_ROOT/$model_name"
    local root="$PROJECT_ROOT/outputs/step5_syco_safe/$model_name/margin_recovery"
    local data_dir="$root/data/syco_training_800"
    local split_dir="$data_dir/training_split"
    local train_root="$root/train/reverse_adjusted"
    local eval_root="$root/reverse_candidate_evaluations/$model_name"
    local safety_root="$root/safety_adjusted"
    local selection="$root/reverse_selection.json"
    local ordinary_summary="$PROJECT_ROOT/outputs/step5_syco_safe/$model_name/syco_evaluation/original/ordinary_sft/syco/summary.md"
    local canonical_eval="$PROJECT_ROOT/outputs/step4_feature_inject/dataset/$model_name/split_train2000_eval400_seed1234/syco_eval.jsonl"
    local anchor_selection="$root/anchor_selection.json"
    local anchor_artifact
    anchor_artifact="$(mr_json_get "$anchor_selection" selected.artifact)"

    IFS=' ' read -r -a names <<< "${REVERSE_2B_NAMES:-reverse_beta_neg0p25 reverse_beta_neg0p5 reverse_beta_neg0p75 reverse_beta_neg1 reverse_beta_neg1p5 reverse_beta_neg2 reverse_beta_neg3 reverse_beta_neg3p25 reverse_beta_neg3p4 reverse_beta_neg3p5 reverse_beta_neg3p6 reverse_beta_neg3p75 reverse_beta_neg4 reverse_beta_neg6 reverse_beta_neg8 reverse_beta_neg10}"
    IFS=' ' read -r -a betas <<< "${REVERSE_2B_BETAS:--0.25 -0.5 -0.75 -1 -1.5 -2 -3 -3.25 -3.4 -3.5 -3.6 -3.75 -4 -6 -8 -10}"
    if (( ${#names[@]} != ${#betas[@]} )); then
        echo "ERROR: REVERSE_2B_NAMES and REVERSE_2B_BETAS differ in length" >&2
        return 1
    fi
    mkdir -p "$train_root" "$eval_root" "$safety_root"

    echo "[adjusted reverse 2B 1/3] train matched reverse candidates from refusal anchor"
    local start index name out slot
    for (( start=0; start<${#names[@]}; start+=8 )); do
        local pids=()
        for slot in 0 1 2 3 4 5 6 7; do
            index=$((start + slot))
            (( index < ${#names[@]} )) || continue
            name="${names[$index]}"
            out="$train_root/$name"
            if mr_train_is_complete "$out"; then
                echo "[2B reverse train:skip] $name"
                continue
            fi
            (
                STEP4_LOG_DIR="$root/logs/train_reverse/$name" \
                bash "$SCRIPT_DIR/run_step4_syco_train.sh" \
                    --config "$root/step4_2b_top1.yaml" \
                    --model "$model_name" --model-path "$anchor_artifact" \
                    --group syco_sft_prevent \
                    --output-dir "$out" --run-name "$name" \
                    --shared-syco-dataset-path "$data_dir/syco_dataset.jsonl" \
                    --syco-split-dir "$split_dir" \
                    --syco-split-train-size 1800 --syco-split-eval-size 400 --syco-split-seed 1234 \
                    --injection-targets syco --beta "${betas[$index]}" --beta-schedule fixed \
                    --epochs 1 --learning-rate 2e-6 \
                    --batch-size 1 --global-batch-size 8 --warmup-ratio 0.03 \
                    --max-length 512 --full-model-export hf --tuning-mode full \
                    --gpu "$slot" --num-gpus 1 --no-deepspeed \
                    --no-gradient-checkpointing --overwrite --skip-completed
            ) &
            pids+=("$!")
        done
        wait_batch "adjusted 2B reverse training batch $start" "${pids[@]}"
    done

    echo "[adjusted reverse 2B 2/3] canonical syco evaluation and safety-blind selection"
    pids=()
    for index in "${!names[@]}"; do
        name="${names[$index]}"
        local eval_gpu=$((index % 8))
        (
            MR_JUDGE_WORKERS="$GRID_JUDGE_WORKERS"
            mr_eval_step4 "$eval_gpu" "$model_name" "$base_model" "$train_root/$name" \
                "$name" "$eval_root" "$canonical_eval" syco
        ) &
        pids+=("$!")
    done
    wait_batch "adjusted 2B reverse syco evaluation" "${pids[@]}"

    local candidates=()
    for name in "${names[@]}"; do
        candidates+=("$name=$train_root/$name")
    done
    select_reverse "$model_name" "$ordinary_summary" "$eval_root" "$selection" "${candidates[@]}"

    echo "[adjusted reverse 2B 3/3] Step 5 safety on frozen reverse"
    local artifact
    artifact="$(mr_json_get "$selection" chosen.artifact)"
    eval_selected_reverse 0 "$model_name" "$base_model" "$artifact" "$safety_root"
}

run_9b_reverse() {
    local model_name="Qwen3.5-9B-Base"
    local base_model="$MODEL_ROOT/$model_name"
    local root="$PROJECT_ROOT/outputs/step5_syco_safe/$model_name/margin_recovery"
    local data_dir="$root/data/syco_training_400"
    local split_dir="$data_dir/training_split"
    local train_root="$root/train/reverse_adjusted"
    local eval_root="$root/reverse_candidate_evaluations/$model_name"
    local safety_root="$root/safety"
    local selection="$root/reverse_selection.json"
    local ordinary_summary="$PROJECT_ROOT/outputs/step5_syco_safe/$model_name/syco_evaluation/original/ordinary_sft/syco/summary.md"
    local canonical_eval="$PROJECT_ROOT/outputs/step4_feature_inject/dataset/$model_name/split_train2000_eval400_seed1234/syco_eval.jsonl"

    IFS=' ' read -r -a names <<< "${REVERSE_9B_NAMES:-reverse_beta_neg10 reverse_beta_neg20 reverse_beta_neg40 reverse_beta_neg80 reverse_beta_neg120 reverse_beta_neg160}"
    IFS=' ' read -r -a betas <<< "${REVERSE_9B_BETAS:--10 -20 -40 -80 -120 -160}"
    if (( ${#names[@]} != ${#betas[@]} )); then
        echo "ERROR: REVERSE_9B_NAMES and REVERSE_9B_BETAS differ in length" >&2
        return 1
    fi
    mkdir -p "$train_root" "$eval_root" "$safety_root"

    echo "[adjusted reverse 9B 1/3] train matched reverse candidates"
    local start index name out slot
    for (( start=0; start<${#names[@]}; start+=2 )); do
        local pids=()
        for slot in 0 1; do
            index=$((start + slot))
            (( index < ${#names[@]} )) || continue
            name="${names[$index]}"
            out="$train_root/$name"
            if mr_train_is_complete "$out"; then
                echo "[9B reverse train:skip] $name"
                continue
            fi
            local gpu_group
            if (( slot == 0 )); then
                gpu_group="0,1,2,3"
            else
                gpu_group="4,5,6,7"
            fi
            (
                STEP4_LOG_DIR="$root/logs/train_reverse/$name" \
                bash "$SCRIPT_DIR/run_step4_syco_train.sh" \
                    --config "$root/step4_9b_top1.yaml" \
                    --model "$model_name" --model-path "$base_model" \
                    --group syco_sft_prevent \
                    --output-dir "$out" --run-name "$name" \
                    --shared-syco-dataset-path "$data_dir/syco_dataset.jsonl" \
                    --syco-split-dir "$split_dir" \
                    --syco-split-train-size 1400 --syco-split-eval-size 400 --syco-split-seed 1234 \
                    --injection-targets all --beta "${betas[$index]}" --beta-schedule fixed \
                    --epochs 1 --learning-rate 5e-7 \
                    --batch-size 1 --global-batch-size 8 --warmup-ratio 0.03 \
                    --max-length 512 --full-model-export hf --tuning-mode full \
                    --gpu "$gpu_group" --num-gpus 4 \
                    --gradient-checkpointing \
                    --deepspeed-config "$PROJECT_ROOT/configs/deepspeed_step4_zero2.json" \
                    --overwrite --skip-completed
            ) &
            pids+=("$!")
        done
        wait_batch "adjusted 9B reverse training batch $start" "${pids[@]}"
    done

    echo "[adjusted reverse 9B 2/3] canonical syco evaluation and safety-blind selection"
    local pids=()
    for index in "${!names[@]}"; do
        name="${names[$index]}"
        local eval_gpu=$((index % 8))
        (
            MR_JUDGE_WORKERS="$GRID_JUDGE_WORKERS"
            mr_eval_step4 "$eval_gpu" "$model_name" "$base_model" "$train_root/$name" \
                "$name" "$eval_root" "$canonical_eval" syco
        ) &
        pids+=("$!")
    done
    wait_batch "adjusted 9B reverse syco evaluation" "${pids[@]}"

    local candidates=()
    for name in "${names[@]}"; do
        candidates+=("$name=$train_root/$name")
    done
    select_reverse "$model_name" "$ordinary_summary" "$eval_root" "$selection" "${candidates[@]}"

    echo "[adjusted reverse 9B 3/3] Step 5 safety on frozen reverse"
    local artifact
    artifact="$(mr_json_get "$selection" chosen.artifact)"
    eval_selected_reverse 0 "$model_name" "$base_model" "$artifact" "$safety_root"
}

run_2b_reverse
run_9b_reverse

echo "[adjusted reverse release] canonical syco re-evaluation and final report"
"$PYTHON_BIN" src/step5_unified_syco_eval.py --group adjusted --judge-workers "$GRID_JUDGE_WORKERS"
"$PYTHON_BIN" src/step5_write_final_report.py
echo "Adjusted reverse endpoints complete."
