#!/usr/bin/env bash
# Select 2B/9B reverse-alpha endpoints that recover their syco_sft behavior.
#
# Selection is safety-blind.  It reads only the canonical 400-row sycophancy
# holdout and requires:
#   * repetitive% + uncertain% <= MAX_ANOMALY (default 4.5)
#   * |reverse syco% - syco_sft syco%| <= MAX_RECOVERY_GAP_PP (default 2.0)
#
# The full-dose Step 4 alpha checkpoints already share the corresponding
# syco_sft data, Base start, epoch count, learning rate, and injection scope.
# This script reuses their deterministic model outputs, applies the current
# canonical judge, selects the closest endpoint, runs Step 5 safety, and
# regenerates the final cross-model report.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_step5_margin_recovery_common.sh"

MAX_ANOMALY="${MAX_ANOMALY:-4.5}"
MAX_RECOVERY_GAP_PP="${MAX_RECOVERY_GAP_PP:-2.0}"
GRID_JUDGE_WORKERS="${MR_GRID_JUDGE_WORKERS:-8}"
SAFETY_2B_GPU="${FULL_RECOVERY_2B_GPU:-0}"
SAFETY_9B_GPU="${FULL_RECOVERY_9B_GPU:-0}"

materialize_candidate() {
    local source_syco="$1" destination_syco="$2"
    mkdir -p "$destination_syco"
    "$PYTHON_BIN" - "$source_syco" "$destination_syco" <<'PY'
import hashlib
import shutil
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
source_outputs = source / "model_outputs.parquet"
target_outputs = destination / "model_outputs.parquet"
if not source_outputs.is_file():
    raise FileNotFoundError(source_outputs)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

changed = (
    not target_outputs.is_file()
    or sha256(source_outputs) != sha256(target_outputs)
)
if changed:
    shutil.copy2(source_outputs, target_outputs)
    for name in ("judge_checkpoint.jsonl", "judge_results.parquet", "summary.md"):
        (destination / name).unlink(missing_ok=True)
PY
}

judge_candidates() {
    local model_name="$1" source_eval_root="$2" eval_root="$3"
    shift 3
    local pids=()
    local spec label source_run source_syco destination_syco
    mkdir -p "$eval_root"
    for spec in "$@"; do
        label="${spec%%=*}"
        source_run="${spec#*=}"
        source_syco="$source_eval_root/$source_run/syco"
        destination_syco="$eval_root/$label/syco"
        materialize_candidate "$source_syco" "$destination_syco"
        if [[ -s "$destination_syco/summary.md" && \
              -s "$destination_syco/judge_results.parquet" ]]; then
            echo "[full-recovery judge:skip] $model_name/$label"
            continue
        fi
        echo "[full-recovery judge:start] $model_name/$label"
        (
            "$PYTHON_BIN" src/step5_margin_recovery.py syco-judge \
                --eval-dir "$destination_syco" \
                --workers "$GRID_JUDGE_WORKERS" \
                --max-retries 10 \
                --timeout 180
        ) >"$eval_root/$label/judge.log" 2>&1 &
        pids+=("$!")
    done
    (( ${#pids[@]} == 0 )) || mr_wait_jobs "$model_name full-recovery judge" "${pids[@]}"
}

select_full_recovery() {
    local model_name="$1" syco_sft_summary="$2" eval_root="$3"
    local train_root="$4" output="$5"
    shift 5
    local args=()
    local spec label source_run
    for spec in "$@"; do
        label="${spec%%=*}"
        source_run="${spec#*=}"
        args+=(--candidate "$label=$train_root/$source_run")
    done
    "$PYTHON_BIN" src/step5_margin_recovery.py select-full-recovery-reverse \
        --model-name "$model_name" \
        --syco-sft-summary "$syco_sft_summary" \
        --eval-root "$eval_root" \
        "${args[@]}" \
        --max-anomaly "$MAX_ANOMALY" \
        --max-recovery-gap-pp "$MAX_RECOVERY_GAP_PP" \
        --output "$output"
}

eval_selected() {
    local gpu="$1" model_name="$2" base_model="$3"
    local selection="$4" safety_root="$5"
    local artifact
    artifact="$(mr_json_get "$selection" chosen.artifact)"
    local existing="$safety_root/$model_name/reverse"
    "$PYTHON_BIN" - "$existing" "$artifact" <<'PY'
import json
import shutil
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
artifact = sys.argv[2]
summary = run_dir / "summary.json"
if summary.is_file():
    existing = json.loads(summary.read_text(encoding="utf-8")).get("artifact_dir")
    if existing != artifact:
        shutil.rmtree(run_dir)
PY
    mr_eval_step5 "$gpu" "$model_name" "$base_model" "$artifact" reverse "$safety_root"
}

run_model() {
    local model_name="$1" source_experiment="$2" safety_gpu="$3"
    shift 3
    local base_model="$MODEL_ROOT/$model_name"
    local step4_train="$PROJECT_ROOT/outputs/step4_feature_inject/train/$model_name/$source_experiment"
    local step4_eval="$PROJECT_ROOT/outputs/step4_feature_inject/eval/$model_name/$source_experiment"
    local output_root="$PROJECT_ROOT/outputs/step5_syco_safe/$model_name"
    local eval_root="$output_root/full_recovery_reverse_candidate_evaluations"
    local selection="$output_root/full_recovery_reverse_selection.json"
    local safety_root="$output_root/full_recovery_safety"
    local syco_sft_summary="$output_root/syco_evaluation/original/ordinary_sft/syco/summary.md"

    judge_candidates "$model_name" "$step4_eval" "$eval_root" "$@"
    select_full_recovery \
        "$model_name" "$syco_sft_summary" "$eval_root" "$step4_train" \
        "$selection" "$@"
    eval_selected "$safety_gpu" "$model_name" "$base_model" "$selection" "$safety_root"
}

run_model \
    "Qwen3.5-2B-Base" \
    "gate_f28758_train2000_eval400_syco1000_alpaca1000_seed1234_lr2e-6_ep2" \
    "$SAFETY_2B_GPU" \
    "reverse_alpha_neg0p5=syco_sft_prevent_f28758_alpha_neg0p5_full_lr2e-6_ep2_gbs8" \
    "reverse_alpha_neg1=syco_sft_prevent_f28758_alpha_neg1_full_lr2e-6_ep2_gbs8" \
    "reverse_alpha_neg2=syco_sft_prevent_f28758_alpha_neg2_full_lr2e-6_ep2_gbs8" \
    "reverse_alpha_neg3=syco_sft_prevent_f28758_alpha_neg3_full_lr2e-6_ep2_gbs8"

run_model \
    "Qwen3.5-9B-Base" \
    "gate_f61718_train2000_eval400_syco1000_alpaca1000_seed1234_lr5e-7_ep1" \
    "$SAFETY_9B_GPU" \
    "reverse_alpha_neg1=syco_sft_prevent_f61718_alpha_neg1_full_lr5e-7_ep1_gbs8" \
    "reverse_alpha_neg3=syco_sft_prevent_f61718_alpha_neg3_full_lr5e-7_ep1_gbs8" \
    "reverse_alpha_neg5=syco_sft_prevent_f61718_alpha_neg5_full_lr5e-7_ep1_gbs8" \
    "reverse_alpha_neg7=syco_sft_prevent_f61718_alpha_neg7_full_lr5e-7_ep1_gbs8"

echo "[full-recovery release] canonical syco evaluation and report"
"$PYTHON_BIN" src/step5_unified_syco_eval.py \
    --group adjusted --judge-workers "$GRID_JUDGE_WORKERS"
"$PYTHON_BIN" src/step5_write_final_report.py
echo "Full-recovery reverse endpoints complete."
