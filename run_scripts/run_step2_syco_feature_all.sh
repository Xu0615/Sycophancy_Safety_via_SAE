#!/bin/bash
# ============================================================
# run_step2_syco_feature_all.sh — Extract paired SAE sycophancy
# features for all configured Qwen3.5 models with GPU assignment.
#
# Usage:
#   bash run_scripts/run_step2_syco_feature_all.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_scripts/run_step2_syco_feature_all.sh
#   bash run_scripts/run_step2_syco_feature_all.sh --overwrite
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FEATURE_SCRIPT="$SCRIPT_DIR/run_step2_syco_feature.sh"
NUM_GPUS=8
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
LOG_DIR="$PROJECT_ROOT/logs/step2"
PROGRESS_DIR="$LOG_DIR/progress"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-15}"
PROGRESS_STALL_INTERVAL="${PROGRESS_STALL_INTERVAL:-300}"
OVERWRITE_REQUESTED=false

MODELS=(
    "Qwen3.5-2B-Base"
    "Qwen3.5-9B-Base"
    "Qwen3.5-35B-A3B-Base"
)

for arg in "$@"; do
    case "$arg" in
        --model-name|--model_name|--model|--model-path|--model_path)
            echo "ERROR: run_step2_syco_feature_all.sh sets model/model-path automatically."
            exit 1
            ;;
    esac
done

ARGS=("$@")
IDX=0
while [[ "$IDX" -lt "${#ARGS[@]}" ]]; do
    case "${ARGS[$IDX]}" in
        --overwrite)
            OVERWRITE_REQUESTED=true
            ;;
    esac
    IDX=$((IDX + 1))
done

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra ALL_GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
    ALL_GPUS=()
    for ((i=0; i<NUM_GPUS; i++)); do
        ALL_GPUS+=("$i")
    done
fi

if [[ ${#ALL_GPUS[@]} -lt 8 ]]; then
    echo "ERROR: run_step2_syco_feature_all.sh requires 8 GPUs for the default layout."
    echo "       CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    exit 1
fi

join_by_comma() {
    local IFS=","
    echo "$*"
}

count_lines() {
    local FILE_PATH="$1"
    if [[ -f "$FILE_PATH" ]]; then
        wc -l < "$FILE_PATH"
    else
        echo 0
    fi
}

cap_count() {
    local VALUE="$1"
    local MAX_VALUE="$2"
    if [[ "$VALUE" -gt "$MAX_VALUE" ]]; then
        echo "$MAX_VALUE"
    else
        echo "$VALUE"
    fi
}

render_bar() {
    local DONE="$1"
    local TOTAL="$2"
    local WIDTH=28
    local PCT=0
    if [[ "$TOTAL" -gt 0 ]]; then
        PCT=$((100 * DONE / TOTAL))
    fi
    local FILLED=$((WIDTH * PCT / 100))
    local EMPTY=$((WIDTH - FILLED))
    local BAR_FILLED BAR_EMPTY
    BAR_FILLED="$(printf "%${FILLED}s" "" | tr ' ' '#')"
    BAR_EMPTY="$(printf "%${EMPTY}s" "" | tr ' ' '-')"
    printf "[%s%s] %6d/%-6d %3d%%" "$BAR_FILLED" "$BAR_EMPTY" "$DONE" "$TOTAL" "$PCT"
}

feature_target_pairs_for_model() {
    local MODEL_NAME="$1"
    local DATASET_PATH="$PROJECT_ROOT/outputs/step2/$MODEL_NAME/syco_dataset/syco_dataset.jsonl"
    local ROWS
    ROWS="$(count_lines "$DATASET_PATH")"
    echo $((ROWS / 2))
}

feature_processed_pairs_for_model() {
    local MODEL_NAME="$1"
    local CKPT_PATH="$PROJECT_ROOT/outputs/step2/$MODEL_NAME/syco_feature/checkpoint.pt"
    local LOG_FILE="$LOG_DIR/syco_feature_${MODEL_NAME}.log"
    local LOG_DONE=0

    if [[ -f "$LOG_FILE" ]]; then
        LOG_DONE="$(
            grep -o "Checkpoint saved at [0-9][0-9]* pairs" "$LOG_FILE" 2>/dev/null \
                | awk '{print $4}' \
                | tail -n 1
        )"
        LOG_DONE="${LOG_DONE:-0}"
    fi

    if [[ ! -f "$CKPT_PATH" ]]; then
        echo "$LOG_DONE"
        return
    fi

    local CKPT_DONE
    CKPT_DONE="$("$PYTHON_BIN" - "$CKPT_PATH" <<'PY' 2>/dev/null || true
import sys
import torch

path = sys.argv[1]
ckpt = torch.load(path, map_location="cpu", weights_only=False)
print(int(ckpt.get("processed_pairs", 0)))
PY
)"
    CKPT_DONE="${CKPT_DONE:-0}"
    if [[ "$LOG_DONE" -gt "$CKPT_DONE" ]]; then
        echo "$LOG_DONE"
    else
        echo "$CKPT_DONE"
    fi
}

feature_counts_for_model() {
    local MODEL_NAME="$1"
    local TARGET
    TARGET="$(feature_target_pairs_for_model "$MODEL_NAME")"

    local DONE=0
    if [[ -f "$PROJECT_ROOT/outputs/step2/$MODEL_NAME/syco_feature/top_features.json" ]]; then
        DONE="$TARGET"
    else
        DONE="$(feature_processed_pairs_for_model "$MODEL_NAME")"
        DONE="$(cap_count "$DONE" "$TARGET")"
    fi

    echo "$DONE $TARGET"
}

progress_monitor() {
    local STOP_FILE="$1"
    local GROUP_NAME="$2"
    shift 2
    local WATCH_MODELS=("$@")
    local LAST_STATE=""
    local PRINTED_ONCE=false
    local LAST_PRINT_TS=0
    local NOW STATE ROW
    local ROWS=()

    while [[ ! -f "$STOP_FILE" ]]; do
        STATE=""
        ROWS=()
        for MODEL_NAME in "${WATCH_MODELS[@]}"; do
            read -r DONE TOTAL <<< "$(feature_counts_for_model "$MODEL_NAME")"
            STATE+="${MODEL_NAME}:${DONE}/${TOTAL};"
            ROWS+=("$MODEL_NAME $DONE $TOTAL")
        done

        NOW="$(date +%s)"
        if [[ "$PRINTED_ONCE" == false || "$STATE" != "$LAST_STATE" ]]; then
            echo ""
            echo "  [run_step2_syco_feature_all] feature progress ${GROUP_NAME} $(date '+%H:%M:%S')"
            for ROW in "${ROWS[@]}"; do
                read -r MODEL_NAME DONE TOTAL <<< "$ROW"
                printf "    %-24s " "$MODEL_NAME"
                render_bar "$DONE" "$TOTAL"
                if [[ "$TOTAL" -eq 0 ]]; then
                    printf "  waiting for syco_dataset.jsonl"
                elif [[ -f "$PROJECT_ROOT/outputs/step2/$MODEL_NAME/syco_feature/top_features.json" ]]; then
                    printf "  outputs ready"
                else
                    printf "  scoring pairs"
                fi
                echo ""
            done
            PRINTED_ONCE=true
            LAST_STATE="$STATE"
            LAST_PRINT_TS="$NOW"
        elif (( NOW - LAST_PRINT_TS >= PROGRESS_STALL_INTERVAL )); then
            echo ""
            echo "  [run_step2_syco_feature_all] feature still running ${GROUP_NAME} $(date '+%H:%M:%S') - no checkpoint progress yet"
            echo "  [run_step2_syco_feature_all] logs: $LOG_DIR/*_syco_feature_all.log and $LOG_DIR/syco_feature_*.log"
            LAST_PRINT_TS="$NOW"
        fi

        sleep "$PROGRESS_INTERVAL"
    done

    echo ""
    echo "  [run_step2_syco_feature_all] feature progress ${GROUP_NAME} final"
    for MODEL_NAME in "${WATCH_MODELS[@]}"; do
        read -r DONE TOTAL <<< "$(feature_counts_for_model "$MODEL_NAME")"
        printf "    %-24s " "$MODEL_NAME"
        render_bar "$DONE" "$TOTAL"
        if [[ "$TOTAL" -eq 0 ]]; then
            printf "  waiting for syco_dataset.jsonl"
        elif [[ -f "$PROJECT_ROOT/outputs/step2/$MODEL_NAME/syco_feature/top_features.json" ]]; then
            printf "  outputs ready"
        else
            printf "  scoring pairs"
        fi
        echo ""
    done
}

prepare_feature_outputs_for_overwrite() {
    if [[ "$OVERWRITE_REQUESTED" != true ]]; then
        return
    fi

    local MODEL_NAME BASE_DIR
    for MODEL_NAME in "$@"; do
        BASE_DIR="$PROJECT_ROOT/outputs/step2/$MODEL_NAME/syco_feature"
        rm -f \
            "$BASE_DIR/checkpoint.pt" \
            "$BASE_DIR/feature_stats.pt" \
            "$BASE_DIR/features.jsonl" \
            "$BASE_DIR/top_features.json" \
            "$BASE_DIR/summary.json" \
            "$BASE_DIR/summary.md" \
            "$LOG_DIR/syco_feature_${MODEL_NAME}.log" \
            "$LOG_DIR/${MODEL_NAME}_syco_feature_all.log"
    done
}

run_feature_group() {
    local GROUP_NAME="$1"
    shift
    local SPECS=("$@")
    local PIDS=()
    local WATCH_MODELS=()

    echo ""
    echo "================================================================"
    echo "  [run_step2_syco_feature_all] feature group: $GROUP_NAME"
    echo "================================================================"

    local SPEC MODEL_NAME GPU_LIST LOG_FILE
    for SPEC in "${SPECS[@]}"; do
        IFS=':' read -r MODEL_NAME GPU_LIST <<< "$SPEC"
        LOG_FILE="$LOG_DIR/${MODEL_NAME}_syco_feature_all.log"
        WATCH_MODELS+=("$MODEL_NAME")
        echo "  [run_step2_syco_feature_all] $MODEL_NAME on GPU[$GPU_LIST]"
        CUDA_VISIBLE_DEVICES="$GPU_LIST" bash "$FEATURE_SCRIPT" --model-name "$MODEL_NAME" "${ARGS[@]}" \
            > "$LOG_FILE" 2>&1 &
        PIDS+=($!)
    done

    local STOP_FILE="$PROGRESS_DIR/feature_${GROUP_NAME//[^A-Za-z0-9_]/_}.stop"
    rm -f "$STOP_FILE"
    progress_monitor "$STOP_FILE" "$GROUP_NAME" "${WATCH_MODELS[@]}" &
    local MONITOR_PID=$!

    local FAILED=0
    for PID in "${PIDS[@]}"; do
        if ! wait "$PID"; then
            FAILED=1
        fi
    done
    touch "$STOP_FILE"
    wait "$MONITOR_PID" || true

    if [[ "$FAILED" -eq 1 ]]; then
        echo "ERROR: one or more Step 2 syco feature runs failed. See $LOG_DIR/*_syco_feature_all.log"
        exit 1
    fi
}

GPU_2B="$(join_by_comma "${ALL_GPUS[@]:0:1}")"
GPU_9B="$(join_by_comma "${ALL_GPUS[@]:1:3}")"
GPU_35B_A3B="$(join_by_comma "${ALL_GPUS[@]:4:4}")"

echo ""
echo "================================================================"
echo "  [run_step2_syco_feature_all] Extracting syco SAE features"
echo "  [run_step2_syco_feature_all] layout: 2B=GPU[$GPU_2B], 9B=GPU[$GPU_9B], 35B-A3B=GPU[$GPU_35B_A3B]"
echo "  [run_step2_syco_feature_all] progress interval: ${PROGRESS_INTERVAL}s"
echo "================================================================"

mkdir -p "$LOG_DIR" "$PROGRESS_DIR"
prepare_feature_outputs_for_overwrite "${MODELS[@]}"

run_feature_group "all_models" \
    "Qwen3.5-2B-Base:$GPU_2B" \
    "Qwen3.5-9B-Base:$GPU_9B" \
    "Qwen3.5-35B-A3B-Base:$GPU_35B_A3B"

echo ""
echo "All Step 2 syco feature runs completed."
