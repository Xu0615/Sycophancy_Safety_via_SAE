#!/bin/bash
# ============================================================
# run_step2_syco_dataset_all.sh - Run Step 2 sycophancy dataset
# generation for all configured Qwen3.5 models with GPU assignment.
#
# Usage:
#   bash run_scripts/run_step2_syco_dataset_all.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_scripts/run_step2_syco_dataset_all.sh
#   bash run_scripts/run_step2_syco_dataset_all.sh --overwrite --per-domain 500
#
# Notes:
#   - This wraps run_step2_syco_dataset.sh and keeps the same pipeline/default args.
#   - Each model run writes queries.jsonl, syco_dataset.jsonl, and
#     syco_dataset_eval.jsonl under outputs/step2/<model>/syco_dataset/.
#   - Extra args are forwarded to each run_step2_syco_dataset.sh invocation.
#   - Do not pass --model_name/--model_path here; this script sets them.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STEP2_SCRIPT="$SCRIPT_DIR/run_step2_syco_dataset.sh"
NUM_GPUS=8
LOG_DIR="$PROJECT_ROOT/logs/step2"
PROGRESS_DIR="$LOG_DIR/progress"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-15}"
PROGRESS_STALL_INTERVAL="${PROGRESS_STALL_INTERVAL:-300}"
DOMAIN_COUNT="${STEP2_SYCO_DOMAIN_COUNT:-7}"
PER_DOMAIN_FOR_PROGRESS=100
EVAL_PER_DOMAIN_FOR_PROGRESS=20
EVAL_SYCOPHANCY_PER_DOMAIN_FOR_PROGRESS=""
OVERWRITE_REQUESTED=false

MODELS=(
    "Qwen3.5-2B-Base"
    "Qwen3.5-9B-Base"
    "Qwen3.5-35B-A3B-Base"
)

for arg in "$@"; do
    case "$arg" in
        --model_name|--model-name|--model_path|--model-path)
            echo "ERROR: run_step2_syco_dataset_all.sh sets model_name/model_path automatically."
            echo "       Remove $arg and pass only shared step2 options."
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
        --per-domain)
            IDX=$((IDX + 1))
            if [[ "$IDX" -lt "${#ARGS[@]}" ]]; then
                PER_DOMAIN_FOR_PROGRESS="${ARGS[$IDX]}"
            fi
            ;;
        --per-domain=*)
            PER_DOMAIN_FOR_PROGRESS="${ARGS[$IDX]#--per-domain=}"
            ;;
        --eval-per-domain)
            IDX=$((IDX + 1))
            if [[ "$IDX" -lt "${#ARGS[@]}" ]]; then
                EVAL_PER_DOMAIN_FOR_PROGRESS="${ARGS[$IDX]}"
            fi
            ;;
        --eval-per-domain=*)
            EVAL_PER_DOMAIN_FOR_PROGRESS="${ARGS[$IDX]#--eval-per-domain=}"
            ;;
        --eval-sycophancy-per-domain)
            IDX=$((IDX + 1))
            if [[ "$IDX" -lt "${#ARGS[@]}" ]]; then
                EVAL_SYCOPHANCY_PER_DOMAIN_FOR_PROGRESS="${ARGS[$IDX]}"
            fi
            ;;
        --eval-sycophancy-per-domain=*)
            EVAL_SYCOPHANCY_PER_DOMAIN_FOR_PROGRESS="${ARGS[$IDX]#--eval-sycophancy-per-domain=}"
            ;;
    esac
    IDX=$((IDX + 1))
done

if [[ ! -x "$STEP2_SCRIPT" ]]; then
    echo "ERROR: Step 2 script is not executable or not found: $STEP2_SCRIPT"
    exit 1
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra ALL_GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
    ALL_GPUS=()
    for ((i=0; i<NUM_GPUS; i++)); do
        ALL_GPUS+=("$i")
    done
fi

if [[ ${#ALL_GPUS[@]} -lt 8 ]]; then
    echo "ERROR: run_step2_syco_dataset_all.sh requires 8 GPUs for the default layout."
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

dataset_counts_for_model() {
    local MODEL_NAME="$1"
    local BASE_DIR="$PROJECT_ROOT/outputs/step2/$MODEL_NAME/syco_dataset"
    local GENERATED_PER_DOMAIN=$((PER_DOMAIN_FOR_PROGRESS + EVAL_PER_DOMAIN_FOR_PROGRESS))
    local QUERIES_TOTAL=$((GENERATED_PER_DOMAIN * DOMAIN_COUNT))
    local RAW_TOTAL=$((QUERIES_TOTAL * 2))
    local DATASET_TOTAL=$((PER_DOMAIN_FOR_PROGRESS * DOMAIN_COUNT * 2))
    local EVAL_TOTAL=$((EVAL_PER_DOMAIN_FOR_PROGRESS * DOMAIN_COUNT))

    local Q_DONE RAW_DONE D_DONE E_DONE
    Q_DONE="$(cap_count "$(count_lines "$BASE_DIR/queries.jsonl")" "$QUERIES_TOTAL")"
    RAW_DONE="$(cap_count "$(count_lines "$BASE_DIR/syco_dataset_all.jsonl")" "$RAW_TOTAL")"
    D_DONE="$(cap_count "$(count_lines "$BASE_DIR/syco_dataset.jsonl")" "$DATASET_TOTAL")"
    E_DONE="$(cap_count "$(count_lines "$BASE_DIR/syco_dataset_eval.jsonl")" "$EVAL_TOTAL")"

    local DONE=$((Q_DONE + RAW_DONE + D_DONE + E_DONE))
    local TOTAL=$((QUERIES_TOTAL + RAW_TOTAL + DATASET_TOTAL + EVAL_TOTAL))
    echo "$DONE $TOTAL $Q_DONE $QUERIES_TOTAL $RAW_DONE $RAW_TOTAL $D_DONE $DATASET_TOTAL $E_DONE $EVAL_TOTAL"
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
            read -r DONE TOTAL Q_DONE Q_TOTAL RAW_DONE RAW_TOTAL D_DONE D_TOTAL E_DONE E_TOTAL <<< "$(dataset_counts_for_model "$MODEL_NAME")"
            STATE+="${MODEL_NAME}:${DONE}/${TOTAL};"
            ROWS+=("$MODEL_NAME $DONE $TOTAL $Q_DONE $Q_TOTAL $RAW_DONE $RAW_TOTAL $D_DONE $D_TOTAL $E_DONE $E_TOTAL")
        done

        NOW="$(date +%s)"
        if [[ "$PRINTED_ONCE" == false || "$STATE" != "$LAST_STATE" ]]; then
            echo ""
            echo "  [step2 all] dataset progress ${GROUP_NAME} $(date '+%H:%M:%S')"
            for ROW in "${ROWS[@]}"; do
                read -r MODEL_NAME DONE TOTAL Q_DONE Q_TOTAL RAW_DONE RAW_TOTAL D_DONE D_TOTAL E_DONE E_TOTAL <<< "$ROW"
                printf "    %-24s " "$MODEL_NAME"
                render_bar "$DONE" "$TOTAL"
                printf "  q=%s/%s raw=%s/%s feature=%s/%s eval=%s/%s\n" "$Q_DONE" "$Q_TOTAL" "$RAW_DONE" "$RAW_TOTAL" "$D_DONE" "$D_TOTAL" "$E_DONE" "$E_TOTAL"
            done
            PRINTED_ONCE=true
            LAST_STATE="$STATE"
            LAST_PRINT_TS="$NOW"
        elif (( NOW - LAST_PRINT_TS >= PROGRESS_STALL_INTERVAL )); then
            echo ""
            echo "  [step2 all] dataset still running ${GROUP_NAME} $(date '+%H:%M:%S') - no new output rows yet"
            echo "  [step2 all] logs: $LOG_DIR/*_syco_dataset_all.log"
            LAST_PRINT_TS="$NOW"
        fi

        sleep "$PROGRESS_INTERVAL"
    done

    echo ""
    echo "  [step2 all] dataset progress ${GROUP_NAME} final"
    for MODEL_NAME in "${WATCH_MODELS[@]}"; do
        read -r DONE TOTAL Q_DONE Q_TOTAL RAW_DONE RAW_TOTAL D_DONE D_TOTAL E_DONE E_TOTAL <<< "$(dataset_counts_for_model "$MODEL_NAME")"
        printf "    %-24s " "$MODEL_NAME"
        render_bar "$DONE" "$TOTAL"
        printf "  q=%s/%s raw=%s/%s feature=%s/%s eval=%s/%s\n" "$Q_DONE" "$Q_TOTAL" "$RAW_DONE" "$RAW_TOTAL" "$D_DONE" "$D_TOTAL" "$E_DONE" "$E_TOTAL"
    done
}

prepare_dataset_outputs_for_overwrite() {
    if [[ "$OVERWRITE_REQUESTED" != true ]]; then
        return
    fi

    local MODEL_NAME BASE_DIR
    for MODEL_NAME in "$@"; do
        BASE_DIR="$PROJECT_ROOT/outputs/step2/$MODEL_NAME/syco_dataset"
        rm -f \
            "$BASE_DIR/queries.jsonl" \
            "$BASE_DIR/syco_dataset_all.jsonl" \
            "$BASE_DIR/syco_dataset.jsonl" \
            "$BASE_DIR/syco_dataset_eval.jsonl" \
            "$LOG_DIR/step2_syco_dataset_${MODEL_NAME}.log" \
            "$LOG_DIR/${MODEL_NAME}_syco_dataset_all.log"
    done
}

max_num_batched_tokens_for_tp() {
    local TP_SIZE="$1"
    case "$TP_SIZE" in
        3)
            echo 6144
            ;;
        *)
            echo ""
            ;;
    esac
}

run_dataset_group() {
    local GROUP_NAME="$1"
    shift
    local SPECS=("$@")
    local PIDS=()
    local WATCH_MODELS=()

    echo ""
    echo "================================================================"
    echo "  [step2 all] dataset group: $GROUP_NAME"
    echo "================================================================"

    local SPEC MODEL_NAME GPU_LIST TP_SIZE LOG_FILE
    for SPEC in "${SPECS[@]}"; do
        IFS=':' read -r MODEL_NAME GPU_LIST TP_SIZE <<< "$SPEC"
        LOG_FILE="$LOG_DIR/${MODEL_NAME}_syco_dataset_all.log"
        WATCH_MODELS+=("$MODEL_NAME")
        local VLLM_EXTRA_ARGS=()
        local MAX_NUM_BATCHED_TOKENS
        MAX_NUM_BATCHED_TOKENS="$(max_num_batched_tokens_for_tp "$TP_SIZE")"
        if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then
            VLLM_EXTRA_ARGS+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
        fi

        echo "  [step2 all] $MODEL_NAME on GPU[$GPU_LIST] tensor_parallel_size=$TP_SIZE max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-config}"
        CUDA_VISIBLE_DEVICES="$GPU_LIST" bash "$STEP2_SCRIPT" --model_name "$MODEL_NAME" "${ARGS[@]}" \
            --tensor-parallel-size "$TP_SIZE" \
            "${VLLM_EXTRA_ARGS[@]}" \
            > "$LOG_FILE" 2>&1 &
        PIDS+=($!)
    done

    local STOP_FILE="$PROGRESS_DIR/dataset_${GROUP_NAME//[^A-Za-z0-9_]/_}.stop"
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
        echo "ERROR: one or more Step 2 syco dataset runs failed. See $LOG_DIR/*_syco_dataset_all.log"
        exit 1
    fi
}

GPU_2B="$(join_by_comma "${ALL_GPUS[@]:0:1}")"
GPU_9B="$(join_by_comma "${ALL_GPUS[@]:1:3}")"
GPU_35B_A3B="$(join_by_comma "${ALL_GPUS[@]:4:4}")"

echo ""
echo "================================================================"
echo "  [step2 all] running models:"
for model in "${MODELS[@]}"; do
echo "    - $model"
done
echo "  [step2 all] layout: 2B=GPU[$GPU_2B], 9B=GPU[$GPU_9B], 35B-A3B=GPU[$GPU_35B_A3B]"
echo "  [step2 all] shared args: $*"
echo "  [step2 all] progress: feature_per_domain=$PER_DOMAIN_FOR_PROGRESS, eval_per_domain=$EVAL_PER_DOMAIN_FOR_PROGRESS, domains=$DOMAIN_COUNT, interval=${PROGRESS_INTERVAL}s"
echo "================================================================"

mkdir -p "$LOG_DIR" "$PROGRESS_DIR"
prepare_dataset_outputs_for_overwrite "${MODELS[@]}"

run_dataset_group "all_models" \
    "Qwen3.5-2B-Base:$GPU_2B:1" \
    "Qwen3.5-9B-Base:$GPU_9B:3" \
    "Qwen3.5-35B-A3B-Base:$GPU_35B_A3B:4"

echo ""
echo "All Step 2 runs completed."
