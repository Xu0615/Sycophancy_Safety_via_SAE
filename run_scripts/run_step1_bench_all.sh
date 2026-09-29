#!/bin/bash
# ============================================================
# run_step1_bench_all.sh — Run Step 1 benchmark evaluation for
# all configured Qwen3.5 models with GPU assignment.
#
# Default 8-GPU layout:
#   Qwen3.5-2B-Base  -> GPU 0, TP=1
#   Qwen3.5-9B-Base  -> GPU 1,2,3, TP=3
#   Qwen3.5-35B-A3B-Base -> GPU 4,5,6,7, TP=4
#
# Usage:
#   bash run_scripts/run_step1_bench_all.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_scripts/run_step1_bench_all.sh
#   bash run_scripts/run_step1_bench_all.sh --dry-run
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

STEP1_SCRIPT="$SCRIPT_DIR/run_step1_bench.sh"
NUM_GPUS=8
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model|--all-models|--tensor-parallel-size)
            echo "ERROR: run_step1_bench_all.sh sets model and tensor parallel automatically."
            echo "       Remove $1 and pass only shared Step 1 options."
            exit 1
            ;;
        --gpu)
            NUM_GPUS="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    IFS=',' read -ra ALL_GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
    ALL_GPUS=()
    for ((i=0; i<NUM_GPUS; i++)); do
        ALL_GPUS+=("$i")
    done
fi

if [[ ${#ALL_GPUS[@]} -lt 8 ]]; then
    echo "ERROR: run_step1_bench_all.sh requires 8 GPUs for the default layout."
    echo "       CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    exit 1
fi

join_by_comma() {
    local IFS=","
    echo "$*"
}

GPU_2B="$(join_by_comma "${ALL_GPUS[@]:0:1}")"
GPU_9B="$(join_by_comma "${ALL_GPUS[@]:1:3}")"
GPU_35B_A3B="$(join_by_comma "${ALL_GPUS[@]:4:4}")"

echo ""
echo "================================================================"
echo "  [run_step1_bench_all] Step 1 benchmark all-model run"
echo "  [run_step1_bench_all] layout: 2B=GPU[$GPU_2B], 9B=GPU[$GPU_9B], 35B-A3B=GPU[$GPU_35B_A3B]"
echo "================================================================"

mkdir -p logs/step1
PIDS=()

CUDA_VISIBLE_DEVICES="$GPU_2B" "$STEP1_SCRIPT" \
    --model Qwen3.5-2B-Base --tensor-parallel-size 1 \
    "${EXTRA_ARGS[@]}" > logs/step1/Qwen3.5-2B-Base_bench_all.log 2>&1 &
PIDS+=($!)

CUDA_VISIBLE_DEVICES="$GPU_9B" "$STEP1_SCRIPT" \
    --model Qwen3.5-9B-Base --tensor-parallel-size 3 \
    "${EXTRA_ARGS[@]}" > logs/step1/Qwen3.5-9B-Base_bench_all.log 2>&1 &
PIDS+=($!)

CUDA_VISIBLE_DEVICES="$GPU_35B_A3B" "$STEP1_SCRIPT" \
    --model Qwen3.5-35B-A3B-Base --tensor-parallel-size 4 \
    "${EXTRA_ARGS[@]}" > logs/step1/Qwen3.5-35B-A3B-Base_bench_all.log 2>&1 &
PIDS+=($!)

FAILED=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        FAILED=1
    fi
done

if [[ "$FAILED" -eq 1 ]]; then
    echo "ERROR: one or more Step 1 benchmark runs failed. See logs/step1/*_bench_all.log"
    exit 1
fi

echo ""
echo "All Step 1 benchmark runs completed."
