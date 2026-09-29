bash -lc 'set -euo pipefail
cd ./sycophancy

export PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
export TORCHRUN_BIN="${TORCHRUN_BIN:-$(command -v torchrun)}"
export NNODES="${NNODES:-2}"
export LOCAL_GPUS="${LOCAL_GPUS:-8}"
export CUDA_LIST="${CUDA_LIST:-0,1,2,3,4,5,6,7}"
export MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-29500}}"
export MAX_LENGTH="${MAX_LENGTH:-512}"
export STEP4_QWEN35_A3B_FULL_MAX_LENGTH="${STEP4_QWEN35_A3B_FULL_MAX_LENGTH:-512}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed_step4_zero3_35b_512_fast.json}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-max_split_size_mb:512}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export ENABLE_DLC_MONITOR="${ENABLE_DLC_MONITOR:-1}"
export DLC_MONITOR_INTERVAL="${DLC_MONITOR_INTERVAL:-60}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
export FULL_MODEL_EXPORT="${FULL_MODEL_EXPORT:-checkpoint}"
export MIN_CUDA_DRIVER_VERSION="${MIN_CUDA_DRIVER_VERSION:-12060}"

# Train full syco_sft plus negative/positive preventative runs on the
# syco4000+alpaca1000 train / syco400 eval split selected by the run script.
export STEP4_DLC_BETAS="${STEP4_DLC_BETAS:--1 -5 -15 -30 -50 -80 -100}"
export STEP4_DLC_RUN_FULL="${STEP4_DLC_RUN_FULL:-1}"
export STEP4_DLC_RUN_NEGATIVE="${STEP4_DLC_RUN_NEGATIVE:-1}"
export STEP4_DLC_RUN_POSITIVE="${STEP4_DLC_RUN_POSITIVE:-1}"
export OVERWRITE="${OVERWRITE:-0}"

bash ./sycophancy/tests/run_step4_35b_train5000_eval400_seed1234.sh
'
