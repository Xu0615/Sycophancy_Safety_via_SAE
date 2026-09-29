#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "$PYTHON_BIN")/torchrun}"

MODEL_NAME="Qwen3.5-35B-A3B-Base"
MODEL_ROOT="./models"
MODEL_PATH="$MODEL_ROOT/$MODEL_NAME"

DATASET_TAG="${DATASET_TAG:-train5000_eval400_seed1234_syco4000_alpaca1000}"
SPLIT_TAG="${SPLIT_TAG:-split_train5000_eval400_seed1234_syco4000_alpaca1000}"
SYCO_SPLIT_TRAIN_SIZE="${SYCO_SPLIT_TRAIN_SIZE:-5000}"
SYCO_SPLIT_EVAL_SIZE="${SYCO_SPLIT_EVAL_SIZE:-400}"
SYCO_SPLIT_SEED="${SYCO_SPLIT_SEED:-1234}"
FEATURE_ID="${FEATURE_ID:-2362}"
LAYER="${LAYER:-27}"
FEATURE_TAG="${FEATURE_TAG:-f${FEATURE_ID}}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-syco${FEATURE_ID}_}"
EXPERIMENT_KEY="${EXPERIMENT_KEY:-${OUTPUT_PREFIX}${DATASET_TAG}}"
STEP4_DLC_COORD_SCOPE="${STEP4_DLC_COORD_SCOPE:-$EXPERIMENT_KEY}"
DATA="$PROJECT_ROOT/outputs/step4_feature_inject/dataset/$MODEL_NAME/$SPLIT_TAG/syco_dataset.jsonl"
SPLIT="$PROJECT_ROOT/outputs/step4_feature_inject/dataset/$MODEL_NAME/$SPLIT_TAG"
TRAIN_DATA="$SPLIT/syco_train.jsonl"

TRAIN_BASE="$PROJECT_ROOT/outputs/step4_feature_inject/train/$MODEL_NAME"
NEG_TRAIN_ROOT="$TRAIN_BASE/${OUTPUT_PREFIX}neg_$DATASET_TAG"
POS_TRAIN_ROOT="$TRAIN_BASE/${OUTPUT_PREFIX}pos_$DATASET_TAG"

CUDA_LIST="${CUDA_LIST:-0,1,2,3,4,5,6,7}"
LOCAL_GPUS="${LOCAL_GPUS:-8}"
NNODES="${NNODES:-${PET_NNODES:-${DLC_NNODES:-2}}}"

infer_dlc_node_rank() {
  local value
  if [[ "${HOSTNAME:-}" =~ ^(.+)-master-([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[2]}"
    return 0
  fi
  if [[ "${HOSTNAME:-}" =~ ^(.+)-worker-([0-9]+)$ ]]; then
    echo $((BASH_REMATCH[2] + 1))
    return 0
  fi
  value="${NODE_RANK:-}"
  if [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "$value"
    return 0
  fi
  for name in RANK INDEX WORKER_ID; do
    value="${!name:-}"
    if [[ "$value" =~ ^[0-9]+$ ]]; then
      echo "$value"
      return 0
    fi
  done
  echo 0
}

infer_dlc_master_addr() {
  local value
  for name in MASTER_ADDR PET_MASTER_ADDR PADDLE_MASTER; do
    value="${!name:-}"
    if [[ -n "$value" ]]; then
      echo "$value"
      return 0
    fi
  done
  if [[ "${HOSTNAME:-}" =~ ^(.+)-(master|worker)-[0-9]+$ ]]; then
    echo "${BASH_REMATCH[1]}-master-0"
    return 0
  fi
}

infer_dlc_job_name() {
  if [[ "${HOSTNAME:-}" =~ ^(.+)-(master|worker)-[0-9]+$ ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  echo "manual"
}

NODE_RANK="$(infer_dlc_node_rank)"
MASTER_ADDR="$(infer_dlc_master_addr)"
MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-29500}}"
GLOBAL_GPUS=$((NNODES * LOCAL_GPUS))
DLC_JOB_NAME="${DLC_JOB_NAME:-$(infer_dlc_job_name)}"
DLC_COORD_ROOT="${STEP4_DLC_COORD_ROOT:-$PROJECT_ROOT/tmp/dlc16_coord_${DLC_JOB_NAME}_${STEP4_DLC_COORD_SCOPE}}"
COORD_HEARTBEAT_INTERVAL="${COORD_HEARTBEAT_INTERVAL:-5}"
COORD_HEARTBEAT_MAX_AGE="${COORD_HEARTBEAT_MAX_AGE:-30}"
COORD_HEARTBEAT_PID=""

BATCH_SIZE=1
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$GLOBAL_GPUS}"
EPOCHS=1
LEARNING_RATE=2e-6
MAX_LENGTH="${MAX_LENGTH:-512}"
STEP4_QWEN35_A3B_FULL_MAX_LENGTH="${STEP4_QWEN35_A3B_FULL_MAX_LENGTH:-512}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed_step4_zero3_35b_512_fast.json}"
FULL_MODEL_EXPORT="${FULL_MODEL_EXPORT:-checkpoint}"
INJECTION_TARGETS="${STEP4_INJECTION_TARGETS:-syco}"
MIN_CUDA_DRIVER_VERSION="${MIN_CUDA_DRIVER_VERSION:-12060}"

RUN_SUFFIX="full_lr2e-6_ep1_gbs${GLOBAL_BATCH_SIZE}_maxlen${MAX_LENGTH}_dlc${GLOBAL_GPUS}gpu"
FULL_RUN_NAME="syco_sft_${RUN_SUFFIX}"
STEP4_DLC_BETAS="${STEP4_DLC_BETAS:--1 -5 -15 -30 -50 -80 -100}"
read -r -a BETAS <<< "$STEP4_DLC_BETAS"
STEP4_DLC_RUN_FULL="${STEP4_DLC_RUN_FULL:-1}"
STEP4_DLC_RUN_NEGATIVE="${STEP4_DLC_RUN_NEGATIVE:-1}"
STEP4_DLC_RUN_POSITIVE="${STEP4_DLC_RUN_POSITIVE:-1}"

ALLOW_MAX_LENGTH_TRUNCATION="${ALLOW_MAX_LENGTH_TRUNCATION:-1}"
ENABLE_DLC_MONITOR="${ENABLE_DLC_MONITOR:-1}"
DLC_MONITOR_INTERVAL="${DLC_MONITOR_INTERVAL:-60}"

LOG_DIR_TRAIN_NEG="$PROJECT_ROOT/logs/step4/$MODEL_NAME/neg_$DATASET_TAG"
LOG_DIR_TRAIN_POS="$PROJECT_ROOT/logs/step4/$MODEL_NAME/pos_$DATASET_TAG"
if [[ -n "$OUTPUT_PREFIX" ]]; then
  LOG_DIR_TRAIN_NEG="$PROJECT_ROOT/logs/step4/$MODEL_NAME/${OUTPUT_PREFIX}neg_$DATASET_TAG"
  LOG_DIR_TRAIN_POS="$PROJECT_ROOT/logs/step4/$MODEL_NAME/${OUTPUT_PREFIX}pos_$DATASET_TAG"
fi
PREFLIGHT_OK_FILE="$DLC_COORD_ROOT/preflight.ok"
LINK_FULL_DONE_FILE="$DLC_COORD_ROOT/link_full.done"

export PYTHON_BIN TORCHRUN_BIN CUDA_VISIBLE_DEVICES="$CUDA_LIST"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-max_split_size_mb:512}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-12}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-12}"
export STEP4_QWEN35_A3B_FULL_MAX_LENGTH
export NNODES NODE_RANK MASTER_ADDR MASTER_PORT

cd "$PROJECT_ROOT"

echo "[dlc-env] host=${HOSTNAME:-unknown} nnodes=$NNODES node_rank=$NODE_RANK local_gpus=$LOCAL_GPUS master=$MASTER_ADDR:$MASTER_PORT cuda=$CUDA_LIST"

if [[ "${NODE_RANK:-0}" == "0" ]]; then
  mkdir -p "$PROJECT_ROOT/tmp"
fi

is_primary() {
  [[ "${NODE_RANK:-0}" == "0" ]]
}

wait_for_file() {
  local path="$1"
  local timeout="${2:-3600}"
  local waited=0
  while [[ ! -e "$path" && "$waited" -lt "$timeout" ]]; do
    sleep 5
    waited=$((waited + 5))
  done
  [[ -e "$path" ]]
}

wait_for_fresh_file() {
  local path="$1"
  local min_epoch="$2"
  local timeout="${3:-1800}"
  local waited=0
  local mtime
  while [[ "$waited" -lt "$timeout" ]]; do
    if [[ -e "$path" ]]; then
      mtime="$(stat -c %Y "$path" 2>/dev/null || echo 0)"
      if [[ "$mtime" =~ ^[0-9]+$ && "$mtime" -ge "$min_epoch" ]]; then
        return 0
      fi
    fi
    sleep 5
    waited=$((waited + 5))
  done
  return 1
}

wait_for_recent_file() {
  local path="$1"
  local max_age="${2:-30}"
  local timeout="${3:-1800}"
  local waited=0
  local now
  local mtime
  local age
  while [[ "$waited" -lt "$timeout" ]]; do
    if [[ -e "$path" ]]; then
      now="$(date +%s)"
      mtime="$(stat -c %Y "$path" 2>/dev/null || echo 0)"
      if [[ "$mtime" =~ ^[0-9]+$ ]]; then
        age=$((now - mtime))
        if [[ "$age" -le "$max_age" ]]; then
          return 0
        fi
      fi
    fi
    sleep 5
    waited=$((waited + 5))
  done
  return 1
}

wait_for_rank_files() {
  local prefix="$1"
  local timeout="${2:-3600}"
  local waited=0
  local rank
  while [[ "$waited" -lt "$timeout" ]]; do
    local ready=1
    for ((rank = 0; rank < NNODES; rank++)); do
      if [[ ! -e "${prefix}.rank${rank}" ]]; then
        ready=0
        break
      fi
    done
    if [[ "$ready" -eq 1 ]]; then
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
  done
  return 1
}

wait_for_decision() {
  local coord_dir="$1"
  local timeout="${2:-1800}"
  local waited=0
  while [[ "$waited" -lt "$timeout" ]]; do
    if [[ -e "$coord_dir/decision.skip" || -e "$coord_dir/decision.train" ]]; then
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
  done
  return 1
}

stop_bg_pid() {
  local pid="${1:-}"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

start_coord_heartbeat() {
  (
    while true; do
      date '+%Y-%m-%d %H:%M:%S' > "$DLC_COORD_ROOT/heartbeat.tmp"
      mv "$DLC_COORD_ROOT/heartbeat.tmp" "$DLC_COORD_ROOT/heartbeat"
      sleep "$COORD_HEARTBEAT_INTERVAL"
    done
  ) &
  COORD_HEARTBEAT_PID="$!"
}

cleanup_coord_heartbeat() {
  stop_bg_pid "$COORD_HEARTBEAT_PID"
}
trap cleanup_coord_heartbeat EXIT

init_coord_root() {
  if is_primary; then
    rm -rf "$DLC_COORD_ROOT"
    mkdir -p "$DLC_COORD_ROOT"
    date '+%Y-%m-%d %H:%M:%S' > "$DLC_COORD_ROOT/init.ready"
    start_coord_heartbeat
    echo "[coord] initialized root=$DLC_COORD_ROOT heartbeat_pid=$COORD_HEARTBEAT_PID"
  else
    if ! wait_for_recent_file "$DLC_COORD_ROOT/heartbeat" "$COORD_HEARTBEAT_MAX_AGE" 1800; then
      echo "ERROR: timeout waiting for active DLC coordination heartbeat: $DLC_COORD_ROOT/heartbeat" >&2
      exit 1
    fi
    echo "[coord] joined root=$DLC_COORD_ROOT"
  fi
}

print_gpu_monitor_snapshot() {
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[gpu-monitor] ts=$ts host=${HOSTNAME:-unknown} node_rank=$NODE_RANK cuda=$CUDA_LIST"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[gpu-monitor] nvidia-smi not found in PATH"
    return 0
  fi
  nvidia-smi \
    --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader,nounits \
    | awk -F',' -v host="${HOSTNAME:-unknown}" -v node="$NODE_RANK" '
      {
        for (i = 1; i <= NF; i++) {
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
        }
        printf("[gpu-monitor] host=%s node_rank=%s gpu=%s name=%s mem=%s/%s MiB util=%s%% temp=%sC\n",
          host, node, $1, $2, $3, $4, $5, $6)
      }'
}

print_training_progress_snapshot() {
  local phase="$1"
  local run_name="$2"
  local output_dir="$3"
  local progress_path="$output_dir/training_progress.json"
  if [[ ! -f "$progress_path" ]]; then
    echo "[train-progress] phase=$phase run=$run_name status=waiting_for_progress_file path=$progress_path"
    return 0
  fi
  "$PYTHON_BIN" - "$progress_path" "$phase" "$run_name" <<'PY'
import json
import math
import sys

path, phase, run_name = sys.argv[1:]
try:
    with open(path, encoding="utf-8") as f:
        row = json.load(f)
except Exception as exc:
    print(f"[train-progress] phase={phase} run={run_name} status=unreadable_progress_file error={exc}")
    raise SystemExit(0)

step = int(row.get("step") or 0)
total = int(row.get("total_steps") or 0)
elapsed = float(row.get("elapsed_sec") or 0.0)
pct = (100.0 * step / total) if total > 0 else 0.0
eta = None
if step > 0 and total > step and elapsed > 0:
    eta = elapsed / step * (total - step)

def fmt_float(value, digits=4):
    if value is None:
        return "NA"
    try:
        if math.isnan(float(value)):
            return "NA"
        return f"{float(value):.{digits}g}"
    except Exception:
        return str(value)

eta_text = f"{eta / 60.0:.1f}m" if eta is not None else "NA"
elapsed_text = f"{elapsed / 60.0:.1f}m"
print(
    "[train-progress] "
    f"phase={phase} run={run_name} status={row.get('status', 'training')} "
    f"step={step}/{total} pct={pct:.1f}% epoch={row.get('epoch')}/{row.get('epochs')} "
    f"loss={fmt_float(row.get('loss'))} lr={fmt_float(row.get('lr'), 3)} "
    f"beta={fmt_float(row.get('beta'))} elapsed={elapsed_text} eta={eta_text}"
)
PY
}

monitor_run_status() {
  local phase="$1"
  local run_name="$2"
  local output_dir="$3"
  while true; do
    print_gpu_monitor_snapshot
    if is_primary; then
      print_training_progress_snapshot "$phase" "$run_name" "$output_dir"
    fi
    sleep "$DLC_MONITOR_INTERVAL"
  done
}

stop_monitor() {
  local pid="${1:-}"
  stop_bg_pid "$pid"
}

cuda_driver_version_code() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 0
    return 0
  fi
  local cuda_version
  cuda_version="$(
    nvidia-smi 2>/dev/null \
      | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' \
      | head -1
  )"
  if [[ -z "$cuda_version" ]]; then
    echo 0
    return 0
  fi
  "$PYTHON_BIN" - "$cuda_version" <<'PY'
import sys

parts = sys.argv[1].split(".")
major = int(parts[0]) if parts and parts[0].isdigit() else 0
minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
print(major * 1000 + minor * 10)
PY
}

check_cuda_driver_version() {
  local version_code
  local ok=1
  local status_file="$DLC_COORD_ROOT/driver.rank${NODE_RANK}"
  version_code="$(cuda_driver_version_code)"
  if [[ "$version_code" -le 0 ]]; then
    ok=0
  fi
  if [[ "$version_code" -lt "$MIN_CUDA_DRIVER_VERSION" ]]; then
    ok=0
  fi
  printf 'ok=%s version_code=%s required=%s host=%s node_rank=%s\n' \
    "$ok" "$version_code" "$MIN_CUDA_DRIVER_VERSION" "${HOSTNAME:-unknown}" "$NODE_RANK" > "$status_file"
  if ! wait_for_rank_files "$DLC_COORD_ROOT/driver" 1800; then
    echo "ERROR: timeout waiting for all nodes to report CUDA driver version." >&2
    exit 1
  fi
  cat "$DLC_COORD_ROOT"/driver.rank*
  if grep -q '^ok=0 ' "$DLC_COORD_ROOT"/driver.rank*; then
    echo "ERROR: at least one DLC node has a CUDA driver interface version too old for this PyTorch/CUDA build." >&2
    echo "This 35B-A3B ZeRO-3 run previously succeeded on CUDA driver interface 13000 nodes; the failed retry was scheduled on a 12060 node and crashed during MoE expert all-gather." >&2
    echo "Resubmit the DLC job to nodes with CUDA driver interface >= $MIN_CUDA_DRIVER_VERSION, or use an image built for the older CUDA driver." >&2
    exit 1
  fi
}

if [[ -z "$MASTER_ADDR" ]]; then
  echo "ERROR: MASTER_ADDR/PET_MASTER_ADDR is empty. In DLC, set MASTER_ADDR to worker-0 address." >&2
  exit 1
fi
if [[ "$MAX_LENGTH" -gt "$STEP4_QWEN35_A3B_FULL_MAX_LENGTH" ]]; then
  echo "ERROR: MAX_LENGTH=$MAX_LENGTH exceeds STEP4_QWEN35_A3B_FULL_MAX_LENGTH=$STEP4_QWEN35_A3B_FULL_MAX_LENGTH for 35B-A3B full tuning." >&2
  echo "Lower STEP4_QWEN35_A3B_FULL_MAX_LENGTH or MAX_LENGTH if this 16x80GB ZeRO-3 run still OOMs." >&2
  exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: model path not found: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$DATA" || ! -f "$TRAIN_DATA" || ! -d "$SPLIT" ]]; then
  echo "ERROR: dataset/split files not found under $SPLIT" >&2
  exit 1
fi

init_coord_root
check_cuda_driver_version

if is_primary; then
  "$PYTHON_BIN" - "$MODEL_PATH" "$TRAIN_DATA" "$MAX_LENGTH" "$ALLOW_MAX_LENGTH_TRUNCATION" <<'PY'
import json
import statistics
import sys
from transformers import AutoTokenizer

model_path, train_path, max_length_s, allow_truncation_s = sys.argv[1:]
max_length = int(max_length_s)
allow_truncation = allow_truncation_s == "1"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def extract_input_ids(tokenized):
    ids = tokenized["input_ids"] if isinstance(tokenized, dict) or hasattr(tokenized, "data") else tokenized
    if hasattr(ids, "detach"):
        ids = ids.detach().cpu().tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]

def format_len(prompt, response):
    messages = [{"role": "user", "content": str(prompt)}]
    try:
        prefix_ids = extract_input_ids(tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False))
    except TypeError:
        prefix_ids = extract_input_ids(tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True))
    response_ids = tokenizer(str(response), add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        response_ids = list(response_ids) + [int(tokenizer.eos_token_id)]
    return len(prefix_ids) + len(response_ids), len(prefix_ids), len(response_ids)

lengths = []
long_rows = []
with open(train_path, encoding="utf-8") as f:
    for lineno, line in enumerate(f, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        total, prompt_tokens, response_tokens = format_len(
            row.get("prompt") or row.get("prompt_text", ""),
            row.get("response", ""),
        )
        lengths.append(total)
        if total > max_length:
            long_rows.append((total, prompt_tokens, response_tokens, lineno, row.get("id", "")))

if not lengths:
    raise SystemExit(f"ERROR: no rows found in {train_path}")
lengths_sorted = sorted(lengths)
def pct(q):
    idx = min(len(lengths_sorted) - 1, max(0, int(round((len(lengths_sorted) - 1) * q))))
    return lengths_sorted[idx]

print(
    "[preflight] train token lengths: "
    f"rows={len(lengths)} max={max(lengths)} mean={statistics.mean(lengths):.1f} "
    f"p95={pct(0.95)} p99={pct(0.99)} max_length={max_length}"
)
if long_rows:
    preview = ", ".join(
        f"line={line} id={row_id} total={total} prompt={prompt} response={resp}"
        for total, prompt, resp, line, row_id in sorted(long_rows, reverse=True)[:8]
    )
    message = (
        f"{len(long_rows)} training rows exceed MAX_LENGTH={max_length}; "
        f"they will be left-truncated by the training dataset. Longest: {preview}"
    )
    if not allow_truncation:
        raise SystemExit(f"ERROR: {message}")
    print(f"[preflight] WARNING: {message}")
PY
  touch "$PREFLIGHT_OK_FILE"
else
  wait_for_file "$PREFLIGHT_OK_FILE" 1800
fi

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[preflight] OK: DLC 16-GPU configuration is valid on node_rank=$NODE_RANK."
  exit 0
fi

if is_primary; then
  mkdir -p \
    "$NEG_TRAIN_ROOT" "$POS_TRAIN_ROOT" \
    "$LOG_DIR_TRAIN_NEG" "$LOG_DIR_TRAIN_POS" \
    "$PROJECT_ROOT/tmp"
fi

is_train_complete() {
  local output_dir="$1"
  "$PYTHON_BIN" - "$output_dir" <<'PY'
import json
import os
import sys

path = sys.argv[1]
summary_path = os.path.join(path, "training_summary.json")
loss_path = os.path.join(path, "loss_history.csv")
if not (os.path.exists(summary_path) and os.path.exists(loss_path)):
    raise SystemExit(1)
with open(summary_path, encoding="utf-8") as f:
    summary = json.load(f)
if str(summary.get("tuning_mode", "")).lower() != "full":
    raise SystemExit(1)
full_save = summary.get("full_model_save") or {}

def has_complete_safetensors_model(root):
    single = os.path.join(root, "model.safetensors")
    if os.path.exists(single) and os.path.getsize(single) > 0:
        return True
    index_path = os.path.join(root, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return False
    try:
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
    except Exception:
        return False
    shard_names = set((index.get("weight_map") or {}).values())
    return bool(shard_names) and all(
        os.path.getsize(os.path.join(root, name)) > 0
        for name in shard_names
        if name
    )

def has_complete_deepspeed_checkpoint(root, expected_world_size):
    ckpt_dir = os.path.join(root, "deepspeed_checkpoint")
    latest_path = os.path.join(ckpt_dir, "latest")
    if not os.path.exists(latest_path):
        return False
    with open(latest_path, encoding="utf-8") as f:
        tag = f.read().strip()
    tag_dir = os.path.join(ckpt_dir, tag)
    if not os.path.isdir(tag_dir):
        return False
    model_count = len([x for x in os.listdir(tag_dir) if x.endswith("_model_states.pt")])
    optim_count = len([x for x in os.listdir(tag_dir) if x.endswith("_optim_states.pt")])
    if expected_world_size > 1:
        return model_count == expected_world_size and optim_count == expected_world_size
    return model_count > 0 and model_count == optim_count

expected_world_size = int(summary.get("world_size") or os.environ.get("GLOBAL_GPUS") or 0)
if has_complete_safetensors_model(path) or has_complete_deepspeed_checkpoint(path, expected_world_size):
    raise SystemExit(0)
raise SystemExit(1)
PY
}

remove_selected_outputs_for_overwrite() {
  if ! is_primary || [[ "${OVERWRITE:-0}" != "1" ]]; then
    return 0
  fi
  local beta tag run_name
  if [[ "$STEP4_DLC_RUN_FULL" == "1" ]]; then
    rm -rf "$NEG_TRAIN_ROOT/$FULL_RUN_NAME"
  fi
  if [[ "$STEP4_DLC_RUN_NEGATIVE" == "1" ]]; then
    for beta in "${BETAS[@]}"; do
      tag="${beta#-}"
      run_name="syco_sft_prevent_${FEATURE_TAG}_alpha_neg${tag}_${RUN_SUFFIX}"
      rm -rf "$NEG_TRAIN_ROOT/$run_name"
    done
  fi
  if [[ "$STEP4_DLC_RUN_POSITIVE" == "1" ]]; then
    for beta in "${BETAS[@]}"; do
      tag="${beta#-}"
      run_name="syco_sft_prevent_${FEATURE_TAG}_alpha${tag}_${RUN_SUFFIX}"
      rm -rf "$POS_TRAIN_ROOT/$run_name"
    done
  fi
}

prepare_train_output_dir() {
  local output_dir="$1"
  if is_primary && [[ -d "$output_dir" ]] && ! is_train_complete "$output_dir"; then
    echo "[train] remove incomplete output dir before retry: $output_dir"
    rm -rf "$output_dir"
  fi
}

run_train() {
  local phase="$1"
  local train_root="$2"
  local log_dir="$3"
  local run_name="$4"
  local group="$5"
  local beta="${6:-}"
  local output_dir="$train_root/$run_name"
  local coord_dir="$DLC_COORD_ROOT/${phase}_${run_name}"
  local done_file="$coord_dir/primary.done"
  local skip_file="$coord_dir/primary.skip"
  if is_primary; then
    rm -rf "$coord_dir"
    mkdir -p "$coord_dir"
    rm -f "$done_file" "$skip_file" 2>/dev/null || true
    touch "$coord_dir/init.ready"
  fi
  if ! wait_for_file "$coord_dir/init.ready" 1800; then
    echo "ERROR: timeout waiting for run coordination init: $run_name" >&2
    exit 1
  fi
  touch "$coord_dir/arrive.rank${NODE_RANK}"
  echo "[coord] arrived run=$phase/$run_name node_rank=$NODE_RANK coord=$coord_dir"
  if ! wait_for_rank_files "$coord_dir/arrive" 1800; then
    echo "ERROR: timeout waiting for all nodes before run decision: $run_name" >&2
    exit 1
  fi

  if is_primary; then
    if is_train_complete "$output_dir"; then
      echo "[train:$phase] skip completed: $run_name"
      touch "$skip_file"
      touch "$coord_dir/decision.skip"
    else
      prepare_train_output_dir "$output_dir"
      touch "$coord_dir/decision.train"
    fi
  fi
  if ! wait_for_decision "$coord_dir" 1800; then
    echo "ERROR: timeout waiting for run decision: $run_name" >&2
    exit 1
  fi
  if [[ -e "$coord_dir/decision.skip" ]]; then
    echo "[coord] skip run=$phase/$run_name node_rank=$NODE_RANK"
    touch "$coord_dir/skip.rank${NODE_RANK}"
    if ! wait_for_rank_files "$coord_dir/skip" 1800; then
      echo "ERROR: timeout waiting for all nodes to acknowledge skip: $run_name" >&2
      exit 1
    fi
    return 0
  fi
  echo "[coord] train run=$phase/$run_name node_rank=$NODE_RANK"

  local beta_args=()
  if [[ -n "$beta" ]]; then
    beta_args=(--beta "$beta" --beta-schedule fixed)
  fi

  echo
  echo "============================================================"
  echo "[train:$phase] $run_name node_rank=$NODE_RANK nnodes=$NNODES local_gpus=$LOCAL_GPUS global_gpus=$GLOBAL_GPUS"
  echo "============================================================"

  local monitor_pid=""
  if [[ "$ENABLE_DLC_MONITOR" == "1" ]]; then
    monitor_run_status "$phase" "$run_name" "$output_dir" &
    monitor_pid="$!"
  fi

  set +e
  STEP4_LOG_DIR="$log_dir" \
    bash "$PROJECT_ROOT/run_scripts/run_step4_syco_train.sh" \
      --model "$MODEL_NAME" \
      --model-root "$MODEL_ROOT" \
      --model-path "$MODEL_PATH" \
      --group "$group" \
      "${beta_args[@]}" \
      --feature-id "$FEATURE_ID" \
      --layer "$LAYER" \
      --tuning-mode full \
      --output-dir "$output_dir" \
      --run-name "$run_name" \
      --injection-targets "$INJECTION_TARGETS" \
      --shared-syco-dataset-path "$DATA" \
      --syco-split-dir "$SPLIT" \
      --syco-split-train-size "$SYCO_SPLIT_TRAIN_SIZE" \
      --syco-split-eval-size "$SYCO_SPLIT_EVAL_SIZE" \
      --syco-split-seed "$SYCO_SPLIT_SEED" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --global-batch-size "$GLOBAL_BATCH_SIZE" \
      --learning-rate "$LEARNING_RATE" \
      --warmup-ratio 0.03 \
      --max-length "$MAX_LENGTH" \
      --full-model-export "$FULL_MODEL_EXPORT" \
      --gpu "$CUDA_LIST" \
      --num-gpus "$LOCAL_GPUS" \
      --deepspeed-config "$DEEPSPEED_CONFIG"
  local train_status="$?"
  set -e
  stop_monitor "$monitor_pid"
  if [[ "$train_status" -ne 0 ]]; then
    touch "$coord_dir/failed.rank${NODE_RANK}" 2>/dev/null || true
    return "$train_status"
  fi

  touch "$coord_dir/done.rank${NODE_RANK}"
  if is_primary; then
    touch "$done_file"
  else
    if ! wait_for_file "$done_file" 3600; then
      echo "ERROR: timeout waiting for primary done file after training: $run_name" >&2
      exit 1
    fi
  fi
  if ! wait_for_rank_files "$coord_dir/done" 1800; then
    echo "ERROR: timeout waiting for all nodes to leave training run: $run_name" >&2
    exit 1
  fi
}

run_full_once() {
  run_train "no_injection" "$NEG_TRAIN_ROOT" "$LOG_DIR_TRAIN_NEG" "$FULL_RUN_NAME" "syco_sft"
}

link_full_into_pos_root() {
  local src="$NEG_TRAIN_ROOT/$FULL_RUN_NAME"
  local dst="$POS_TRAIN_ROOT/$FULL_RUN_NAME"
  if ! is_primary; then
    wait_for_file "$LINK_FULL_DONE_FILE" 3600
    return 0
  fi
  if ! is_train_complete "$src"; then
    echo "ERROR: no-injection run is not complete, cannot link into positive root: $src" >&2
    exit 1
  fi
  if [[ -e "$dst" || -L "$dst" ]]; then
    if [[ "$(readlink -f "$dst")" == "$(readlink -f "$src")" ]]; then
      touch "$LINK_FULL_DONE_FILE"
      return 0
    fi
    if is_train_complete "$dst"; then
      echo "[link] keep existing completed no-injection run in positive root: $dst"
      touch "$LINK_FULL_DONE_FILE"
      return 0
    fi
    echo "ERROR: target already exists and is not the expected symlink: $dst" >&2
    exit 1
  fi
  ln -s "$src" "$dst"
  touch "$LINK_FULL_DONE_FILE"
}

run_negative() {
  local beta="$1"
  local tag="${beta#-}"
  local run_name="syco_sft_prevent_${FEATURE_TAG}_alpha_neg${tag}_${RUN_SUFFIX}"
  run_train "negative" "$NEG_TRAIN_ROOT" "$LOG_DIR_TRAIN_NEG" "$run_name" "syco_sft_prevent" "$beta"
}

run_positive() {
  local raw_beta="$1"
  local tag="${raw_beta#-}"
  local beta="$tag"
  local run_name="syco_sft_prevent_${FEATURE_TAG}_alpha${tag}_${RUN_SUFFIX}"
  run_train "positive" "$POS_TRAIN_ROOT" "$LOG_DIR_TRAIN_POS" "$run_name" "syco_sft_prevent" "$beta"
}

echo
echo "============================================================"
echo "Step4 35B-A3B DLC 16-GPU full SFT training"
echo "model      : $MODEL_PATH"
echo "experiment : $EXPERIMENT_KEY"
echo "dataset    : $SPLIT"
echo "feature    : $FEATURE_TAG layer=$LAYER"
echo "node       : rank=$NODE_RANK nnodes=$NNODES local_gpus=$LOCAL_GPUS global_gpus=$GLOBAL_GPUS"
echo "master     : $MASTER_ADDR:$MASTER_PORT"
echo "max length : $MAX_LENGTH"
echo "safe cap   : $STEP4_QWEN35_A3B_FULL_MAX_LENGTH"
echo "export     : $FULL_MODEL_EXPORT"
echo "betas      : ${BETAS[*]}"
echo "run full   : $STEP4_DLC_RUN_FULL"
echo "run neg    : $STEP4_DLC_RUN_NEGATIVE"
echo "run pos    : $STEP4_DLC_RUN_POSITIVE"
echo "train neg  : $NEG_TRAIN_ROOT"
echo "train pos  : $POS_TRAIN_ROOT"
echo "ds config  : $DEEPSPEED_CONFIG"
echo "============================================================"

remove_selected_outputs_for_overwrite

if [[ "$STEP4_DLC_RUN_FULL" == "1" ]]; then
  run_full_once
fi
if [[ "$STEP4_DLC_RUN_NEGATIVE" == "1" ]]; then
  for beta in "${BETAS[@]}"; do
    run_negative "$beta"
  done
fi

if [[ "$STEP4_DLC_RUN_POSITIVE" == "1" ]]; then
  link_full_into_pos_root
  for beta in "${BETAS[@]}"; do
    run_positive "$beta"
  done
fi

if is_primary; then
  echo
  echo "Done."
  echo "  negative train root : $NEG_TRAIN_ROOT"
  echo "  positive train root : $POS_TRAIN_ROOT"
fi
