#!/bin/bash
# ============================================================
# run_step4_syco_train.sh - Step 4 sycophancy SFT launcher
#
# Usage:
#   bash run_scripts/run_step4_syco_train.sh --group syco_sft
#   bash run_scripts/run_step4_syco_train.sh --groups "syco_sft syco_sft_prevent" --beta-values "5 10"
#   bash run_scripts/run_step4_syco_train.sh --group syco_sft --epochs 2 --learning-rate 2e-6
#   bash run_scripts/run_step4_syco_train.sh --model Qwen3.5-35B-A3B-Base --group syco_sft --tuning-mode full
#   bash run_scripts/run_step4_syco_train.sh --model Qwen3.5-9B-Base --group syco_sft_prevent --feature-id 61718 --layer 19
#
# Groups are defined in configs/step4_vaccine.yaml:
#   syco_sft, syco_sft_prevent, objective_sft, objective_sft_prevent, vaccine_sft,
#   vaccine_sft_random, vaccine_sft_negative
# ============================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
ORIGINAL_ARGS=("$@")

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-$(dirname "$PYTHON_BIN")/deepspeed}"
MODEL_ROOT="${MODEL_ROOT:-./models}"
CONDA_BIN_DIR="$(dirname "$PYTHON_BIN")"
export PATH="$CONDA_BIN_DIR:$PATH"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-max_split_size_mb:512}"
TORCH_EXTENSIONS_BASE="${TORCH_EXTENSIONS_BASE:-/tmp/torch_extensions_sycophancy}"
if [[ -z "${TORCH_EXTENSIONS_DIR:-}" || "${TORCH_EXTENSIONS_DIR:-}" == "$TORCH_EXTENSIONS_BASE" ]]; then
    TORCH_EXTENSIONS_TAG="$("$PYTHON_BIN" - <<'PY'
import importlib.metadata
import re
import sys

import torch


def clean(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


try:
    deepspeed_version = importlib.metadata.version("deepspeed")
except importlib.metadata.PackageNotFoundError:
    deepspeed_version = "unknown"

parts = [
    f"py{sys.version_info.major}{sys.version_info.minor}",
    "torch" + clean(torch.__version__),
    "cu" + clean(torch.version.cuda or "cpu"),
    "ds" + clean(deepspeed_version),
]
print("_".join(parts))
PY
)"
    export TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_BASE/$TORCH_EXTENSIONS_TAG"
else
    export TORCH_EXTENSIONS_DIR
fi
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "$PYTHON_BIN")/torchrun}"
CONFIG="configs/step4_vaccine.yaml"
MODEL_NAME="Qwen3.5-35B-A3B-Base"
MODEL_PATH=""
GROUP="syco_sft"
EXPERIMENT_GROUPS=""
BETA_VALUES="${BETA_VALUES:-5 10 20 30}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-3,4,5,6,7}"
FEATURE_ID=""
LAYER=""
SAE_DIR=""
BETA=""
SHARED_SYCO_DATASET_PATH=""
SYCO_SPLIT_DIR=""
SYCO_SPLIT_TRAIN_SIZE=""
SYCO_SPLIT_EVAL_SIZE=""
SYCO_SPLIT_SEED=""
OBJECTIVE_TRAIN_PATH=""
AHC_TRAIN_PATH=""
ALLOW_MISSING_AHC=0
EPOCHS="3"
BATCH_SIZE="1"
GLOBAL_BATCH_SIZE="5"
GRADIENT_ACCUMULATION_STEPS=""
GRADIENT_CHECKPOINTING=""
LEARNING_RATE=""
WARMUP_RATIO="0.03"
WARMUP_STEPS=""
MAX_STEPS=""
MAX_LENGTH=""
MAX_TRAIN_EXAMPLES=""
MAX_SYCO_EXAMPLES=""
MAX_HARM_EXAMPLES=""
LORA_RANK=""
LORA_ALPHA=""
LORA_DROPOUT=""
TUNING_MODE="${STEP4_TUNING_MODE:-full}"
FULL_MODEL_EXPORT="${STEP4_FULL_MODEL_EXPORT:-}"
SYCO_FRONT_TOKEN_COUNT=""
SYCO_FRONT_TOKEN_WEIGHT=""
BETA_SCHEDULE=""
INJECTION_TARGETS="${STEP4_INJECTION_TARGETS:-syco}"
INJECTION_FRONT_TOKEN_COUNT=""
NUM_GPUS="${NUM_GPUS:-}"
DEEPSPEED_CONFIG="configs/deepspeed_step4_zero3_35b_512_safe.json"
OUTPUT_ROOT="outputs/step4_feature_inject/train"
OUTPUT_DIR=""
RUN_NAME=""
USER_OUTPUT_DIR=0
USER_RUN_NAME=0
SKIP_COMPLETED="${STEP4_SKIP_COMPLETED:-0}"
NO_DEEPSPEED=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model|--model-name|--model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --model-path|--model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --model-root|--model_root)
            MODEL_ROOT="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --group)
            GROUP="$2"
            shift 2
            ;;
        --groups)
            EXPERIMENT_GROUPS="$2"
            shift 2
            ;;
        --beta-values|--beta_values)
            BETA_VALUES="$2"
            shift 2
            ;;
        --gpu|--gpus)
            GPU_LIST="$2"
            shift 2
            ;;
        --feature-id|--feature_id)
            FEATURE_ID="$2"
            shift 2
            ;;
        --layer)
            LAYER="$2"
            shift 2
            ;;
        --sae-dir|--sae_dir)
            SAE_DIR="$2"
            shift 2
            ;;
        --beta|--steering-coef|--steering_coef)
            BETA="$2"
            shift 2
            ;;
        --beta-max|--beta_max)
            EXTRA_ARGS+=(--beta-max "$2")
            shift 2
            ;;
        --beta-min|--beta_min)
            EXTRA_ARGS+=(--beta-min "$2")
            shift 2
            ;;
        --beta-schedule|--beta_schedule)
            BETA_SCHEDULE="$2"
            shift 2
            ;;
        --injection-targets|--injection_targets)
            INJECTION_TARGETS="$2"
            shift 2
            ;;
        --injection-front-token-count|--injection_front_token_count)
            INJECTION_FRONT_TOKEN_COUNT="$2"
            shift 2
            ;;
        --output-dir|--output_dir)
            OUTPUT_DIR="$2"
            USER_OUTPUT_DIR=1
            shift 2
            ;;
        --run-name|--run_name)
            RUN_NAME="$2"
            USER_RUN_NAME=1
            shift 2
            ;;
        --output-root|--output_root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --shared-syco-dataset-path|--shared_syco_dataset_path)
            SHARED_SYCO_DATASET_PATH="$2"
            shift 2
            ;;
        --syco-split-dir|--syco_split_dir)
            SYCO_SPLIT_DIR="$2"
            shift 2
            ;;
        --syco-split-train-size|--syco_split_train_size)
            SYCO_SPLIT_TRAIN_SIZE="$2"
            shift 2
            ;;
        --syco-split-eval-size|--syco_split_eval_size)
            SYCO_SPLIT_EVAL_SIZE="$2"
            shift 2
            ;;
        --syco-split-seed|--syco_split_seed)
            SYCO_SPLIT_SEED="$2"
            shift 2
            ;;
        --objective-train-path|--objective_train_path)
            OBJECTIVE_TRAIN_PATH="$2"
            shift 2
            ;;
        --ahc-train-path|--ahc_train_path)
            AHC_TRAIN_PATH="$2"
            shift 2
            ;;
        --allow-missing-ahc|--allow_missing_ahc)
            ALLOW_MISSING_AHC=1
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size|--batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --global-batch-size|--global_batch_size)
            GLOBAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --gradient-accumulation-steps|--gradient_accumulation_steps)
            GRADIENT_ACCUMULATION_STEPS="$2"
            shift 2
            ;;
        --gradient-checkpointing|--gradient_checkpointing)
            GRADIENT_CHECKPOINTING="1"
            shift
            ;;
        --no-gradient-checkpointing|--no_gradient_checkpointing)
            GRADIENT_CHECKPOINTING="0"
            shift
            ;;
        --learning-rate|--learning_rate|--lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --warmup-ratio|--warmup_ratio)
            WARMUP_RATIO="$2"
            WARMUP_STEPS=""
            shift 2
            ;;
        --warmup-steps|--warmup_steps)
            WARMUP_STEPS="$2"
            WARMUP_RATIO=""
            shift 2
            ;;
        --max-steps|--max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --max-length|--max_length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --max-train-examples|--max_train_examples)
            MAX_TRAIN_EXAMPLES="$2"
            shift 2
            ;;
        --max-syco-examples|--max_syco_examples)
            MAX_SYCO_EXAMPLES="$2"
            shift 2
            ;;
        --max-harm-examples|--max_harm_examples)
            MAX_HARM_EXAMPLES="$2"
            shift 2
            ;;
        --tuning-mode|--tuning_mode)
            TUNING_MODE="$2"
            shift 2
            ;;
        --full-finetune|--full_finetune)
            TUNING_MODE="full"
            shift
            ;;
        --lora)
            TUNING_MODE="lora"
            shift
            ;;
        --lora-rank|--lora_rank)
            LORA_RANK="$2"
            shift 2
            ;;
        --lora-alpha|--lora_alpha)
            LORA_ALPHA="$2"
            shift 2
            ;;
        --lora-dropout|--lora_dropout)
            LORA_DROPOUT="$2"
            shift 2
            ;;
        --syco-front-token-count|--syco_front_token_count)
            SYCO_FRONT_TOKEN_COUNT="$2"
            shift 2
            ;;
        --syco-front-token-weight|--syco_front_token_weight)
            SYCO_FRONT_TOKEN_WEIGHT="$2"
            shift 2
            ;;
        --num-gpus|--num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --deepspeed-config|--deepspeed_config)
            DEEPSPEED_CONFIG="$2"
            shift 2
            ;;
        --full-model-export|--full_model_export)
            FULL_MODEL_EXPORT="$2"
            shift 2
            ;;
        --no-deepspeed)
            NO_DEEPSPEED=1
            shift
            ;;
        --skip-completed)
            SKIP_COMPLETED="1"
            shift
            ;;
        --no-skip-completed)
            SKIP_COMPLETED="0"
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

export CUDA_VISIBLE_DEVICES="$GPU_LIST"

count_visible_gpus() {
    local value="$1"
    local count=0
    local gpu_id
    IFS=',' read -ra _gpu_ids <<< "$value"
    for gpu_id in "${_gpu_ids[@]}"; do
        if [[ -n "${gpu_id//[[:space:]]/}" ]]; then
            count=$((count + 1))
        fi
    done
    echo "$count"
}

if [[ -z "$NUM_GPUS" ]]; then
    NUM_GPUS="$(count_visible_gpus "$CUDA_VISIBLE_DEVICES")"
fi

infer_node_rank() {
    for name in NODE_RANK INDEX OMPI_COMM_WORLD_RANK RANK WORKER_ID; do
        local value="${!name:-}"
        if [[ "$value" =~ ^[0-9]+$ ]]; then
            echo "$value"
            return 0
        fi
    done
    echo 0
}

is_multinode_torchrun() {
    local nnodes="${NNODES:-${PET_NNODES:-${DLC_NNODES:-1}}}"
    [[ "$nnodes" =~ ^[0-9]+$ && "$nnodes" -gt 1 ]]
}

if [[ "$MODEL_NAME" == /* ]]; then
    MODEL_PATH="$MODEL_NAME"
    MODEL_NAME="$(basename "${MODEL_PATH%/}")"
fi
if [[ -z "$MODEL_PATH" ]]; then
    MODEL_PATH="$MODEL_ROOT/$MODEL_NAME"
fi

if [[ -z "$SHARED_SYCO_DATASET_PATH" ]]; then
    SHARED_SYCO_DATASET_PATH="outputs/step4_feature_inject/dataset/$MODEL_NAME/syco_dataset.jsonl"
fi
infer_cached_split_train_size() {
    local metadata_path
    metadata_path="$(dirname "$SHARED_SYCO_DATASET_PATH")/dataset_metadata.json"
    if [[ -s "$metadata_path" ]]; then
        "$PYTHON_BIN" - "$metadata_path" <<'PY' 2>/dev/null || true
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        metadata = json.load(handle)
except (OSError, json.JSONDecodeError):
    raise SystemExit(0)
status = str(metadata.get("recovery_status") or "")
if status.startswith("complete_for_train1100_eval400"):
    value = metadata.get("train_size")
    if isinstance(value, int) and value > 0:
        print(value)
PY
    fi
}
if [[ -z "$SYCO_SPLIT_TRAIN_SIZE" ]]; then
    SYCO_SPLIT_TRAIN_SIZE="$(infer_cached_split_train_size)"
    SYCO_SPLIT_TRAIN_SIZE="${SYCO_SPLIT_TRAIN_SIZE:-5000}"
fi
if [[ -z "$SYCO_SPLIT_EVAL_SIZE" ]]; then
    SYCO_SPLIT_EVAL_SIZE=400
fi
if [[ -z "$SYCO_SPLIT_SEED" ]]; then
    SYCO_SPLIT_SEED=1234
fi
if [[ -z "$SYCO_SPLIT_DIR" ]]; then
    if [[ "$MODEL_NAME" == "Qwen3.5-35B-A3B-Base" && "$SYCO_SPLIT_TRAIN_SIZE" == "5000" && -d "outputs/step4_feature_inject/dataset/$MODEL_NAME/split_train5000_eval400_seed1234_syco4000_alpaca1000" ]]; then
        SYCO_SPLIT_DIR="outputs/step4_feature_inject/dataset/$MODEL_NAME/split_train5000_eval400_seed1234_syco4000_alpaca1000"
    else
        SYCO_SPLIT_DIR="outputs/step4_feature_inject/dataset/$MODEL_NAME/split_train${SYCO_SPLIT_TRAIN_SIZE}_eval${SYCO_SPLIT_EVAL_SIZE}_seed${SYCO_SPLIT_SEED}"
    fi
fi

auto_run_name_for_group() {
    local group="$1"
    local beta="${2:-}"
    "$PYTHON_BIN" - "$CONFIG" "$MODEL_NAME" "$group" "$TUNING_MODE" "$EPOCHS" "$GLOBAL_BATCH_SIZE" "$LEARNING_RATE" "$MAX_STEPS" "$FEATURE_ID" "$LORA_RANK" "$beta" <<'PY'
import sys
import yaml

(
    config_path,
    model_name,
    group,
    tuning_mode,
    epochs_arg,
    global_bs_arg,
    lr_arg,
    max_steps_arg,
    feature_id_arg,
    lora_rank_arg,
    beta_arg,
) = sys.argv[1:]

def fmt(value):
    try:
        text = f"{float(value):g}"
    except (TypeError, ValueError):
        text = str(value)
    text = text.replace("E", "e")
    if "e" in text:
        mantissa, exponent = text.split("e", 1)
        sign = ""
        if exponent.startswith(("+", "-")):
            sign = exponent[0]
            exponent = exponent[1:]
        exponent = exponent.lstrip("0") or "0"
        text = f"{mantissa}e{sign}{exponent}"
    if text.startswith("-"):
        text = "neg" + text[1:]
    return text.replace(".", "p")

with open(config_path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
group_cfg = (cfg.get("experiments") or {}).get(group)
if not group_cfg:
    raise SystemExit(f"Unknown Step 4 group: {group}")

train_cfg = cfg.setdefault("training", {})
lora_cfg = cfg.setdefault("lora", {})
if tuning_mode:
    train_cfg["tuning_mode"] = tuning_mode
if epochs_arg:
    train_cfg["epochs"] = float(epochs_arg)
if global_bs_arg:
    train_cfg["global_batch_size"] = int(global_bs_arg)
if lr_arg:
    train_cfg["learning_rate"] = float(lr_arg)
if max_steps_arg:
    train_cfg["max_steps"] = int(max_steps_arg)
if lora_rank_arg:
    lora_cfg["rank"] = int(lora_rank_arg)

if beta_arg and group_cfg.get("injection", "none") != "none":
    beta = float(beta_arg)
    group_cfg["beta"] = beta
    group_cfg["beta_max"] = beta

if feature_id_arg:
    fid = int(feature_id_arg)
    feature = cfg.setdefault("feature", {})
    feature["feature_id"] = fid
    feature["feature_ids"] = [fid]
    feature.setdefault("by_model", {}).setdefault(model_name, {})["feature_id"] = fid
    feature["by_model"][model_name]["feature_ids"] = [fid]

parts = [group]
if group_cfg.get("injection", "none") != "none":
    feature = dict(cfg.get("feature") or {})
    feature.update((cfg.get("feature", {}).get("by_model") or {}).get(model_name, {}))
    if group_cfg.get("injection") == "random":
        fids = [int(group_cfg.get("random_feature_id", feature.get("random_feature_id", 0)))]
    else:
        fids = [int(x) for x in feature.get("feature_ids", [])] or [int(feature["feature_id"])]
    beta = float(group_cfg.get("beta", group_cfg.get("beta_max", 0.0)))
    parts.extend(["f" + "-".join(str(x) for x in fids), "alpha" + fmt(beta)])
mode = str(train_cfg.get("tuning_mode", "full")).lower()
parts.append(mode)
lr = train_cfg.get("learning_rate")
if lr is not None:
    parts.append("lr" + fmt(lr))
max_steps = train_cfg.get("max_steps")
if max_steps is not None:
    parts.append("steps" + fmt(max_steps))
else:
    parts.append("ep" + fmt(train_cfg.get("epochs", 1)))
parts.append("gbs" + fmt(train_cfg.get("global_batch_size", 1)))
if mode == "lora":
    parts.append("r" + fmt(lora_cfg.get("rank", 0)))
print("_".join(parts))
PY
}

run_is_complete() {
    local out_dir="$1"
    local expected_mode="$2"
    local group="${3:-}"
    [[ -s "$out_dir/training_summary.json" && \
       -s "$out_dir/loss_history.csv" ]] || return 1
    "$PYTHON_BIN" - "$out_dir" "$expected_mode" <<'PY' || return 1
import json
import os
import sys

out_dir, expected_mode = sys.argv[1:]
with open(os.path.join(out_dir, "training_summary.json"), encoding="utf-8") as f:
    summary = json.load(f)
mode = str(summary.get("tuning_mode") or "lora").lower()
if mode != expected_mode:
    raise SystemExit(1)
if mode == "lora":
    ok = (
        os.path.getsize(os.path.join(out_dir, "adapter_model.pt")) > 0
        and os.path.getsize(os.path.join(out_dir, "adapter_config.json")) > 0
    )
else:
    full_save = summary.get("full_model_save") or {}
    if bool(full_save.get("full_model_eval_ready")):
        ok = True
    else:
        ckpt_dir = os.path.join(out_dir, "deepspeed_checkpoint")
        latest_path = os.path.join(ckpt_dir, "latest")
        tag = ""
        if os.path.exists(latest_path):
            with open(latest_path, encoding="utf-8") as f:
                tag = f.read().strip()
        tag_dir = os.path.join(ckpt_dir, tag) if tag else ""
        if os.path.isdir(tag_dir):
            model_count = len([x for x in os.listdir(tag_dir) if x.endswith("_model_states.pt")])
            optim_count = len([x for x in os.listdir(tag_dir) if x.endswith("_optim_states.pt")])
        else:
            model_count = 0
            optim_count = 0
        expected = int(summary.get("world_size") or 0)
        if expected > 1:
            ok = model_count == expected and optim_count == expected
        else:
            ok = model_count > 0 and model_count == optim_count
raise SystemExit(0 if ok else 1)
PY
}

if [[ -n "$EXPERIMENT_GROUPS" ]]; then
    if [[ -n "$OUTPUT_DIR" ]]; then
        echo "ERROR: --output-dir is only valid for a single --group run; use --output-root with --groups." >&2
        exit 1
    fi
    if [[ -z "${STEP4_LOG_ACTIVE:-}" ]]; then
        LOG_DIR="${STEP4_LOG_DIR:-logs/step4_syco_pipeline}"
        mkdir -p "$LOG_DIR"
        SAFE_MODEL="${MODEL_NAME//\//_}"
        LOG_FILE="$LOG_DIR/step4_${SAFE_MODEL}_multi_$(date +%Y%m%d_%H%M%S).log"
        export STEP4_LOG_ACTIVE=1
        export STEP4_LOG_FILE="$LOG_FILE"
        exec > >(tee -a "$LOG_FILE") 2>&1
    fi

    "$PYTHON_BIN" - "$CONFIG" "$EXPERIMENT_GROUPS" <<'PY'
import sys
import yaml

config_path, groups_text = sys.argv[1:]
with open(config_path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
valid = set((cfg.get("experiments") or {}).keys())
requested = groups_text.split()
bad = [group for group in requested if group not in valid]
if bad:
    raise SystemExit(
        "Unknown Step 4 group(s): "
        + ", ".join(bad)
        + ". Available: "
        + ", ".join(sorted(valid))
    )
PY

    echo ""
    echo "================================================================"
    echo "  [step4] multi-run SFT"
    echo "  model : $MODEL_NAME"
    echo "  path  : $MODEL_PATH"
    echo "  tuning: $TUNING_MODE"
    echo "  groups: $EXPERIMENT_GROUPS"
    echo "  betas : $BETA_VALUES"
    echo "  output root: $OUTPUT_ROOT"
    echo "  gpus  : $NUM_GPUS ($CUDA_VISIBLE_DEVICES)"
    echo "  torch extensions: $TORCH_EXTENSIONS_DIR"
    echo "  log   : ${STEP4_LOG_FILE:-}"
    echo "================================================================"

    for EXPERIMENT_GROUP in $EXPERIMENT_GROUPS; do
        if [[ "$EXPERIMENT_GROUP" == "syco_sft" || "$EXPERIMENT_GROUP" == "objective_sft" ]]; then
            RUN_BETAS="0"
        else
            RUN_BETAS="$BETA_VALUES"
        fi
        for RUN_BETA in $RUN_BETAS; do
            RUN_NAME="$(auto_run_name_for_group "$EXPERIMENT_GROUP" "$RUN_BETA")"
            TARGET_OUTPUT_DIR="$OUTPUT_ROOT/$MODEL_NAME/$RUN_NAME"
            if [[ "$SKIP_COMPLETED" == "1" ]] && run_is_complete "$TARGET_OUTPUT_DIR" "$TUNING_MODE" "$EXPERIMENT_GROUP"; then
                echo "  [step4] skip completed: $TARGET_OUTPUT_DIR"
                continue
            fi
            RUN_ARGS=(
                --model "$MODEL_NAME"
                --model-path "$MODEL_PATH"
                --config "$CONFIG"
                --group "$EXPERIMENT_GROUP"
                --beta "$RUN_BETA"
                --tuning-mode "$TUNING_MODE"
                --output-root "$OUTPUT_ROOT"
                --gpu "$CUDA_VISIBLE_DEVICES"
                --num-gpus "$NUM_GPUS"
                --deepspeed-config "$DEEPSPEED_CONFIG"
                --injection-targets "$INJECTION_TARGETS"
            )
            if [[ -n "$FEATURE_ID" ]]; then
                RUN_ARGS+=(--feature-id "$FEATURE_ID")
            fi
            if [[ -n "$LAYER" ]]; then
                RUN_ARGS+=(--layer "$LAYER")
            fi
            if [[ -n "$SAE_DIR" ]]; then
                RUN_ARGS+=(--sae-dir "$SAE_DIR")
            fi
            if [[ -n "$BETA_SCHEDULE" ]]; then
                RUN_ARGS+=(--beta-schedule "$BETA_SCHEDULE")
            fi
            if [[ -n "$INJECTION_FRONT_TOKEN_COUNT" ]]; then
                RUN_ARGS+=(--injection-front-token-count "$INJECTION_FRONT_TOKEN_COUNT")
            fi
            if [[ -n "$SHARED_SYCO_DATASET_PATH" ]]; then
                RUN_ARGS+=(--shared-syco-dataset-path "$SHARED_SYCO_DATASET_PATH")
            fi
            if [[ -n "$SYCO_SPLIT_DIR" ]]; then
                RUN_ARGS+=(--syco-split-dir "$SYCO_SPLIT_DIR")
            fi
            if [[ -n "$SYCO_SPLIT_TRAIN_SIZE" ]]; then
                RUN_ARGS+=(--syco-split-train-size "$SYCO_SPLIT_TRAIN_SIZE")
            fi
            if [[ -n "$SYCO_SPLIT_EVAL_SIZE" ]]; then
                RUN_ARGS+=(--syco-split-eval-size "$SYCO_SPLIT_EVAL_SIZE")
            fi
            if [[ -n "$SYCO_SPLIT_SEED" ]]; then
                RUN_ARGS+=(--syco-split-seed "$SYCO_SPLIT_SEED")
            fi
            if [[ -n "$OBJECTIVE_TRAIN_PATH" ]]; then
                RUN_ARGS+=(--objective-train-path "$OBJECTIVE_TRAIN_PATH")
            fi
            if [[ -n "$AHC_TRAIN_PATH" ]]; then
                RUN_ARGS+=(--ahc-train-path "$AHC_TRAIN_PATH")
            fi
            if [[ "$ALLOW_MISSING_AHC" -eq 1 ]]; then
                RUN_ARGS+=(--allow-missing-ahc)
            fi
            if [[ -n "$EPOCHS" ]]; then
                RUN_ARGS+=(--epochs "$EPOCHS")
            fi
            if [[ -n "$BATCH_SIZE" ]]; then
                RUN_ARGS+=(--batch-size "$BATCH_SIZE")
            fi
            if [[ -n "$GLOBAL_BATCH_SIZE" ]]; then
                RUN_ARGS+=(--global-batch-size "$GLOBAL_BATCH_SIZE")
            fi
            if [[ -n "$GRADIENT_ACCUMULATION_STEPS" ]]; then
                RUN_ARGS+=(--gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS")
            fi
            if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
                RUN_ARGS+=(--gradient-checkpointing)
            elif [[ "$GRADIENT_CHECKPOINTING" == "0" ]]; then
                RUN_ARGS+=(--no-gradient-checkpointing)
            fi
            if [[ -n "$LEARNING_RATE" ]]; then
                RUN_ARGS+=(--learning-rate "$LEARNING_RATE")
            fi
            if [[ -n "$WARMUP_RATIO" ]]; then
                RUN_ARGS+=(--warmup-ratio "$WARMUP_RATIO")
            fi
            if [[ -n "$WARMUP_STEPS" ]]; then
                RUN_ARGS+=(--warmup-steps "$WARMUP_STEPS")
            fi
            if [[ -n "$MAX_STEPS" ]]; then
                RUN_ARGS+=(--max-steps "$MAX_STEPS")
            fi
            if [[ -n "$MAX_LENGTH" ]]; then
                RUN_ARGS+=(--max-length "$MAX_LENGTH")
            fi
            if [[ -n "$FULL_MODEL_EXPORT" ]]; then
                RUN_ARGS+=(--full-model-export "$FULL_MODEL_EXPORT")
            fi
            if [[ -n "$LORA_RANK" ]]; then
                RUN_ARGS+=(--lora-rank "$LORA_RANK")
            fi
            if [[ -n "$LORA_ALPHA" ]]; then
                RUN_ARGS+=(--lora-alpha "$LORA_ALPHA")
            fi
            if [[ -n "$LORA_DROPOUT" ]]; then
                RUN_ARGS+=(--lora-dropout "$LORA_DROPOUT")
            fi
            if [[ -n "$SYCO_FRONT_TOKEN_COUNT" ]]; then
                RUN_ARGS+=(--syco-front-token-count "$SYCO_FRONT_TOKEN_COUNT")
            fi
            if [[ -n "$SYCO_FRONT_TOKEN_WEIGHT" ]]; then
                RUN_ARGS+=(--syco-front-token-weight "$SYCO_FRONT_TOKEN_WEIGHT")
            fi
            if [[ "$NO_DEEPSPEED" -eq 1 ]]; then
                RUN_ARGS+=(--no-deepspeed)
            fi
            if [[ "$SKIP_COMPLETED" == "1" ]]; then
                RUN_ARGS+=(--skip-completed)
            fi
            bash "$0" "${RUN_ARGS[@]}" "${EXTRA_ARGS[@]}"
        done
    done
    echo ""
    echo "All requested Step 4 SFT runs completed."
    echo "Results: $OUTPUT_ROOT/$MODEL_NAME/"
    exit 0
fi

if [[ "$USER_RUN_NAME" -eq 1 && "$USER_OUTPUT_DIR" -eq 0 ]]; then
    OUTPUT_DIR="$OUTPUT_ROOT/$MODEL_NAME/$RUN_NAME"
fi

if [[ -z "${STEP4_LOG_ACTIVE:-}" ]]; then
    LOG_DIR="${STEP4_LOG_DIR:-logs/step4_syco_pipeline}"
    mkdir -p "$LOG_DIR"
    SAFE_MODEL="${MODEL_NAME//\//_}"
    SAFE_GROUP="${GROUP//\//_}"
    LOG_FILE="$LOG_DIR/step4_${SAFE_MODEL}_${SAFE_GROUP}_$(date +%Y%m%d_%H%M%S).log"
    export STEP4_LOG_ACTIVE=1
    export STEP4_LOG_FILE="$LOG_FILE"
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

"$PYTHON_BIN" - "$CONFIG" "$GROUP" <<'PY'
import sys
import yaml

config_path, group = sys.argv[1:]
with open(config_path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
valid = set((cfg.get("experiments") or {}).keys())
if group not in valid:
    raise SystemExit(
        f"Unknown Step 4 group: {group}. Available: {', '.join(sorted(valid))}"
    )
PY

TMP_CONFIG="$CONFIG"
if [[ -n "$FEATURE_ID" || -n "$LAYER" || -n "$SAE_DIR" ]]; then
    TMP_CONFIG="tmp/step4_${MODEL_NAME}_${GROUP}_$(date +%Y%m%d_%H%M%S%N)_pid$$.yaml"
    mkdir -p tmp
    "$PYTHON_BIN" - "$CONFIG" "$TMP_CONFIG" "$MODEL_NAME" "$FEATURE_ID" "$LAYER" "$SAE_DIR" <<'PY'
import sys
import yaml

src, dst, model_name, feature_id, layer, sae_dir = sys.argv[1:]
with open(src, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

if feature_id:
    fid = int(feature_id)
    cfg["feature"]["feature_id"] = fid
    cfg["feature"]["feature_ids"] = [fid]
    cfg["feature"].setdefault("by_model", {}).setdefault(model_name, {})["feature_id"] = fid
    cfg["feature"]["by_model"][model_name]["feature_ids"] = [fid]

if layer or sae_dir:
    sae_cfg = cfg.setdefault("sae", {}).setdefault("configs", {}).setdefault(model_name, {})
    if layer:
        sae_cfg["layer"] = int(layer)
    if sae_dir:
        sae_cfg["sae_dir"] = sae_dir

with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
fi

COMMON_ARGS=(
    --config "$TMP_CONFIG"
    --model-name "$MODEL_NAME"
    --model-path "$MODEL_PATH"
    --group "$GROUP"
    --tuning-mode "$TUNING_MODE"
    --output-root "$OUTPUT_ROOT"
    --injection-targets "$INJECTION_TARGETS"
)

if [[ -n "$BETA" ]]; then
    COMMON_ARGS+=(--beta "$BETA")
fi
if [[ -n "$BETA_SCHEDULE" ]]; then
    COMMON_ARGS+=(--beta-schedule "$BETA_SCHEDULE")
fi
if [[ -n "$INJECTION_FRONT_TOKEN_COUNT" ]]; then
    COMMON_ARGS+=(--injection-front-token-count "$INJECTION_FRONT_TOKEN_COUNT")
fi
if [[ "$NO_DEEPSPEED" -eq 0 ]]; then
    COMMON_ARGS+=(--deepspeed-config "$DEEPSPEED_CONFIG")
else
    COMMON_ARGS+=(--deepspeed-config "")
fi
if [[ -n "$OUTPUT_DIR" ]]; then
    COMMON_ARGS+=(--output-dir "$OUTPUT_DIR")
fi
if [[ -n "$RUN_NAME" ]]; then
    COMMON_ARGS+=(--run-name "$RUN_NAME")
fi
if [[ -n "$SHARED_SYCO_DATASET_PATH" ]]; then
    COMMON_ARGS+=(--shared-syco-dataset-path "$SHARED_SYCO_DATASET_PATH")
fi
if [[ -n "$SYCO_SPLIT_DIR" ]]; then
    COMMON_ARGS+=(--syco-split-dir "$SYCO_SPLIT_DIR")
fi
if [[ -n "$SYCO_SPLIT_TRAIN_SIZE" ]]; then
    COMMON_ARGS+=(--syco-split-train-size "$SYCO_SPLIT_TRAIN_SIZE")
fi
if [[ -n "$SYCO_SPLIT_EVAL_SIZE" ]]; then
    COMMON_ARGS+=(--syco-split-eval-size "$SYCO_SPLIT_EVAL_SIZE")
fi
if [[ -n "$SYCO_SPLIT_SEED" ]]; then
    COMMON_ARGS+=(--syco-split-seed "$SYCO_SPLIT_SEED")
fi
if [[ -n "$OBJECTIVE_TRAIN_PATH" ]]; then
    COMMON_ARGS+=(--objective-train-path "$OBJECTIVE_TRAIN_PATH")
fi
if [[ -n "$AHC_TRAIN_PATH" ]]; then
    COMMON_ARGS+=(--ahc-train-path "$AHC_TRAIN_PATH")
fi
if [[ "$ALLOW_MISSING_AHC" -eq 1 ]]; then
    COMMON_ARGS+=(--allow-missing-ahc)
fi
if [[ -n "$EPOCHS" ]]; then
    COMMON_ARGS+=(--epochs "$EPOCHS")
fi
if [[ -n "$BATCH_SIZE" ]]; then
    COMMON_ARGS+=(--batch-size "$BATCH_SIZE")
fi
if [[ -n "$GLOBAL_BATCH_SIZE" ]]; then
    COMMON_ARGS+=(--global-batch-size "$GLOBAL_BATCH_SIZE")
fi
if [[ -n "$GRADIENT_ACCUMULATION_STEPS" ]]; then
    COMMON_ARGS+=(--gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS")
fi
if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
    COMMON_ARGS+=(--gradient-checkpointing)
elif [[ "$GRADIENT_CHECKPOINTING" == "0" ]]; then
    COMMON_ARGS+=(--no-gradient-checkpointing)
fi
if [[ -n "$LEARNING_RATE" ]]; then
    COMMON_ARGS+=(--learning-rate "$LEARNING_RATE")
fi
if [[ -n "$WARMUP_RATIO" ]]; then
    COMMON_ARGS+=(--warmup-ratio "$WARMUP_RATIO")
fi
if [[ -n "$WARMUP_STEPS" ]]; then
    COMMON_ARGS+=(--warmup-steps "$WARMUP_STEPS")
fi
if [[ -n "$MAX_STEPS" ]]; then
    COMMON_ARGS+=(--max-steps "$MAX_STEPS")
fi
if [[ -n "$MAX_LENGTH" ]]; then
    COMMON_ARGS+=(--max-length "$MAX_LENGTH")
fi
if [[ -n "$MAX_TRAIN_EXAMPLES" ]]; then
    COMMON_ARGS+=(--max-train-examples "$MAX_TRAIN_EXAMPLES")
fi
if [[ -n "$MAX_SYCO_EXAMPLES" ]]; then
    COMMON_ARGS+=(--max-syco-examples "$MAX_SYCO_EXAMPLES")
fi
if [[ -n "$MAX_HARM_EXAMPLES" ]]; then
    COMMON_ARGS+=(--max-harm-examples "$MAX_HARM_EXAMPLES")
fi
if [[ -n "$FULL_MODEL_EXPORT" ]]; then
    COMMON_ARGS+=(--full-model-export "$FULL_MODEL_EXPORT")
fi
if [[ -n "$LORA_RANK" ]]; then
    COMMON_ARGS+=(--lora-rank "$LORA_RANK")
fi
if [[ -n "$LORA_ALPHA" ]]; then
    COMMON_ARGS+=(--lora-alpha "$LORA_ALPHA")
fi
if [[ -n "$LORA_DROPOUT" ]]; then
    COMMON_ARGS+=(--lora-dropout "$LORA_DROPOUT")
fi
if [[ -n "$SYCO_FRONT_TOKEN_COUNT" ]]; then
    COMMON_ARGS+=(--syco-front-token-count "$SYCO_FRONT_TOKEN_COUNT")
fi
if [[ -n "$SYCO_FRONT_TOKEN_WEIGHT" ]]; then
    COMMON_ARGS+=(--syco-front-token-weight "$SYCO_FRONT_TOKEN_WEIGHT")
fi

echo "  [step4] training defaults/resolved: epochs=$EPOCHS global_batch_size=$GLOBAL_BATCH_SIZE warmup_ratio=${WARMUP_RATIO:-none} warmup_steps=${WARMUP_STEPS:-none}"

echo ""
echo "================================================================"
echo "  [step4] model : $MODEL_NAME"
echo "  [step4] path  : $MODEL_PATH"
echo "  [step4] tuning: $TUNING_MODE"
echo "  [step4] group : $GROUP"
if [[ -n "$BETA" ]]; then
    echo "  [step4] beta  : $BETA"
fi
echo "  [step4] config: $TMP_CONFIG"
echo "  [step4] log   : ${STEP4_LOG_FILE:-}"
echo "  [step4] output root: $OUTPUT_ROOT"
if [[ -n "$OUTPUT_DIR" ]]; then
    echo "  [step4] output: $OUTPUT_DIR"
fi
if [[ -n "$RUN_NAME" ]]; then
    echo "  [step4] run name: $RUN_NAME"
fi
if [[ -n "$OBJECTIVE_TRAIN_PATH" ]]; then
    echo "  [step4] objective train path: $OBJECTIVE_TRAIN_PATH"
fi
if [[ -n "$AHC_TRAIN_PATH" ]]; then
    echo "  [step4] AHC train path: $AHC_TRAIN_PATH"
fi
if [[ "$ALLOW_MISSING_AHC" -eq 1 ]]; then
    echo "  [step4] allow missing AHC: yes"
fi
if [[ -n "$EPOCHS" ]]; then
    echo "  [step4] epochs: $EPOCHS"
fi
if [[ -n "$BATCH_SIZE" ]]; then
    echo "  [step4] micro batch: $BATCH_SIZE"
fi
if [[ -n "$GLOBAL_BATCH_SIZE" ]]; then
    echo "  [step4] global batch: $GLOBAL_BATCH_SIZE"
fi
if [[ -n "$GRADIENT_ACCUMULATION_STEPS" ]]; then
    echo "  [step4] grad accum: $GRADIENT_ACCUMULATION_STEPS"
fi
if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
    echo "  [step4] gradient checkpointing: enabled"
elif [[ "$GRADIENT_CHECKPOINTING" == "0" ]]; then
    echo "  [step4] gradient checkpointing: disabled"
fi
if [[ -n "$LEARNING_RATE" ]]; then
    echo "  [step4] lr: $LEARNING_RATE"
fi
if [[ -n "$WARMUP_RATIO" ]]; then
    echo "  [step4] warmup ratio: $WARMUP_RATIO"
fi
if [[ -n "$WARMUP_STEPS" ]]; then
    echo "  [step4] warmup steps: $WARMUP_STEPS"
fi
if [[ -n "$MAX_STEPS" ]]; then
    echo "  [step4] max steps: $MAX_STEPS"
fi
if [[ -n "$MAX_LENGTH" ]]; then
    echo "  [step4] max length: $MAX_LENGTH"
fi
if [[ -n "$FULL_MODEL_EXPORT" ]]; then
    echo "  [step4] full model export: $FULL_MODEL_EXPORT"
fi
if [[ -n "$LORA_RANK" ]]; then
    echo "  [step4] lora rank: $LORA_RANK"
fi
if [[ -n "$LORA_ALPHA" ]]; then
    echo "  [step4] lora alpha: $LORA_ALPHA"
fi
if [[ -n "$LORA_DROPOUT" ]]; then
    echo "  [step4] lora dropout: $LORA_DROPOUT"
fi
if [[ -n "$BETA_SCHEDULE" ]]; then
    echo "  [step4] beta schedule: $BETA_SCHEDULE"
fi
echo "  [step4] injection targets: $INJECTION_TARGETS"
echo "  [step4] injection front tokens: ${INJECTION_FRONT_TOKEN_COUNT:-all}"
if [[ "$NO_DEEPSPEED" -eq 0 ]]; then
    echo "  [step4] ds    : $DEEPSPEED_CONFIG"
    if [[ "$TUNING_MODE" == "full" ]]; then
        echo "  [step4] note  : full tuning uses the ZeRO-3/offload policy declared in the selected config"
    elif [[ "$MODEL_NAME" == "Qwen3.5-35B-A3B-Base" ]]; then
        echo "  [step4] note  : Python may switch this 35B LoRA run to ZeRO-2 + micro=1 for memory safety"
    fi
    echo "  [step4] gpus  : $NUM_GPUS ($CUDA_VISIBLE_DEVICES)"
else
    echo "  [step4] ds    : disabled"
fi
if [[ -n "$FEATURE_ID" ]]; then
    echo "  [step4] feature override: $FEATURE_ID"
fi
if [[ -n "$LAYER" ]]; then
    echo "  [step4] layer override  : $LAYER"
fi
echo "  [step4] python: $PYTHON_BIN"
echo "  [step4] torch extensions: $TORCH_EXTENSIONS_DIR"
echo "================================================================"

if [[ "$NO_DEEPSPEED" -eq 0 ]]; then
    if is_multinode_torchrun; then
        NNODES="${NNODES:-${PET_NNODES:-${DLC_NNODES:-1}}}"
        NODE_RANK="${NODE_RANK:-$(infer_node_rank)}"
        MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-${PADDLE_MASTER:-}}}"
        MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-29500}}"
        if [[ -z "$MASTER_ADDR" ]]; then
            echo "ERROR: multi-node launch requires MASTER_ADDR or PET_MASTER_ADDR." >&2
            exit 1
        fi
        VISIBLE_GPU_COUNT="$(count_visible_gpus "$CUDA_VISIBLE_DEVICES")"
        if [[ "$VISIBLE_GPU_COUNT" -ne "$NUM_GPUS" ]]; then
            echo "ERROR: --num-gpus=$NUM_GPUS but CUDA_VISIBLE_DEVICES exposes $VISIBLE_GPU_COUNT local GPU(s): $CUDA_VISIBLE_DEVICES" >&2
            exit 1
        fi
        if [[ ! -x "$TORCHRUN_BIN" ]]; then
            TORCHRUN_BIN="$(command -v torchrun || true)"
        fi
        if [[ -z "$TORCHRUN_BIN" || ! -x "$TORCHRUN_BIN" ]]; then
            echo "ERROR: torchrun not found. Set TORCHRUN_BIN or install torch distributed launcher." >&2
            exit 1
        fi
        echo "  [step4] using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        echo "  [step4] torchrun: nnodes=$NNODES node_rank=$NODE_RANK nproc_per_node=$NUM_GPUS master=$MASTER_ADDR:$MASTER_PORT"
        "$TORCHRUN_BIN" \
            --nnodes "$NNODES" \
            --node_rank "$NODE_RANK" \
            --nproc_per_node "$NUM_GPUS" \
            --master_addr "$MASTER_ADDR" \
            --master_port "$MASTER_PORT" \
            src/step4_vaccine.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
        exit $?
    fi
    DEEPSPEED_LAUNCH_ARGS=()
    if [[ -z "${MASTER_PORT:-}" ]]; then
        MASTER_PORT="$("$PYTHON_BIN" - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1", 0))
    print(s.getsockname()[1])
PY
)"
    fi
    DEEPSPEED_LAUNCH_ARGS+=(--master_port "$MASTER_PORT")
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -ra VISIBLE_GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
        VISIBLE_GPU_COUNT=0
        for gpu_id in "${VISIBLE_GPU_IDS[@]}"; do
            if [[ -n "${gpu_id//[[:space:]]/}" ]]; then
                VISIBLE_GPU_COUNT=$((VISIBLE_GPU_COUNT + 1))
            fi
        done
        if [[ "$VISIBLE_GPU_COUNT" -ne "$NUM_GPUS" ]]; then
            echo "ERROR: --num-gpus=$NUM_GPUS but CUDA_VISIBLE_DEVICES exposes $VISIBLE_GPU_COUNT GPU(s): $CUDA_VISIBLE_DEVICES" >&2
            exit 1
        fi
        echo "  [step4] using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        echo "  [step4] master port: $MASTER_PORT"
    else
        DEEPSPEED_LAUNCH_ARGS+=(--num_gpus "$NUM_GPUS")
        echo "  [step4] master port: $MASTER_PORT"
    fi
    "$DEEPSPEED_BIN" "${DEEPSPEED_LAUNCH_ARGS[@]}" src/step4_vaccine.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
else
    "$PYTHON_BIN" -m src.step4_vaccine "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
fi
