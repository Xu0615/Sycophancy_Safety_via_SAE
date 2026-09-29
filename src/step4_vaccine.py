"""Step 4 — Sycophancy SFT with optional train-time SAE injection.

This module implements the post-train preventative steering pipeline described
in ``posttrain.md``.  During SFT, it can inject the positive sycophancy SAE
decoder direction into response-token residual states at a target layer:

    h_l[:, assistant_tokens, :] <- h_l[:, assistant_tokens, :] + beta * v_syc

The training hook is removed before saving/evaluation.  The launcher supports
full-parameter fine-tuning and LoRA; neither path requires inference-time hooks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

try:
    import deepspeed
    from deepspeed import zero
    from transformers.integrations import HfDeepSpeedConfig
except Exception as exc:  # pragma: no cover - validated at runtime when requested.
    deepspeed = None
    zero = None
    HfDeepSpeedConfig = None
    _DEEPSPEED_IMPORT_ERROR = exc
else:
    _DEEPSPEED_IMPORT_ERROR = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from step3_steering import build_steering_vector, load_sae_decoder  # noqa: E402
from step4_syco_split import (  # noqa: E402
    DEFAULT_EVAL_SIZE,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TRAIN_SIZE,
    ensure_shared_syco_sft_split,
)
from step4_lora import (  # noqa: E402
    LoRAConfig,
    inject_lora_adapters,
    merge_lora_adapters,
    trainable_parameter_summary,
)


logger = logging.getLogger(__name__)
IGNORE_INDEX = -100
TUNING_MODE_FULL = "full"
TUNING_MODE_LORA = "lora"
SUPPORTED_TUNING_MODES = {TUNING_MODE_FULL, TUNING_MODE_LORA}
TARGET_TYPE_SYCO = "syco"
TARGET_TYPE_INSTRUCTION = "instruction"
SYCO_RESPONSE_TYPES = {"sycophantic", "syco"}
INSTRUCTION_RESPONSE_TYPES = {"instruction", "neutral_instruction", "alpaca", "alpaca_cleaned"}
DEFAULT_INJECTION_TARGETS = (TARGET_TYPE_SYCO,)
FULL_TUNING_SAFE_DEEPSPEED_CONFIG = "configs/deepspeed_step4_zero3_35b_512_safe.json"
FULL_TUNING_SAFE_DEEPSPEED_CONFIGS = {
    "configs/deepspeed_step4_zero3_35b_512_safe.json",
    "configs/deepspeed_step4_zero3_35b_512_fast.json",
    "configs/deepspeed_step4_zero3_35b_512_16gpu_no_offload.json",
}
FULL_TUNING_NO_OFFLOAD_DEEPSPEED_CONFIGS = {
    "configs/deepspeed_step4_zero3_35b_512_16gpu_no_offload.json",
}
LORA_LARGE_MODEL_SAFE_DEEPSPEED_CONFIG = "configs/deepspeed_step4_zero2.json"

# The historical 35B main result used a 5,000-row mixed split (4,000 syco +
# 1,000 Alpaca).  ``recover_step4_historical_datasets`` installs a metadata-only
# package when those 4,000 rows are unavailable.  Keep the status string here so
# the training entry point fails with the actual provenance problem instead of
# falling through to a generic missing-file or wrong-split error.
HISTORICAL_35B_MODEL = "Qwen3.5-35B-A3B-Base"
BLOCKED_HISTORICAL_35B_STATUS = "blocked_missing_historical_syco_train4000"
HISTORICAL_SMALL_SPLIT_STATUS_PREFIX = "complete_for_train1100_eval400"


@dataclass
class RuntimeContext:
    """Distributed runtime metadata for a single DeepSpeed or local process."""

    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def print_main(runtime: RuntimeContext, *args, **kwargs) -> None:
    if runtime.is_main:
        print(*args, **kwargs)


def barrier(runtime: RuntimeContext) -> None:
    if runtime.distributed and torch.distributed.is_initialized():
        torch.distributed.barrier()


def setup_file_logger(log_file: str) -> None:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
        force=True,
    )


def setup_rank_logger(log_file: str, rank: int) -> None:
    """Log rank 0 to file and stderr, and keep worker ranks quiet."""

    if rank == 0:
        setup_file_logger(log_file)
        return
    logging.basicConfig(
        level=logging.WARNING,
        format=f"%(asctime)s | rank={rank} | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stderr)],
        force=True,
    )


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def torch_dtype_from_name(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def get_text_config(config):
    return getattr(config, "text_config", config)


def get_text_model(model):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model, "language_model"):
        return model.language_model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    if hasattr(model, "layers"):
        return model
    raise AttributeError("Could not locate language model layers")


def get_layer_module(model, layer_idx: int):
    text_model = get_text_model(model)
    return text_model.layers[int(layer_idx)]


def state_dict_export_model(model):
    """Return the smallest causal-LM module that retains its output head."""

    # AutoModelForCausalLM loads Qwen3.5 as a text-only CausalLM even when the
    # source checkpoint uses an outer multimodal config.  Its ``model`` child
    # is only the backbone; exporting that child silently drops the separately
    # trained ``lm_head`` when word embeddings are not tied.
    text_model = get_text_model(model)
    config = get_text_config(getattr(model, "config", None))
    if bool(getattr(config, "tie_word_embeddings", False)):
        return text_model
    if hasattr(model, "lm_head"):
        return model
    if hasattr(text_model, "lm_head"):
        return text_model
    return text_model


def set_trainable_lora_only(model) -> None:
    """Freeze base parameters and keep only LoRA matrices trainable."""

    for name, param in model.named_parameters():
        param.requires_grad_("lora_A" in name or "lora_B" in name)


def set_trainable_full_model(model) -> None:
    """Enable gradients for every model parameter."""

    for param in model.parameters():
        param.requires_grad_(True)


def resolve_tuning_mode(cfg: Dict[str, Any]) -> str:
    """Return the Step 4 tuning mode, defaulting to full-parameter tuning."""

    mode = str(cfg.setdefault("training", {}).get("tuning_mode", TUNING_MODE_FULL)).lower().strip()
    aliases = {
        "full_param": TUNING_MODE_FULL,
        "full-parameter": TUNING_MODE_FULL,
        "full_parameter": TUNING_MODE_FULL,
        "full_finetune": TUNING_MODE_FULL,
        "full-finetune": TUNING_MODE_FULL,
        "full_sft": TUNING_MODE_FULL,
        "adapter": TUNING_MODE_LORA,
        "peft": TUNING_MODE_LORA,
    }
    mode = aliases.get(mode, mode)
    if mode not in SUPPORTED_TUNING_MODES:
        raise ValueError(
            "training.tuning_mode must be one of "
            f"{sorted(SUPPORTED_TUNING_MODES)}; got {mode!r}"
        )
    cfg["training"]["tuning_mode"] = mode
    return mode


def is_lora_tuning(cfg: Dict[str, Any]) -> bool:
    return resolve_tuning_mode(cfg) == TUNING_MODE_LORA


def find_response_token_span(input_ids: Sequence[int], response_ids: Sequence[int]) -> Tuple[int, int]:
    """Find the first exact response-token span in a tokenized full example."""

    if not response_ids:
        raise ValueError("Empty response tokenization")
    n = len(response_ids)
    ids = list(input_ids)
    resp = list(response_ids)
    for start in range(0, len(ids) - n + 1):
        if ids[start:start + n] == resp:
            return start, start + n
    raise ValueError("Could not find response token span in chat-formatted sequence")


def extract_input_ids(tokenized: Any) -> List[int]:
    """Normalize tokenizer/chat-template outputs to a flat token-id list."""

    if isinstance(tokenized, dict) or hasattr(tokenized, "data"):
        ids = tokenized["input_ids"]
    else:
        ids = tokenized
    if torch.is_tensor(ids):
        ids = ids.detach().cpu().tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def format_chat_pair(tokenizer, prompt: str, response: str) -> Tuple[List[int], List[int], List[int]]:
    """Tokenize a user/assistant pair and return input ids, labels, assistant mask."""

    messages = [{"role": "user", "content": str(prompt)}]
    try:
        prefix_ids = extract_input_ids(tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        ))
    except TypeError:
        prefix_ids = extract_input_ids(tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        ))

    response_ids = tokenizer(str(response), add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        response_ids = response_ids + [int(tokenizer.eos_token_id)]

    input_ids = list(prefix_ids) + list(response_ids)
    labels = [IGNORE_INDEX] * len(prefix_ids) + list(response_ids)
    assistant_mask = [0] * len(prefix_ids) + [1] * len(response_ids)
    return input_ids, labels, assistant_mask


def truncate_example(
    input_ids: List[int],
    labels: List[int],
    assistant_mask: List[int],
    max_length: int,
) -> Tuple[List[int], List[int], List[int]]:
    """Left-truncate long examples while preserving aligned labels and masks."""

    if len(input_ids) <= max_length:
        return input_ids, labels, assistant_mask
    input_ids = input_ids[-max_length:]
    labels = labels[-max_length:]
    assistant_mask = assistant_mask[-max_length:]
    if all(label == IGNORE_INDEX for label in labels):
        raise ValueError("Truncation removed every supervised response token")
    return input_ids, labels, assistant_mask


@dataclass
class TrainExample:
    sample_id: str
    domain: str
    source: str
    prompt: str
    response: str
    target_type: str


def load_syco_response_examples(dataset_path: str) -> List[Dict[str, Any]]:
    rows = read_jsonl(dataset_path)
    out: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        row_response_type = str(row.get("response_type", "")).strip()
        if is_instruction_response_row(row):
            response_key = "instruction"
            target_type = TARGET_TYPE_INSTRUCTION
            source = str(row.get("source_mix_component") or row.get("source_dataset") or "instruction")
        elif row_response_type in SYCO_RESPONSE_TYPES:
            response_key = "sycophantic"
            target_type = TARGET_TYPE_SYCO
            source = str(row.get("source_mix_component") or row.get("source_dataset") or "paired_syco")
        else:
            continue
        prompt = row.get("prompt") or row.get("prompt_text", "")
        response = row.get("response", "")
        pair_id = str(row.get("id") or row.get("sample_id") or stable_hash(str(prompt)))
        if not prompt or not response or pair_id in seen:
            continue
        seen.add(pair_id)
        out.append({
            "id": pair_id,
            "domain": row.get("domain", ""),
            "source": source,
            "prompt": prompt,
            response_key: response,
            "target_type": target_type,
            "response_type": row_response_type,
        })
    return out


def load_objective_response_examples(dataset_path: str) -> List[Dict[str, Any]]:
    return load_response_examples(
        dataset_path,
        response_types={"robust", "objective"},
        response_key="objective",
        source="paired_objective",
    )


def load_response_examples(
    dataset_path: str,
    response_types: set[str],
    response_key: str,
    source: str,
) -> List[Dict[str, Any]]:
    rows = read_jsonl(dataset_path)
    out: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        row_response_type = row.get("response_type")
        if row_response_type not in response_types:
            continue
        prompt = row.get("prompt") or row.get("prompt_text", "")
        response = row.get("response", "")
        pair_id = str(row.get("id") or row.get("sample_id") or stable_hash(str(prompt)))
        if not prompt or not response or pair_id in seen:
            continue
        seen.add(pair_id)
        out.append({
            "id": pair_id,
            "domain": row.get("domain", ""),
            "source": source,
            "prompt": prompt,
            response_key: response,
        })
    return out


def is_instruction_response_row(row: Dict[str, Any]) -> bool:
    response_type = str(row.get("response_type", "")).strip()
    if response_type in INSTRUCTION_RESPONSE_TYPES:
        return True
    markers = (
        row.get("source_mix_component"),
        row.get("source_dataset"),
        row.get("domain"),
        row.get("id"),
        row.get("sample_id"),
    )
    return any(str(value or "").startswith("alpaca_cleaned") for value in markers)


def build_syco_examples(
    dataset_path: str,
    max_examples: Optional[int] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> List[TrainExample]:
    out = []
    pairs = load_syco_response_examples(dataset_path)
    for pair in pairs:
        target_type = str(pair.get("target_type", TARGET_TYPE_SYCO))
        response_key = "instruction" if target_type == TARGET_TYPE_INSTRUCTION else "sycophantic"
        out.append(
            TrainExample(
                sample_id=str(pair["id"]),
                domain=str(pair.get("domain", "")),
                source=str(pair.get("source", "paired_syco")),
                prompt=str(pair["prompt"]),
                response=str(pair[response_key]),
                target_type=target_type,
            )
        )
    if max_examples is not None:
        out = out[: int(max_examples)]
    return out


def build_objective_examples(
    dataset_path: str,
    max_examples: Optional[int] = None,
) -> List[TrainExample]:
    out = []
    pairs = load_objective_response_examples(dataset_path)
    for pair in pairs:
        out.append(
            TrainExample(
                sample_id=str(pair["id"]),
                domain=str(pair.get("domain", "")),
                source=str(pair.get("source", "paired_objective")),
                prompt=str(pair["prompt"]),
                response=str(pair["objective"]),
                target_type="objective",
            )
        )
    if max_examples is not None:
        out = out[: int(max_examples)]
    return out


SAFE_REFUSAL_RESPONSE = (
    "I can't help with instructions, plans, or content that would facilitate harm, "
    "deception, harassment, privacy invasion, or wrongdoing. If your goal is legitimate, "
    "I can help reframe it into a safe alternative, explain the risks, or suggest "
    "constructive next steps."
)


def make_safe_refusal_response(prompt: str) -> str:
    return SAFE_REFUSAL_RESPONSE


def build_harm_refusal_examples(
    ahc_path: str,
    max_examples: Optional[int] = None,
    allow_missing: bool = False,
) -> List[TrainExample]:
    if not os.path.exists(ahc_path):
        message = f"AHC training file not found: {ahc_path}"
        if not allow_missing:
            raise FileNotFoundError(
                message
                + ". target='objective_refusal' needs harmful refusal examples; "
                "generate Step1 AHC for this model, pass --ahc-train-path, or set "
                "data.allow_missing_ahc=true only for an intentional ablation."
            )
        logger.warning("%s; skipping refusal examples", message)
        return []
    df = pd.read_parquet(ahc_path)
    if "prompt_text" not in df.columns:
        raise ValueError(f"Bad AHC training file {ahc_path}; missing prompt_text")
    out: List[TrainExample] = []
    for _, row in df.iterrows():
        prompt = str(row.get("prompt_text", "")).strip()
        if not prompt:
            continue
        sample_id = str(row.get("sample_id") or stable_hash(prompt))
        domain = str(row.get("source_dataset", "harmful"))
        out.append(
            TrainExample(
                sample_id=f"{sample_id}::safe_refusal",
                domain=domain,
                source="ahc_refusal",
                prompt=prompt,
                response=(
                    str(row.get("refusal_response", "")).strip()
                    or make_safe_refusal_response(prompt)
                ),
                target_type="refusal",
            )
        )
    if max_examples is not None:
        out = out[: int(max_examples)]
    return out


def resolve_shared_syco_sft_source(cfg: Dict[str, Any], model_name: str) -> str:
    data_cfg = cfg.get("data", {})
    if data_cfg.get("shared_syco_dataset_path"):
        return data_cfg["shared_syco_dataset_path"]
    shared_root = data_cfg.get(
        "shared_syco_dataset_root",
        "outputs/step4_feature_inject/dataset",
    )
    dataset_dir = data_cfg.get("shared_syco_dataset_dirname")
    if dataset_dir:
        return os.path.join(shared_root, dataset_dir, "syco_dataset.jsonl")
    return os.path.join(shared_root, model_name, "syco_dataset.jsonl")


def load_syco_dataset_metadata(source_path: str) -> Tuple[Dict[str, Any], str]:
    """Load metadata next to a shared source, if present.

    Dataset recovery deliberately leaves a metadata-only directory for the
    unavailable 35B training source.  Reading that file before checking the
    JSONL path lets callers report the precise blocked state.
    """

    metadata_path = os.path.join(
        os.path.dirname(os.path.abspath(source_path)),
        "dataset_metadata.json",
    )
    if not os.path.exists(metadata_path):
        return {}, metadata_path
    try:
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid Step 4 syco dataset metadata: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Step 4 syco dataset metadata must be an object: {metadata_path}")
    return metadata, metadata_path


def resolve_syco_train_dataset(cfg: Dict[str, Any], model_name: str) -> Tuple[str, Dict[str, Any]]:
    data_cfg = cfg["data"]
    shared_syco_dataset = resolve_shared_syco_sft_source(cfg, model_name)
    dataset_metadata, metadata_path = load_syco_dataset_metadata(shared_syco_dataset)
    recovery_status = str(dataset_metadata.get("recovery_status") or "").strip()
    split_dir_hint = data_cfg.get("syco_split_dir")
    split_metadata: Dict[str, Any] = {}
    if split_dir_hint:
        split_metadata_path = os.path.join(os.path.abspath(str(split_dir_hint)), "split_metadata.json")
        if os.path.exists(split_metadata_path):
            try:
                with open(split_metadata_path, encoding="utf-8") as handle:
                    loaded_split_metadata = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid Step 4 syco split metadata: {split_metadata_path}") from exc
            if isinstance(loaded_split_metadata, dict):
                split_metadata = loaded_split_metadata
    missing_split_rows = int(split_metadata.get("missing_syco_train_rows") or 0)
    if (
        model_name == HISTORICAL_35B_MODEL
        and split_metadata.get("train_complete") is False
        and missing_split_rows > 0
    ):
        raise FileNotFoundError(
            "Step 4 training is blocked for "
            f"{model_name}: {split_metadata_path} declares an incomplete historical "
            f"split with {missing_split_rows} missing syco train rows. "
            "Use the correctly named 5,000/400 package only after the historical "
            "4,000 syco rows are supplied."
        )
    if (
        model_name == HISTORICAL_35B_MODEL
        and split_metadata
        and int(split_metadata.get("train_size") or 0) == 1100
    ):
        raise ValueError(
            "Refusing to train the 35B-A3B historical experiment from a 1,100-row "
            f"split ({split_metadata_path}). The 1,100-row recovery package belongs "
            "to 2B/9B; 35B requires 4,000 syco + 1,000 Alpaca rows."
        )
    if recovery_status == BLOCKED_HISTORICAL_35B_STATUS:
        missing_rows = int(dataset_metadata.get("missing_rows") or 4000)
        raise FileNotFoundError(
            "Step 4 training is blocked for "
            f"{model_name}: {metadata_path} records "
            f"{BLOCKED_HISTORICAL_35B_STATUS} ({missing_rows} historical syco rows missing). "
            "The 35B protocol requires 4,000 syco + 1,000 Alpaca train rows; "
            "the local package contains only the verifiable 400-row holdout and "
            "1,000 Alpaca rows. Supply the historical syco source before training; "
            "do not substitute the 2B/9B split_train1100_eval400_seed1234 package."
        )
    if (
        model_name == HISTORICAL_35B_MODEL
        and recovery_status.startswith(HISTORICAL_SMALL_SPLIT_STATUS_PREFIX)
    ):
        raise ValueError(
            "Refusing to train the 35B-A3B historical experiment from a "
            f"{recovery_status} dataset ({shared_syco_dataset}). "
            "35B requires split_train5000_eval400_seed1234_syco4000_alpaca1000; "
            "the 1,100-row recovery package is only for 2B/9B."
        )
    if not os.path.exists(shared_syco_dataset):
        raise FileNotFoundError(
            "target='syco' requires the shared syco SFT dataset for this experiment, "
            f"but it was not found: {shared_syco_dataset}. "
            "Run run_scripts/run_step4_syco_dataset.sh first "
            "or pass --shared-syco-dataset-path to split a different shared source."
        )
    train_size = int(data_cfg.get("syco_split_train_size", DEFAULT_TRAIN_SIZE))
    eval_size = int(data_cfg.get("syco_split_eval_size", DEFAULT_EVAL_SIZE))
    seed = int(data_cfg.get("syco_split_seed", data_cfg.get("sample_seed", DEFAULT_SPLIT_SEED)))
    split_meta = ensure_shared_syco_sft_split(
        shared_syco_dataset,
        train_size=train_size,
        eval_size=eval_size,
        seed=seed,
        split_dir=data_cfg.get("syco_split_dir"),
        overwrite=bool(data_cfg.get("overwrite_syco_split", False)),
    )
    data_cfg["resolved_syco_split"] = split_meta
    return split_meta["train_path"], {
        "source": "shared_syco_sft_split",
        "dataset_metadata_path": metadata_path,
        "dataset_recovery_status": recovery_status,
        **split_meta,
    }


def build_training_examples(cfg: Dict[str, Any], group_name: str, model_name: str) -> List[TrainExample]:
    data_cfg = cfg["data"]
    group_cfg = cfg["experiments"][group_name]
    target = group_cfg["target"]

    paired_dataset = os.path.join(
        data_cfg.get("step2_dir", "outputs/step2"),
        model_name,
        "syco_dataset",
        "syco_dataset.jsonl",
    )
    if target == "syco":
        syco_dataset, syco_dataset_meta = resolve_syco_train_dataset(cfg, model_name)
        data_cfg["resolved_syco_train_dataset"] = syco_dataset_meta
    else:
        data_cfg["resolved_syco_train_dataset"] = {
            "source": "unused_for_target",
        }
    objective_dataset = data_cfg.get("objective_train_path") or paired_dataset
    max_syco = data_cfg.get("max_syco_examples")

    if target == "syco":
        examples = build_syco_examples(syco_dataset, max_examples=max_syco, cfg=cfg)
    elif target == "objective":
        examples = build_objective_examples(objective_dataset, max_examples=max_syco)
    elif target == "objective_refusal":
        examples = build_objective_examples(objective_dataset, max_examples=max_syco)
        ahc_path = data_cfg.get("ahc_train_path") or os.path.join(
            "outputs",
            "step1",
            "step1_bench",
            model_name,
            "ahc.parquet",
        )
        harm_examples = build_harm_refusal_examples(
            ahc_path,
            max_examples=data_cfg.get("max_harm_examples", 512),
            allow_missing=bool(data_cfg.get("allow_missing_ahc", False)),
        )
        if not harm_examples and not bool(data_cfg.get("allow_missing_ahc", False)):
            raise ValueError(
                f"No harmful refusal examples were loaded from {ahc_path}. "
                "Check the AHC parquet or set data.allow_missing_ahc=true only for an ablation."
            )
        examples.extend(harm_examples)
    else:
        raise ValueError(
            "Step 4 supports target='syco', 'objective', or 'objective_refusal'; "
            f"got {target!r}"
        )

    seed = int(cfg.get("training", {}).get("seed", 1234))
    random.Random(seed).shuffle(examples)
    max_total = data_cfg.get("max_train_examples")
    if max_total is not None:
        examples = examples[: int(max_total)]
    return examples


class SFTDataset(Dataset):
    """Pre-tokenized supervised fine-tuning dataset."""

    def __init__(self, examples: List[TrainExample], tokenizer, max_length: int, cfg: Dict[str, Any]):
        self.rows: List[Dict[str, Any]] = []
        train_cfg = cfg.get("training", {})
        self.syco_front_token_weight = float(train_cfg.get("syco_front_token_weight", 8.0))
        self.syco_front_token_count = int(train_cfg.get("syco_front_token_count", 128))
        self.use_weighted_syco_loss = bool(train_cfg.get("weighted_syco_front_loss", False))
        skipped = 0
        for ex in examples:
            try:
                input_ids, labels, assistant_mask = format_chat_pair(tokenizer, ex.prompt, ex.response)
                input_ids, labels, assistant_mask = truncate_example(
                    input_ids,
                    labels,
                    assistant_mask,
                    max_length=max_length,
                )
            except Exception as err:
                skipped += 1
                logger.warning("Skipping sample %s: %s", ex.sample_id, err)
                continue
            loss_weights = self.build_loss_weights(labels, assistant_mask, ex.target_type)
            self.rows.append(
                {
                    "sample_id": ex.sample_id,
                    "domain": ex.domain,
                    "source": ex.source,
                    "target_type": ex.target_type,
                    "input_ids": input_ids,
                    "labels": labels,
                    "assistant_mask": assistant_mask,
                    "loss_weights": loss_weights,
                }
            )
        if skipped:
            logger.warning("Skipped %s malformed examples", skipped)
        if not self.rows:
            raise ValueError("No valid Step 4 training examples")

    def build_loss_weights(
        self,
        labels: List[int],
        assistant_mask: List[int],
        target_type: str,
    ) -> List[float]:
        weights = [0.0 if label == IGNORE_INDEX else 1.0 for label in labels]
        if not self.use_weighted_syco_loss or target_type != "syco":
            return weights
        seen = 0
        for idx, (label, is_assistant) in enumerate(zip(labels, assistant_mask)):
            if label == IGNORE_INDEX or not is_assistant:
                continue
            if seen < self.syco_front_token_count:
                weights[idx] = self.syco_front_token_weight
            seen += 1
        return weights

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


class SFTCollator:
    """Pad input ids, labels, and assistant-token masks."""

    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(row["input_ids"]) for row in batch)
        input_ids = []
        labels = []
        loss_weights = []
        attention_mask = []
        assistant_mask = []
        sample_ids = []
        for row in batch:
            n = len(row["input_ids"])
            pad = max_len - n
            input_ids.append(row["input_ids"] + [self.pad_token_id] * pad)
            labels.append(row["labels"] + [IGNORE_INDEX] * pad)
            loss_weights.append(row["loss_weights"] + [0.0] * pad)
            attention_mask.append([1] * n + [0] * pad)
            assistant_mask.append(row["assistant_mask"] + [0] * pad)
            sample_ids.append(row["sample_id"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "loss_weights": torch.tensor(loss_weights, dtype=torch.float32),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "assistant_mask": torch.tensor(assistant_mask, dtype=torch.bool),
            "sample_ids": sample_ids,
            "target_types": [str(row["target_type"]) for row in batch],
        }


class TrainTimeSteeringHook:
    """Train-time activation injection hook.

    The hook receives a precomputed hidden-position mask.  For causal LM SFT this
    should normally mark positions whose logits predict supervised assistant
    tokens, not the assistant token positions themselves.
    """

    def __init__(
        self,
        layer_module: torch.nn.Module,
        steering_vec: torch.Tensor,
        beta: float,
        mode: str = "positive",
        token_scope: str = "assistant_prediction",
    ):
        self.beta = float(beta)
        self.mode = mode
        self.token_scope = token_scope
        self._steering_vec_cpu = steering_vec.detach().float().cpu()
        self.current_mask: Optional[torch.Tensor] = None
        try:
            self._handle = layer_module.register_forward_hook(self._fn, always_call=True)
        except TypeError:
            self._handle = layer_module.register_forward_hook(self._fn)

    def set_mask(self, assistant_mask: torch.Tensor) -> None:
        self.current_mask = assistant_mask

    def set_beta(self, beta: float) -> None:
        self.beta = float(beta)

    def _fn(self, module, inp, out):
        if self.mode == "none" or self.beta == 0.0 or self.current_mask is None:
            return out

        if isinstance(out, tuple):
            hidden_states = out[0]
        else:
            hidden_states = out
        if not torch.is_tensor(hidden_states) or hidden_states.ndim != 3:
            return out

        mask = self.current_mask.to(device=hidden_states.device)
        if mask.shape[:2] != hidden_states.shape[:2]:
            raise RuntimeError(
                f"Assistant mask shape {tuple(mask.shape)} does not match hidden states "
                f"{tuple(hidden_states.shape[:2])}"
            )
        sign = -1.0 if self.mode == "negative" else 1.0
        vec = self._steering_vec_cpu.to(device=hidden_states.device, dtype=hidden_states.dtype)
        delta = (
            sign
            * float(self.beta)
            * mask[:, :, None].to(hidden_states.dtype)
            * vec[None, None, :]
        )
        hidden_states = hidden_states + delta

        if isinstance(out, tuple):
            return (hidden_states,) + out[1:]
        return hidden_states

    def remove(self) -> None:
        self._handle.remove()


def build_train_time_steering_mask(
    assistant_mask: torch.Tensor,
    labels: torch.Tensor,
    token_scope: str,
) -> torch.Tensor:
    """Return hidden positions that should receive train-time steering.

    Causal LM loss uses ``logits[:, t]`` to predict ``labels[:, t + 1]``.  A
    train-time intervention intended to affect supervised assistant-token
    predictions must therefore be applied at the previous hidden position.  The
    legacy ``assistant_current`` mode is kept only for ablations.
    """

    scope = str(token_scope or "assistant_prediction").lower().strip()
    if scope in {"assistant_prediction", "predict_assistant", "causal_assistant", "assistant"}:
        mask = torch.zeros_like(assistant_mask, dtype=torch.bool)
        if assistant_mask.shape[1] > 1:
            next_is_supervised = labels[:, 1:] != IGNORE_INDEX
            mask[:, :-1] = assistant_mask[:, 1:].bool() & next_is_supervised
        return mask
    if scope in {"assistant_current", "response_current"}:
        return assistant_mask.bool() & (labels != IGNORE_INDEX)
    raise ValueError(
        "Unsupported token_scope for train-time steering: "
        f"{token_scope!r}. Use 'assistant_prediction' or 'assistant_current'."
    )


def limit_mask_to_front_positions(mask: torch.Tensor, count: Optional[int]) -> torch.Tensor:
    """Keep only the first ``count`` selected positions in each sequence."""

    if count is None:
        return mask
    count = int(count)
    if count <= 0:
        raise ValueError("injection_front_token_count must be positive")
    ordinal = mask.to(torch.int64).cumsum(dim=1)
    return mask & (ordinal <= count)


def normalize_injection_targets(raw_targets: Any) -> Tuple[str, ...]:
    if raw_targets is None:
        raw_targets = DEFAULT_INJECTION_TARGETS
    if isinstance(raw_targets, str):
        parts = [part.strip().lower() for part in raw_targets.replace(",", " ").split()]
    elif isinstance(raw_targets, Iterable):
        parts = []
        for item in raw_targets:
            parts.extend(str(item).replace(",", " ").split())
        parts = [part.strip().lower() for part in parts]
    else:
        parts = [str(raw_targets).strip().lower()]
    parts = [part for part in parts if part]
    if not parts:
        parts = list(DEFAULT_INJECTION_TARGETS)

    normalized: List[str] = []
    for part in parts:
        if part in {"all", "*"}:
            return ("all",)
        if part in {"syco", "sycophantic"}:
            target = TARGET_TYPE_SYCO
        elif part in {"alpaca", "instruction", "neutral_instruction", "alpaca_cleaned"}:
            target = TARGET_TYPE_INSTRUCTION
        else:
            raise ValueError(
                "Unsupported injection target "
                f"{part!r}. Use one of: syco, alpaca, instruction, all."
            )
        if target not in normalized:
            normalized.append(target)
    return tuple(normalized)


def build_injection_target_mask(
    assistant_mask: torch.Tensor,
    target_types: Sequence[str],
    injection_targets: Any,
) -> torch.Tensor:
    targets = normalize_injection_targets(injection_targets)
    batch_size = int(assistant_mask.shape[0])
    if len(target_types) != batch_size:
        raise ValueError(
            f"target_types length {len(target_types)} does not match batch size {batch_size}"
        )
    if "all" in targets:
        row_mask = torch.ones(batch_size, dtype=torch.bool, device=assistant_mask.device)
    else:
        target_set = set(targets)
        row_mask = torch.tensor(
            [str(target_type) in target_set for target_type in target_types],
            dtype=torch.bool,
            device=assistant_mask.device,
        )
    return row_mask[:, None].expand_as(assistant_mask)


def normalize_vector(vec: torch.Tensor) -> torch.Tensor:
    norm = vec.norm()
    if norm <= 0:
        raise ValueError("Cannot normalize zero steering vector")
    return vec / norm


def build_train_steering_vector(cfg: Dict[str, Any], group_cfg: Dict[str, Any], model_name: str) -> Optional[torch.Tensor]:
    if group_cfg.get("injection", "none") == "none":
        return None

    feature_cfg = dict(cfg["feature"])
    feature_cfg.update((cfg["feature"].get("by_model") or {}).get(model_name, {}))
    sae_cfg = cfg["sae"]["configs"][model_name]
    sae_path = os.path.join(sae_cfg["sae_dir"], f"layer{sae_cfg['layer']}.ae.pt")
    sae_state = load_sae_decoder(sae_path)
    decoder = sae_state["decoder_weight"]
    n_features = int(sae_state["n_features"])
    d_model = int(sae_state["d_model"])
    expected_n_features = sae_cfg.get("n_features")
    expected_hidden_size = sae_cfg.get("hidden_size")
    if expected_n_features is not None and int(expected_n_features) != n_features:
        raise ValueError(
            f"SAE n_features mismatch for {model_name}: config says {expected_n_features}, "
            f"but {sae_path} has {n_features}. Check configs/step4_vaccine.yaml."
        )
    if expected_hidden_size is not None and int(expected_hidden_size) != d_model:
        raise ValueError(
            f"SAE hidden_size mismatch for {model_name}: config says {expected_hidden_size}, "
            f"but {sae_path} has decoder d_model {d_model}. Check configs/step4_vaccine.yaml."
        )

    injection = group_cfg.get("injection")
    if injection not in {"positive", "negative", "random"}:
        raise ValueError("Step 4 only supports injection='none', 'positive', 'negative', or 'random'")
    if injection == "random":
        random_feature_id = group_cfg.get("random_feature_id", feature_cfg.get("random_feature_id", 0))
        feature_ids = [int(random_feature_id)]
    else:
        feature_ids = [int(fid) for fid in feature_cfg.get("feature_ids", [])]
        if not feature_ids:
            feature_ids = [int(feature_cfg["feature_id"])]
    bad_feature_ids = [fid for fid in feature_ids if fid < 0 or fid >= n_features]
    if bad_feature_ids:
        model_feature_cfg = (cfg.get("feature", {}).get("by_model") or {}).get(model_name)
        raise ValueError(
            f"Invalid Step 4 feature id(s) for {model_name}: {bad_feature_ids}; "
            f"SAE {sae_path} has valid ids [0, {n_features - 1}]. "
            f"Resolved feature cfg={feature_cfg}. "
            f"Model-specific override present={model_feature_cfg is not None}. "
            "Use feature ids from outputs/step2/<model>/syco_feature/top_features.json "
            "or pass --feature-id with a valid id."
        )

    vec = build_steering_vector(decoder, feature_ids, normalize=False)
    if feature_cfg.get("normalize", True):
        vec = normalize_vector(vec)
    return vec


def sample_train_beta(group_cfg: Dict[str, Any], default_beta: float, rng: random.Random) -> float:
    schedule = str(group_cfg.get("beta_schedule", "fixed")).lower()
    if schedule in {"fixed", "constant", "none"}:
        return float(default_beta)
    if schedule == "uniform":
        beta_min = float(group_cfg.get("beta_min", 0.0))
        beta_max = float(group_cfg.get("beta_max", group_cfg.get("beta", default_beta)))
        if beta_max < beta_min:
            raise ValueError(f"beta_max must be >= beta_min, got {beta_max} < {beta_min}")
        return rng.uniform(beta_min, beta_max)
    if schedule == "discrete":
        values = group_cfg.get("beta_values")
        if not values:
            raise ValueError("beta_schedule='discrete' requires group beta_values")
        return float(rng.choice([float(x) for x in values]))
    raise ValueError(f"Unsupported beta_schedule: {schedule!r}")


def save_dataset_manifest(path: str, examples: List[TrainExample], dataset: SFTDataset) -> None:
    rows = []
    by_id = {ex.sample_id: ex for ex in examples}
    for row in dataset.rows:
        ex = by_id.get(row["sample_id"])
        rows.append(
            {
                "sample_id": row["sample_id"],
                "domain": row["domain"],
                "source": row["source"],
                "target_type": row["target_type"],
                "tokens": len(row["input_ids"]),
                "supervised_tokens": sum(1 for label in row["labels"] if label != IGNORE_INDEX),
                "weighted_loss_tokens": round(float(sum(row["loss_weights"])), 3),
                "prompt": ex.prompt if ex else "",
                "response": ex.response if ex else "",
            }
        )
    write_jsonl(path, rows)


def save_loss_history(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["step", "epoch", "loss", "lr", "beta", "elapsed_sec"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_training_summary(path: str, summary: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def save_training_progress(path: str, row: Dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp_path, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 4 sycophancy SFT")
    parser.add_argument("--config", default="configs/step4_vaccine.yaml")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument(
        "--model-root",
        default=None,
        help="Root directory containing model checkpoints; used to resolve --model-path when omitted.",
    )
    parser.add_argument("--group", required=True, help="Experiment group from config")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root output directory. Used only when --output-dir is omitted.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Subdirectory under output.output_dir/<model>; ignored when --output-dir is set.",
    )
    parser.add_argument("--shared-syco-dataset-path", default=None)
    parser.add_argument("--syco-split-dir", default=None)
    parser.add_argument("--syco-split-train-size", type=int, default=None)
    parser.add_argument("--syco-split-eval-size", type=int, default=None)
    parser.add_argument("--syco-split-seed", type=int, default=None)
    parser.add_argument("--objective-train-path", default=None)
    parser.add_argument("--ahc-train-path", default=None)
    parser.add_argument(
        "--allow-missing-ahc",
        action="store_true",
        help="Allow objective_refusal/vaccine groups to run without AHC refusal examples.",
    )
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-syco-examples", type=int, default=None)
    parser.add_argument("--max-harm-examples", type=int, default=None)
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum tokenized SFT sequence length after chat formatting.",
    )
    parser.add_argument("--syco-front-token-count", type=int, default=None)
    parser.add_argument("--syco-front-token-weight", type=float, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    checkpointing_group = parser.add_mutually_exclusive_group()
    checkpointing_group.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        default=None,
        help="Enable activation gradient checkpointing.",
    )
    checkpointing_group.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable activation gradient checkpointing when memory permits.",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="Desired effective global batch size across all ranks; adjusts gradient accumulation.",
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=None,
        help="Warmup fraction of total optimizer steps, e.g. 0.03 for 3%%.",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--tuning-mode",
        choices=sorted(SUPPORTED_TUNING_MODES),
        default=None,
        help="Fine-tuning mode: full parameter update or LoRA adapters.",
    )
    parser.add_argument(
        "--full-finetune",
        action="store_true",
        help="Alias for --tuning-mode full.",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Alias for --tuning-mode lora.",
    )
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=float, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--beta-max", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--beta-min", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--beta-schedule",
        choices=["fixed", "uniform", "discrete"],
        default=None,
        help="Train-time beta schedule for injection groups.",
    )
    parser.add_argument(
        "--injection-targets",
        "--injection_targets",
        default=None,
        help=(
            "Which target_type rows receive train-time feature injection: "
            "syco, alpaca/instruction, or all. Defaults to syco."
        ),
    )
    parser.add_argument(
        "--injection-front-token-count",
        "--injection_front_token_count",
        type=int,
        default=None,
        help=(
            "Optional cap on injected assistant prediction positions per example. "
            "By default every eligible position is injected."
        ),
    )
    parser.add_argument("--deepspeed-config", default=None)
    parser.add_argument(
        "--full-model-export",
        choices=["checkpoint", "auto", "hf", "disabled"],
        default=None,
        help=(
            "Full-tuning save format. checkpoint is safest for 35B; auto/hf "
            "also attempt a vLLM-loadable HF weight export."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-merged", action="store_true")
    return parser.parse_args()


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    data_cfg = cfg.setdefault("data", {})
    train_cfg = cfg.setdefault("training", {})
    output_cfg = cfg.setdefault("output", {})
    model_cfg = cfg.setdefault("model", {})
    lora_cfg = cfg.setdefault("lora", {})
    group_cfg = cfg["experiments"][args.group]

    if args.shared_syco_dataset_path is not None:
        data_cfg["shared_syco_dataset_path"] = args.shared_syco_dataset_path
    if args.syco_split_dir is not None:
        data_cfg["syco_split_dir"] = args.syco_split_dir
    if args.syco_split_train_size is not None:
        data_cfg["syco_split_train_size"] = args.syco_split_train_size
    if args.syco_split_eval_size is not None:
        data_cfg["syco_split_eval_size"] = args.syco_split_eval_size
    if args.syco_split_seed is not None:
        data_cfg["syco_split_seed"] = args.syco_split_seed
    if args.objective_train_path is not None:
        data_cfg["objective_train_path"] = args.objective_train_path
    if args.ahc_train_path is not None:
        data_cfg["ahc_train_path"] = args.ahc_train_path
    if args.allow_missing_ahc:
        data_cfg["allow_missing_ahc"] = True
    if args.max_train_examples is not None:
        data_cfg["max_train_examples"] = args.max_train_examples
    if args.max_syco_examples is not None:
        data_cfg["max_syco_examples"] = args.max_syco_examples
    if args.max_harm_examples is not None:
        data_cfg["max_harm_examples"] = args.max_harm_examples
    if args.max_length is not None:
        if args.max_length <= 0:
            raise ValueError("--max-length must be positive")
        train_cfg["max_length"] = args.max_length
    if args.syco_front_token_count is not None:
        if args.syco_front_token_count <= 0:
            raise ValueError("--syco-front-token-count must be positive")
        train_cfg["syco_front_token_count"] = args.syco_front_token_count
    if args.syco_front_token_weight is not None:
        if args.syco_front_token_weight <= 0:
            raise ValueError("--syco-front-token-weight must be positive")
        train_cfg["syco_front_token_weight"] = args.syco_front_token_weight
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["per_device_batch_size"] = args.batch_size
    if args.gradient_accumulation_steps is not None:
        train_cfg["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.gradient_checkpointing is not None:
        train_cfg["gradient_checkpointing"] = bool(args.gradient_checkpointing)
    if args.global_batch_size is not None:
        if args.global_batch_size <= 0:
            raise ValueError("--global-batch-size must be positive")
        train_cfg["global_batch_size"] = args.global_batch_size
    if args.learning_rate is not None:
        train_cfg["learning_rate"] = args.learning_rate
    if args.warmup_steps is not None:
        train_cfg["warmup_steps"] = args.warmup_steps
        train_cfg.pop("warmup_ratio", None)
    if args.warmup_ratio is not None:
        if not 0.0 <= args.warmup_ratio <= 1.0:
            raise ValueError("--warmup-ratio must be in [0, 1]")
        train_cfg["warmup_ratio"] = args.warmup_ratio
    if args.max_steps is not None:
        train_cfg["max_steps"] = args.max_steps
    requested_tuning_modes = []
    if args.tuning_mode is not None:
        requested_tuning_modes.append(args.tuning_mode)
    if args.full_finetune:
        requested_tuning_modes.append(TUNING_MODE_FULL)
    if args.lora:
        requested_tuning_modes.append(TUNING_MODE_LORA)
    if len(set(requested_tuning_modes)) > 1:
        raise ValueError("Conflicting tuning mode flags; choose either full or lora")
    if requested_tuning_modes:
        train_cfg["tuning_mode"] = requested_tuning_modes[-1]
    resolve_tuning_mode(cfg)
    if args.lora_rank is not None:
        if args.lora_rank <= 0:
            raise ValueError("--lora-rank must be positive")
        lora_cfg["rank"] = args.lora_rank
    if args.lora_alpha is not None:
        if args.lora_alpha <= 0:
            raise ValueError("--lora-alpha must be positive")
        lora_cfg["alpha"] = args.lora_alpha
    if args.lora_dropout is not None:
        if not 0.0 <= args.lora_dropout < 1.0:
            raise ValueError("--lora-dropout must be in [0, 1)")
        lora_cfg["dropout"] = args.lora_dropout
    if args.deepspeed_config is not None:
        train_cfg["deepspeed_config"] = args.deepspeed_config
    if args.full_model_export is not None:
        output_cfg["full_model_export"] = args.full_model_export
    if args.output_root is not None:
        output_cfg["output_dir"] = args.output_root
    if args.model_root is not None:
        model_cfg["model_root"] = args.model_root
    beta_schedule_override = args.beta_schedule
    if args.beta is not None:
        group_cfg["beta"] = args.beta
        schedule = str(args.beta_schedule or group_cfg.get("beta_schedule", "fixed")).lower()
        if args.beta_max is None and schedule == "uniform":
            # In uniform mode, a CLI beta sweep should sweep the upper bound.
            group_cfg["beta_max"] = args.beta
            if float(args.beta) < float(group_cfg.get("beta_min", 0.0)):
                # Negative beta is used for reverse-direction feature injection.
                # Treat it as an exact requested coefficient instead of sampling
                # uniformly from an invalid [0, negative] interval.
                group_cfg["beta_schedule"] = "fixed"
                group_cfg["beta_min"] = args.beta
                group_cfg["beta_max"] = args.beta
                beta_schedule_override = "fixed"
        elif float(args.beta) < 0 and schedule in {"fixed", "constant", "none"}:
            group_cfg["beta_min"] = args.beta
            group_cfg["beta_max"] = args.beta
    if args.beta_max is not None:
        group_cfg["beta"] = args.beta_max
        group_cfg["beta_max"] = args.beta_max
    if args.beta_min is not None:
        group_cfg["beta_min"] = args.beta_min
    if beta_schedule_override is not None:
        group_cfg["beta_schedule"] = beta_schedule_override
    if args.injection_targets is not None:
        group_cfg["injection_targets"] = list(normalize_injection_targets(args.injection_targets))
    else:
        group_cfg["injection_targets"] = list(
            normalize_injection_targets(group_cfg.get("injection_targets", DEFAULT_INJECTION_TARGETS))
        )
    injection_front_token_count = getattr(args, "injection_front_token_count", None)
    if injection_front_token_count is not None:
        if injection_front_token_count <= 0:
            raise ValueError("--injection-front-token-count must be positive")
        group_cfg["injection_front_token_count"] = int(injection_front_token_count)
    if args.dry_run:
        current_max_train = data_cfg.get("max_train_examples")
        data_cfg["max_train_examples"] = (
            min(int(current_max_train), 16)
            if current_max_train is not None else 16
        )
        train_cfg["epochs"] = 0.02
        current_max_steps = train_cfg.get("max_steps")
        train_cfg["max_steps"] = (
            min(int(current_max_steps), 3)
            if current_max_steps is not None else 3
        )


def resolve_model(cfg: Dict[str, Any], args: argparse.Namespace) -> Tuple[str, str]:
    model_root = cfg.get("model", {}).get(
        "model_root",
        "./models",
    )
    model_name = args.model_name
    model_path = args.model_path
    if model_path and not model_name:
        model_name = os.path.basename(model_path.rstrip("/"))
    if model_name and not model_path:
        model_path = os.path.join(model_root, model_name)
    if not model_name:
        model_name = cfg.get("model", {}).get("default_model_name", "Qwen3.5-35B-A3B-Base")
    if not model_path:
        model_path = os.path.join(model_root, model_name)
    return model_name, model_path


def normalize_deepspeed_path(path: Any) -> str:
    return str(path or "").strip()


def is_full_model_eval_ready(output_dir: str) -> bool:
    """Return whether a directory contains HF weights that vLLM can load."""

    if not os.path.isdir(output_dir):
        return False
    if not os.path.exists(os.path.join(output_dir, "config.json")):
        return False
    weight_names = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    return any(os.path.exists(os.path.join(output_dir, name)) for name in weight_names)


def _read_deepspeed_latest_tag(ckpt_dir: str) -> str:
    latest = os.path.join(ckpt_dir, "latest")
    try:
        with open(latest, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def is_full_model_checkpoint_ready(output_dir: str, expected_world_size: Optional[int] = None) -> bool:
    ckpt_dir = os.path.join(output_dir, "deepspeed_checkpoint")
    if not os.path.isdir(ckpt_dir):
        return False
    tag = _read_deepspeed_latest_tag(ckpt_dir)
    if not tag:
        return False
    tag_dir = os.path.join(ckpt_dir, tag)
    if not os.path.isdir(tag_dir):
        return False
    model_count = len(
        [
            name
            for name in os.listdir(tag_dir)
            if name.endswith("_model_states.pt")
        ]
    )
    optim_count = len(
        [
            name
            for name in os.listdir(tag_dir)
            if name.endswith("_optim_states.pt")
        ]
    )
    if expected_world_size is not None and expected_world_size > 1:
        # ZeRO-2 writes one replicated model state and one optimizer state per
        # rank. ZeRO-3 writes partitioned model and optimizer states per rank.
        zero2_layout = model_count == 1 and optim_count == expected_world_size
        zero3_layout = model_count == expected_world_size and optim_count == expected_world_size
        return zero2_layout or zero3_layout
    return model_count > 0 and model_count == optim_count


def format_beta_for_path(beta: float) -> str:
    text = f"{float(beta):g}"
    return text.replace("-", "neg").replace(".", "p")


def format_number_for_path(value: Any) -> str:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        text = str(value)
    else:
        text = f"{value_f:g}"
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


def resolved_group_beta(group_cfg: Dict[str, Any]) -> float:
    if group_cfg.get("injection", "none") == "none":
        return 0.0
    if "beta" in group_cfg:
        return float(group_cfg["beta"])
    if "beta_max" in group_cfg:
        return float(group_cfg["beta_max"])
    return 0.0


def resolved_feature_ids_for_run(cfg: Dict[str, Any], group_cfg: Dict[str, Any], model_name: str) -> List[int]:
    if group_cfg.get("injection", "none") == "none":
        return []
    feature_cfg = dict(cfg.get("feature") or {})
    feature_cfg.update((cfg.get("feature", {}).get("by_model") or {}).get(model_name, {}))
    if group_cfg.get("injection") == "random":
        return [int(group_cfg.get("random_feature_id", feature_cfg.get("random_feature_id", 0)))]
    feature_ids = [int(fid) for fid in feature_cfg.get("feature_ids", [])]
    if not feature_ids and "feature_id" in feature_cfg:
        feature_ids = [int(feature_cfg["feature_id"])]
    return feature_ids


def feature_caveat_for_run(model_name: str, feature_ids: Sequence[int], group_cfg: Dict[str, Any]) -> Optional[str]:
    if group_cfg.get("injection", "none") == "none":
        return None
    if model_name == "Qwen3.5-2B-Base" and 7933 in [int(fid) for fid in feature_ids]:
        return (
            "Qwen3.5-2B-Base feature f7933 is the historical pair-frequency candidate, "
            "not the current mean-delta Top1. The validated current Step 2/3/4/5 feature "
            "is f28758; use f7933 only for historical comparison."
        )
    return None


def build_auto_run_name(cfg: Dict[str, Any], args: argparse.Namespace, model_name: str) -> str:
    group_cfg = cfg["experiments"][args.group]
    train_cfg = cfg["training"]
    tuning_mode = resolve_tuning_mode(cfg)
    parts = [args.group]

    feature_ids = resolved_feature_ids_for_run(cfg, group_cfg, model_name)
    if feature_ids:
        feature_tag = "f" + "-".join(str(fid) for fid in feature_ids)
        beta = resolved_group_beta(group_cfg)
        parts.extend([feature_tag, f"alpha{format_beta_for_path(beta)}"])
        injection_targets = normalize_injection_targets(
            group_cfg.get("injection_targets", DEFAULT_INJECTION_TARGETS)
        )
        if injection_targets != DEFAULT_INJECTION_TARGETS:
            target_tag = "-".join(
                "alpaca" if target == TARGET_TYPE_INSTRUCTION else str(target)
                for target in injection_targets
            )
            parts.append(f"inject{target_tag}")

    parts.append(tuning_mode)

    lr = train_cfg.get("learning_rate")
    if lr is not None:
        parts.append(f"lr{format_number_for_path(lr)}")

    max_steps = train_cfg.get("max_steps")
    if max_steps is not None:
        parts.append(f"steps{format_number_for_path(max_steps)}")
    else:
        epochs = train_cfg.get("epochs")
        if epochs is not None:
            parts.append(f"ep{format_number_for_path(epochs)}")

    global_bs = train_cfg.get("global_batch_size")
    if global_bs is None:
        global_bs = (
            int(train_cfg.get("per_device_batch_size", 1))
            * int(train_cfg.get("gradient_accumulation_steps", 1))
        )
    parts.append(f"gbs{format_number_for_path(global_bs)}")

    if tuning_mode == TUNING_MODE_LORA:
        lora_cfg = cfg.get("lora") or {}
        parts.append(f"r{format_number_for_path(lora_cfg.get('rank', 0))}")

    return "_".join(str(part) for part in parts if str(part))


def build_output_dir(cfg: Dict[str, Any], args: argparse.Namespace, model_name: str) -> str:
    if args.output_dir:
        return args.output_dir
    base = cfg.get("output", {}).get(
        "output_dir",
        "outputs/step4_feature_inject/train",
    )
    if args.run_name:
        return os.path.join(base, model_name, args.run_name)
    return os.path.join(base, model_name, build_auto_run_name(cfg, args, model_name))


def maybe_prepare_output_dir(output_dir: str, overwrite: bool) -> None:
    if os.path.exists(output_dir) and overwrite:
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


def setup_distributed_from_env() -> RuntimeContext:
    """Initialize a torch.distributed process group from DeepSpeed launcher env."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_distributed = world_size > 1
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    if use_distributed and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    return RuntimeContext(
        distributed=use_distributed,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def load_deepspeed_config(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    path = cfg.get("training", {}).get("deepspeed_config")
    if not path:
        return None
    if str(path).lower() in {"none", "null", "false", "0", "disabled"}:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def uses_zero3(ds_config: Optional[Dict[str, Any]]) -> bool:
    if not ds_config:
        return False
    zero_cfg = ds_config.get("zero_optimization") or {}
    if isinstance(zero_cfg, bool):
        return False
    return int(zero_cfg.get("stage", 0)) == 3


def zero3_param_offload_config(ds_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not uses_zero3(ds_config):
        return {}
    zero_cfg = ds_config.get("zero_optimization") or {}
    offload_cfg = zero_cfg.get("offload_param") or {}
    if not isinstance(offload_cfg, dict):
        return {}
    return offload_cfg


def qwen35_linear_attention_fast_path_available() -> bool:
    """Return whether Qwen3.5 linear-attention CUDA kernels are importable."""

    try:
        from transformers.utils.import_utils import (
            is_causal_conv1d_available,
            is_flash_linear_attention_available,
        )
    except Exception:
        return False
    return bool(is_causal_conv1d_available() and is_flash_linear_attention_available())


def apply_qwen35_large_model_memory_safety_overrides(cfg: Dict[str, Any], model_name: str) -> bool:
    """Keep large Qwen3.5 Step 4 runs trainable on available 80GB GPUs.

    Two separate memory cliffs show up in this environment:

    * Qwen3.5-27B falls back to a memory-hungry PyTorch linear-attention path
      when ``flash-linear-attention`` / ``causal-conv1d`` are absent.
    * Full-parameter SFT of Qwen3.5-35B-A3B-Base needs ZeRO-3 parameter
      partitioning. Eight-GPU runs use optimizer offload, while the dedicated
      16-GPU config keeps parameters and optimizer state resident on GPU.

    The offload-based safety configs use micro batch 1.  The dedicated 16-GPU
    no-offload config may use its empirically validated larger micro batch. If
    a global batch was requested, gradient accumulation is recomputed later.
    """

    protected_models = {"Qwen3.5-27B", "Qwen3.5-35B-A3B-Base"}
    if model_name not in protected_models:
        return False
    tuning_mode = resolve_tuning_mode(cfg)
    fast_path_available = qwen35_linear_attention_fast_path_available()
    if tuning_mode == TUNING_MODE_LORA and model_name == "Qwen3.5-27B" and fast_path_available:
        return False
    if os.environ.get("STEP4_ALLOW_UNSAFE_QWEN35_LARGE_MODEL", "").lower() in {"1", "true", "yes"}:
        return False

    train_cfg = cfg.setdefault("training", {})
    old_micro = int(train_cfg.get("per_device_batch_size", 1))
    old_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    old_ds_config = train_cfg.get("deepspeed_config")
    new_ds_config = old_ds_config
    switched_deepspeed = False
    if tuning_mode == TUNING_MODE_FULL:
        if not old_ds_config:
            new_ds_config = FULL_TUNING_SAFE_DEEPSPEED_CONFIG
            train_cfg["deepspeed_config"] = new_ds_config
            switched_deepspeed = True
        elif normalize_deepspeed_path(old_ds_config) not in FULL_TUNING_SAFE_DEEPSPEED_CONFIGS:
            new_ds_config = FULL_TUNING_SAFE_DEEPSPEED_CONFIG
            train_cfg["deepspeed_config"] = new_ds_config
            switched_deepspeed = True
        if normalize_deepspeed_path(new_ds_config) in FULL_TUNING_NO_OFFLOAD_DEEPSPEED_CONFIGS:
            train_cfg["full_tuning_requires_zero3_optimizer_offload"] = False
        else:
            train_cfg.setdefault("full_tuning_requires_zero3_optimizer_offload", True)
    else:
        if not old_ds_config:
            new_ds_config = old_ds_config
        elif model_name == "Qwen3.5-35B-A3B-Base":
            if str(old_ds_config) != LORA_LARGE_MODEL_SAFE_DEEPSPEED_CONFIG:
                new_ds_config = LORA_LARGE_MODEL_SAFE_DEEPSPEED_CONFIG
                train_cfg["deepspeed_config"] = new_ds_config
                switched_deepspeed = True
        elif (
            old_ds_config
            and "zero3" in str(old_ds_config)
            and os.environ.get("STEP4_KEEP_ZERO3_QWEN35_LARGE_MODEL", "").lower() not in {"1", "true", "yes"}
        ):
            new_ds_config = LORA_LARGE_MODEL_SAFE_DEEPSPEED_CONFIG
            train_cfg["deepspeed_config"] = new_ds_config
            switched_deepspeed = True

    no_offload = (
        tuning_mode == TUNING_MODE_FULL
        and normalize_deepspeed_path(new_ds_config) in FULL_TUNING_NO_OFFLOAD_DEEPSPEED_CONFIGS
    )
    # The dedicated 16-GPU config is tuned empirically and may use a larger
    # micro batch when memory permits. Eight-GPU safety configs still clamp it.
    changed_batch = old_micro > 1 and not no_offload
    if changed_batch:
        new_micro = 1
        if train_cfg.get("global_batch_size") is None:
            new_accum = max(1, old_micro * old_accum)
        else:
            new_accum = old_accum
        train_cfg["per_device_batch_size"] = new_micro
        train_cfg["gradient_accumulation_steps"] = new_accum
    else:
        new_micro = old_micro
        new_accum = old_accum
    train_cfg["linear_attention_fast_path"] = "available" if fast_path_available else "missing"
    old_max_length = int(train_cfg.get("max_length", 512))
    if tuning_mode == TUNING_MODE_FULL and model_name == "Qwen3.5-35B-A3B-Base":
        max_safe_length = int(os.environ.get("STEP4_QWEN35_A3B_FULL_MAX_LENGTH", "512"))
        if old_max_length > max_safe_length:
            train_cfg["max_length"] = max_safe_length
    if no_offload:
        reason = (
            f"{model_name} full-parameter SFT uses 16-rank ZeRO-3 with GPU-resident "
            f"parameters, gradients, and optimizer states; micro={new_micro} and bf16 leave "
            "activation and communication headroom on 80GB GPUs"
        )
    elif tuning_mode == TUNING_MODE_FULL:
        reason = (
            f"{model_name} full-parameter SFT requires ZeRO-3 parameter partitioning, "
            "CPU optimizer offload, micro=1, bf16, and bounded sequences to avoid "
            "80GB GPU OOM without exceeding the container host-memory limit"
        )
    elif model_name == "Qwen3.5-35B-A3B-Base":
        reason = (
            "Qwen3.5-35B-A3B-Base ZeRO-3 can OOM during MoE expert parameter "
            "fetch on the first forward; use ZeRO-2 + micro=1 on the intended "
            "free GPUs for LoRA-only SFT while preserving the requested global batch"
        )
    else:
        reason = (
            "Qwen3.5-27B linear-attention fast path is missing; use a lower "
            "micro batch and ZeRO-2 to avoid the PyTorch fallback and ZeRO-3 "
            "parameter-fetch OOM paths while preserving the per-rank effective batch"
        )
    train_cfg["memory_safety_override"] = {
        "reason": reason,
        "tuning_mode": tuning_mode,
        "model_name": model_name,
        "old_deepspeed_config": old_ds_config,
        "new_deepspeed_config": new_ds_config,
        "switched_deepspeed_config": switched_deepspeed,
        "old_per_device_batch_size": old_micro,
        "old_gradient_accumulation_steps": old_accum,
        "new_per_device_batch_size": new_micro,
        "new_gradient_accumulation_steps": new_accum,
        "old_effective_batch_per_rank": old_micro * old_accum,
        "new_effective_batch_per_rank": new_micro * new_accum,
        "old_max_length": old_max_length,
        "new_max_length": int(train_cfg.get("max_length", old_max_length)),
    }
    return changed_batch or switched_deepspeed or int(train_cfg.get("max_length", old_max_length)) != old_max_length


def apply_global_batch_size_override(cfg: Dict[str, Any], world_size: int) -> bool:
    train_cfg = cfg.setdefault("training", {})
    target_global = train_cfg.get("global_batch_size")
    if target_global is None:
        return False
    target_global = int(target_global)
    if target_global <= 0:
        raise ValueError("training.global_batch_size must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")

    micro = int(train_cfg.get("per_device_batch_size", 1))
    if micro <= 0:
        raise ValueError("training.per_device_batch_size must be positive")
    denom = micro * world_size
    if target_global % denom != 0:
        raise ValueError(
            "global batch size must be divisible by "
            f"per_device_batch_size * world_size ({micro} * {world_size} = {denom}); "
            f"got {target_global}"
        )
    old_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    new_accum = max(1, target_global // denom)
    train_cfg["gradient_accumulation_steps"] = new_accum
    train_cfg["global_batch_size"] = target_global
    train_cfg["global_batch_size_override"] = {
        "target_global_batch_size": target_global,
        "world_size": world_size,
        "per_device_batch_size": micro,
        "old_gradient_accumulation_steps": old_accum,
        "new_gradient_accumulation_steps": new_accum,
    }
    return old_accum != new_accum


def sync_deepspeed_config(ds_config: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Keep DeepSpeed JSON values aligned with the Step 4 training YAML."""

    out = dict(ds_config)
    train_cfg = cfg["training"]
    out["train_micro_batch_size_per_gpu"] = int(train_cfg.get("per_device_batch_size", 1))
    out["gradient_accumulation_steps"] = int(train_cfg.get("gradient_accumulation_steps", 1))
    mp = train_cfg.get("mixed_precision", "bf16")
    out.setdefault("bf16", {})["enabled"] = bool(mp == "bf16")
    out.setdefault("fp16", {})["enabled"] = bool(mp == "fp16")
    if train_cfg.get("max_grad_norm") is not None:
        out["gradient_clipping"] = float(train_cfg["max_grad_norm"])
    optimizer_cfg = out.setdefault("optimizer", {})
    optimizer_cfg.setdefault("type", "AdamW")
    optimizer_params = optimizer_cfg.setdefault("params", {})
    optimizer_params["lr"] = float(train_cfg.get("learning_rate", 1e-5))
    optimizer_params["betas"] = [float(x) for x in train_cfg.get("betas", [0.9, 0.95])]
    optimizer_params["eps"] = float(train_cfg.get("eps", 1e-8))
    optimizer_params["weight_decay"] = float(train_cfg.get("weight_decay", 0.0))
    return out


def validate_full_tuning_deepspeed_config(
    cfg: Dict[str, Any],
    ds_config: Optional[Dict[str, Any]],
    model_name: str,
) -> None:
    if resolve_tuning_mode(cfg) != TUNING_MODE_FULL:
        return
    large_models_require_zero3 = model_name in {"Qwen3.5-27B", "Qwen3.5-35B-A3B-Base"} or bool(
        cfg.get("training", {}).get("full_tuning_requires_zero3_offload", False)
        or cfg.get("training", {}).get("full_tuning_requires_zero3_optimizer_offload", False)
    )
    if not large_models_require_zero3:
        return
    if ds_config is None:
        raise ValueError(
            "Full-parameter tuning requires DeepSpeed ZeRO-3 for the large Qwen3.5 "
            "defaults. Pass --tuning-mode lora for adapter-only SFT or provide a "
            "supported ZeRO-3 config."
        )
    if not uses_zero3(ds_config):
        raise ValueError(
            "Full-parameter tuning requires ZeRO-3. The current DeepSpeed config "
            f"is {cfg.get('training', {}).get('deepspeed_config')!r}."
        )
    requires_optimizer_offload = bool(
        cfg.get("training", {}).get(
            "full_tuning_requires_zero3_optimizer_offload",
            model_name in {"Qwen3.5-27B", "Qwen3.5-35B-A3B-Base"},
        )
    )
    if requires_optimizer_offload:
        zero_cfg = ds_config.get("zero_optimization") or {}
        optim_offload = zero_cfg.get("offload_optimizer") or {}
        if str(optim_offload.get("device", "")).lower() != "cpu":
            raise ValueError("Full-parameter tuning requires zero_optimization.offload_optimizer.device=cpu")


def disable_incompatible_zero3_checkpointing(cfg: Dict[str, Any], ds_config: Optional[Dict[str, Any]]) -> bool:
    """Disable HF gradient checkpointing for ZeRO-3.

    PyTorch checkpoint recomputation can observe ZeRO-3 partitioned parameters
    as empty local shards even when the first forward pass saved full tensors.
    That produces metadata mismatches such as saved ``[12288, 5120]`` vs
    recomputed ``[0]`` during backward.
    """

    train_cfg = cfg.setdefault("training", {})
    if not uses_zero3(ds_config):
        return False
    if not bool(train_cfg.get("gradient_checkpointing", True)):
        return False
    if bool(train_cfg.get("allow_zero3_gradient_checkpointing", False)):
        return False
    train_cfg["gradient_checkpointing"] = False
    train_cfg["gradient_checkpointing_disabled_reason"] = (
        "disabled automatically because HF gradient checkpointing is incompatible "
        "with the current ZeRO-3 parameter-sharding recomputation path"
    )
    return True


def load_model(
    model_path: str,
    cfg: Dict[str, Any],
    runtime: RuntimeContext,
    ds_config: Optional[Dict[str, Any]] = None,
):
    model_cfg = cfg.get("model", {})
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds_zero3 = uses_zero3(ds_config)
    if ds_config is not None and HfDeepSpeedConfig is None:
        raise RuntimeError("DeepSpeed was requested but transformers DeepSpeed integration is unavailable")
    if ds_zero3:
        # Keep this object alive while from_pretrained constructs sharded parameters.
        load_model._hf_ds_config = HfDeepSpeedConfig(ds_config)

    device_map = None if ds_config is not None else model_cfg.get("device_map", "auto")
    max_memory = model_cfg.get("max_memory")
    load_kwargs = {
        "torch_dtype": torch_dtype_from_name(model_cfg.get("torch_dtype", "bfloat16")),
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if device_map:
        load_kwargs["device_map"] = device_map
    if max_memory and ds_config is None:
        load_kwargs["max_memory"] = max_memory

    offload_cfg = zero3_param_offload_config(ds_config)
    zero_init_kwargs = {
        "config_dict_or_path": ds_config,
        "enabled": True,
    }
    if str(offload_cfg.get("device", "")).lower() == "cpu":
        zero_init_kwargs["remote_device"] = "cpu"
        zero_init_kwargs["pin_memory"] = bool(offload_cfg.get("pin_memory", False))
    context = (
        zero.Init(**zero_init_kwargs)
        if ds_zero3 and zero is not None else nullcontext()
    )
    with context:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    if ds_config is None or not ds_zero3:
        model.to(runtime.device)
    text_cfg = get_text_config(model.config)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(text_cfg, "use_cache"):
        text_cfg.use_cache = False
    if cfg.get("training", {}).get("gradient_checkpointing", True):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model, tokenizer


def primary_input_device(model) -> torch.device:
    """Return the device of the input embedding table."""

    return model.get_input_embeddings().weight.device


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move tensor batch fields to the model input device."""

    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def create_optimizer(model, cfg: Dict[str, Any]):
    train_cfg = cfg["training"]
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        params,
        lr=float(train_cfg.get("learning_rate", 1e-5)),
        betas=tuple(train_cfg.get("betas", [0.9, 0.95])),
        eps=float(train_cfg.get("eps", 1e-8)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )


def get_trainable_parameters(model) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def initialize_deepspeed_engine(
    model,
    optimizer,
    scheduler,
    cfg: Dict[str, Any],
    runtime: RuntimeContext,
    ds_config: Optional[Dict[str, Any]],
):
    if ds_config is None:
        return model, optimizer, scheduler
    if deepspeed is None:
        raise RuntimeError(
            "DeepSpeed config was provided, but the deepspeed package is not importable "
            "in this Python environment."
            + (f" Original import error: {_DEEPSPEED_IMPORT_ERROR}" if _DEEPSPEED_IMPORT_ERROR else "")
        )
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        model_parameters=get_trainable_parameters(model),
        lr_scheduler=scheduler,
        config=ds_config,
        dist_init_required=False if runtime.distributed else None,
    )
    return engine, optimizer, scheduler


def optimizer_learning_rate(optimizer, scheduler=None) -> float:
    if scheduler is not None and hasattr(scheduler, "get_last_lr"):
        values = scheduler.get_last_lr()
        if values:
            return float(values[0])
    if hasattr(optimizer, "param_groups") and optimizer.param_groups:
        return float(optimizer.param_groups[0].get("lr", 0.0))
    return 0.0


def reduce_mean_scalar(value: float, runtime: RuntimeContext) -> float:
    tensor = torch.tensor(float(value), device=runtime.device)
    if runtime.distributed and torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        tensor /= float(runtime.world_size)
    return float(tensor.item())


def reduce_sum_scalar(value: float, runtime: RuntimeContext) -> float:
    tensor = torch.tensor(float(value), device=runtime.device)
    if runtime.distributed and torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return float(tensor.item())


def weighted_causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor, loss_weights: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_weights = loss_weights[..., 1:].to(device=shift_logits.device, dtype=torch.float32).contiguous()
    token_loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view_as(shift_labels)
    valid = (shift_labels != IGNORE_INDEX).to(dtype=torch.float32)
    weights = shift_weights * valid
    denom = weights.sum().clamp_min(1.0)
    return (token_loss * weights).sum() / denom


def gather_lora_state_dict(model, runtime: RuntimeContext) -> Dict[str, torch.Tensor]:
    """Collect LoRA tensors from DeepSpeed-sharded modules on rank 0."""

    from step4_lora import iter_lora_modules

    state: Dict[str, torch.Tensor] = {}
    gather_enabled = zero is not None and any(
        hasattr(param, "ds_id")
        for _, module in iter_lora_modules(model)
        for param in (module.lora_A, module.lora_B)
    )
    for name, module in iter_lora_modules(model):
        for suffix, param in (("lora_A", module.lora_A), ("lora_B", module.lora_B)):
            if gather_enabled:
                with zero.GatheredParameters([param], modifier_rank=0):
                    if runtime.is_main:
                        state[f"{name}.{suffix}"] = param.detach().cpu().clone()
            elif runtime.is_main:
                state[f"{name}.{suffix}"] = param.detach().cpu().clone()
    return state


def save_lora_adapter_distributed(
    model,
    output_dir: str,
    config: LoRAConfig,
    metadata: Dict[str, Any],
    runtime: RuntimeContext,
) -> None:
    state = gather_lora_state_dict(model, runtime)
    if not runtime.is_main:
        return
    os.makedirs(output_dir, exist_ok=True)
    torch.save(state, os.path.join(output_dir, "adapter_model.pt"))
    payload = config.to_dict()
    payload["metadata"] = metadata
    with open(os.path.join(output_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _parse_size_to_bytes(value: str) -> int:
    text = str(value).strip().upper()
    units = {
        "GIB": 1024 ** 3,
        "MIB": 1024 ** 2,
        "KIB": 1024,
        "GB": 1000 ** 3,
        "MB": 1000 ** 2,
        "KB": 1000,
        "B": 1,
    }
    for suffix, factor in units.items():
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)].strip()) * factor)
    return int(float(text))


def _get_nested_module(root: torch.nn.Module, path: Sequence[str]) -> torch.nn.Module:
    module = root
    for name in path:
        module = getattr(module, name)
    return module


def _state_dict_key_size_bytes(model: torch.nn.Module, key: str, tensor: torch.Tensor) -> int:
    parts = key.split(".")
    try:
        module = _get_nested_module(model, parts[:-1])
        value = getattr(module, parts[-1])
    except Exception:
        value = tensor
    if torch.is_tensor(value) and hasattr(value, "ds_numel"):
        return int(value.ds_numel) * int(value.element_size())
    return int(tensor.numel() * tensor.element_size())


def _hf_shard_filename(index: int, total: int, safe_serialization: bool) -> str:
    prefix = "model" if safe_serialization else "pytorch_model"
    ext = "safetensors" if safe_serialization else "bin"
    return f"{prefix}-{index:05d}-of-{total:05d}.{ext}"


def _planned_state_dict_shards(
    model: torch.nn.Module,
    max_shard_size: str,
    safe_serialization: bool,
) -> List[Tuple[str, List[str]]]:
    max_bytes = _parse_size_to_bytes(max_shard_size)
    shards: List[List[str]] = []
    current: List[str] = []
    current_bytes = 0
    for key, tensor in model.state_dict().items():
        size = _state_dict_key_size_bytes(model, key, tensor)
        if current and current_bytes + size > max_bytes:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(key)
        current_bytes += size
    if current:
        shards.append(current)
    total = len(shards)
    return [(_hf_shard_filename(i + 1, total, safe_serialization), keys) for i, keys in enumerate(shards)]


def export_zero3_16bit_sharded_model(
    save_model: torch.nn.Module,
    output_dir: str,
    runtime: RuntimeContext,
    max_shard_size: str = "5GB",
    safe_serialization: bool = True,
) -> Dict[str, Any]:
    """Export a ZeRO-3 model shard-by-shard for vLLM/HF loading."""

    if zero is None:
        raise RuntimeError("DeepSpeed ZeRO is required for sharded full-model export")
    if safe_serialization:
        from safetensors.torch import save_file

    shards = _planned_state_dict_shards(save_model, max_shard_size, safe_serialization)
    planned_keys = {key for _, keys in shards for key in keys}
    config = getattr(save_model, "config", None)
    if (
        hasattr(save_model, "lm_head")
        and not bool(getattr(config, "tie_word_embeddings", False))
        and "lm_head.weight" not in planned_keys
    ):
        raise RuntimeError(
            "Refusing to export an untied causal LM without lm_head.weight"
        )
    weight_map: Dict[str, str] = {}
    total_size = 0
    if runtime.is_main:
        os.makedirs(output_dir, exist_ok=True)
    barrier(runtime)

    for filename, keys in shards:
        shard_state = OrderedDict() if runtime.is_main else None
        for key in keys:
            parts = key.split(".")
            module = _get_nested_module(save_model, parts[:-1])
            value = getattr(module, parts[-1])
            gather_values = [value] if torch.is_tensor(value) and hasattr(value, "ds_id") else []
            context = zero.GatheredParameters(gather_values, modifier_rank=0) if gather_values else nullcontext()
            with context:
                if runtime.is_main:
                    tensor = value.detach().cpu().clone() if torch.is_tensor(value) else save_model.state_dict()[key].detach().cpu().clone()
                    shard_state[key] = tensor
                    weight_map[key] = filename
                    total_size += int(tensor.numel() * tensor.element_size())
        if runtime.is_main:
            path = os.path.join(output_dir, filename)
            if safe_serialization:
                save_file(shard_state, path, metadata={"format": "pt"})
            else:
                torch.save(shard_state, path)
            del shard_state
        barrier(runtime)

    index_name = "model.safetensors.index.json" if safe_serialization else "pytorch_model.bin.index.json"
    if runtime.is_main:
        write_json(
            os.path.join(output_dir, index_name),
            {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            },
        )
    barrier(runtime)
    return {
        "hf_model_exported": True,
        "full_model_eval_ready": runtime.is_main,
        "shard_count": len(shards),
        "index_file": index_name,
        "safe_serialization": safe_serialization,
    }


def save_full_model_distributed(
    engine_or_model,
    save_model,
    tokenizer,
    output_dir: str,
    cfg: Dict[str, Any],
    metadata: Dict[str, Any],
    runtime: RuntimeContext,
    ds_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Save a full-parameter run without forcing a large ZeRO-3 gather.

    The default for large-model full SFT is a DeepSpeed checkpoint.  That is
    the safest format during training because gathering a 35B bf16 model into a
    single HF state dict can require a large host-memory spike.  Set
    ``output.full_model_export: auto`` or ``hf`` to additionally try writing
    vLLM-loadable HF weights.
    """

    output_cfg = cfg.get("output", {})
    export_mode = str(output_cfg.get("full_model_export", "checkpoint")).lower().strip()
    if export_mode not in {"checkpoint", "auto", "hf", "disabled", "none", "false"}:
        raise ValueError("output.full_model_export must be one of checkpoint, auto, hf, disabled")

    status: Dict[str, Any] = {
        "full_model_export": export_mode,
        "deepspeed_checkpoint_dir": "",
        "hf_model_exported": False,
        "hf_model_export_error": "",
        "full_model_eval_ready": False,
    }
    if runtime.is_main:
        os.makedirs(output_dir, exist_ok=True)
    barrier(runtime)

    save_disabled = export_mode in {"disabled", "none", "false"}
    if ds_config is not None and hasattr(engine_or_model, "save_checkpoint") and not save_disabled:
        checkpoint_dir = os.path.join(output_dir, "deepspeed_checkpoint")
        engine_or_model.save_checkpoint(
            checkpoint_dir,
            tag="final",
            client_state={"step4_metadata": metadata},
        )
        status["deepspeed_checkpoint_dir"] = checkpoint_dir
    elif runtime.is_main and export_mode not in {"disabled", "none", "false"}:
        save_model.save_pretrained(
            output_dir,
            safe_serialization=True,
            max_shard_size=output_cfg.get("max_shard_size", "5GB"),
        )
        status["hf_model_exported"] = True

    should_export_hf = export_mode in {"auto", "hf"} or (
        ds_config is None and export_mode not in {"disabled", "none", "false"}
    )
    if should_export_hf and ds_config is not None and hasattr(engine_or_model, "save_16bit_model"):
        try:
            if uses_zero3(ds_config):
                export_status = export_zero3_16bit_sharded_model(
                    save_model,
                    output_dir,
                    runtime,
                    max_shard_size=output_cfg.get("max_shard_size", "5GB"),
                    safe_serialization=bool(output_cfg.get("safe_serialization", True)),
                )
                status.update(export_status)
            else:
                ok = bool(engine_or_model.save_16bit_model(output_dir, save_filename="pytorch_model.bin"))
                status["hf_model_exported"] = ok
                if export_mode == "hf" and not ok:
                    raise RuntimeError(
                        "DeepSpeed save_16bit_model returned False. For ZeRO-3 this usually "
                        "means stage3_gather_16bit_weights_on_model_save is disabled."
                    )
        except Exception as err:
            status["hf_model_export_error"] = str(err)
            logger.exception("Failed to export full HF model from DeepSpeed checkpoint")
            if export_mode == "hf":
                raise

    barrier(runtime)
    if runtime.is_main:
        try:
            get_text_config(save_model.config).save_pretrained(output_dir)
        except Exception:
            logger.exception("Failed to save model config to %s", output_dir)
        tokenizer.save_pretrained(output_dir)
        status["full_model_eval_ready"] = is_full_model_eval_ready(output_dir)
        status["deepspeed_checkpoint_ready"] = is_full_model_checkpoint_ready(
            output_dir,
            expected_world_size=runtime.world_size,
        )
        summary_path = os.path.join(output_dir, "full_model_save_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2, ensure_ascii=False)
    if runtime.distributed and torch.distributed.is_initialized():
        payload = [status]
        torch.distributed.broadcast_object_list(payload, src=0)
        status = payload[0]
    return status


def compute_num_training_steps(n_batches: int, cfg: Dict[str, Any]) -> int:
    train_cfg = cfg["training"]
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    epochs = float(train_cfg.get("epochs", 1.0))
    steps_per_epoch = math.ceil(n_batches / grad_accum)
    total_steps = max(1, math.ceil(steps_per_epoch * epochs))
    max_steps = train_cfg.get("max_steps")
    if max_steps is not None:
        total_steps = min(total_steps, int(max_steps))
    return total_steps


def compute_num_training_microbatches(n_batches: int, cfg: Dict[str, Any]) -> int:
    train_cfg = cfg["training"]
    epochs = float(train_cfg.get("epochs", 1.0))
    total_microbatches = max(1, math.ceil(n_batches * epochs))
    max_steps = train_cfg.get("max_steps")
    if max_steps is not None:
        total_microbatches = min(
            total_microbatches,
            int(max_steps) * int(train_cfg.get("gradient_accumulation_steps", 1)),
        )
    return total_microbatches


def resolve_warmup_steps(cfg: Dict[str, Any], total_steps: int) -> int:
    train_cfg = cfg["training"]
    warmup_ratio = train_cfg.get("warmup_ratio")
    if warmup_ratio is not None:
        ratio = float(warmup_ratio)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("training.warmup_ratio must be in [0, 1]")
        warmup_steps = int(math.ceil(total_steps * ratio))
        if ratio > 0.0 and total_steps > 0:
            warmup_steps = max(1, warmup_steps)
        warmup_steps = min(warmup_steps, total_steps)
        train_cfg["warmup_steps"] = warmup_steps
        return warmup_steps
    return int(train_cfg.get("warmup_steps", 0))


def train(args: argparse.Namespace) -> None:
    runtime = setup_distributed_from_env()
    cfg = load_yaml(args.config)
    if args.group not in cfg.get("experiments", {}):
        raise ValueError(f"Unknown Step 4 group {args.group!r}; available={list(cfg.get('experiments', {}))}")
    apply_cli_overrides(cfg, args)
    tuning_mode = resolve_tuning_mode(cfg)

    model_name, model_path = resolve_model(cfg, args)
    applied_memory_safety = apply_qwen35_large_model_memory_safety_overrides(cfg, model_name)
    applied_global_batch = apply_global_batch_size_override(cfg, runtime.world_size)
    output_dir = build_output_dir(cfg, args, model_name)
    if runtime.is_main:
        maybe_prepare_output_dir(output_dir, args.overwrite)
    barrier(runtime)
    log_file = os.path.join(output_dir, "train.log")
    setup_rank_logger(log_file, runtime.rank)

    group_cfg = cfg["experiments"][args.group]
    train_cfg = cfg["training"]
    beta = resolved_group_beta(group_cfg)
    group_cfg["beta"] = beta
    token_scope = str(group_cfg.get("token_scope", "assistant_prediction"))
    injection_front_token_count = group_cfg.get("injection_front_token_count")
    if injection_front_token_count is not None:
        injection_front_token_count = int(injection_front_token_count)
    injection_targets = normalize_injection_targets(
        group_cfg.get("injection_targets", DEFAULT_INJECTION_TARGETS)
    )
    group_cfg["injection_targets"] = list(injection_targets)
    feature_ids_for_run = resolved_feature_ids_for_run(cfg, group_cfg, model_name)
    feature_caveat = feature_caveat_for_run(model_name, feature_ids_for_run, group_cfg)
    ds_config = load_deepspeed_config(cfg)
    if ds_config is not None:
        ds_config = sync_deepspeed_config(ds_config, cfg)
    validate_full_tuning_deepspeed_config(cfg, ds_config, model_name)
    disabled_checkpointing = disable_incompatible_zero3_checkpointing(cfg, ds_config)
    seed = int(train_cfg.get("seed", 1234))
    random.seed(seed + runtime.rank)
    torch.manual_seed(seed + runtime.rank)
    torch.cuda.manual_seed_all(seed)
    beta_rng = random.Random(seed + 1009 * (runtime.rank + 1))

    print_main(runtime, "")
    print_main(runtime, "=" * 64)
    print_main(runtime, "  [step4] sycophancy SFT")
    print_main(runtime, f"  model : {model_name}")
    print_main(runtime, f"  tuning: {tuning_mode}")
    print_main(runtime, f"  group : {args.group}")
    print_main(runtime, f"  target: {group_cfg['target']}")
    print_main(runtime, f"  inject: {group_cfg.get('injection', 'none')}")
    print_main(runtime, f"  beta  : {beta:g}")
    if group_cfg.get("injection", "none") != "none":
        print_main(runtime, f"  scope : {token_scope}")
        print_main(runtime, f"  inject targets: {','.join(injection_targets)}")
        print_main(
            runtime,
            "  inject front: "
            + (str(injection_front_token_count) if injection_front_token_count is not None else "all"),
        )
        if feature_ids_for_run:
            print_main(runtime, f"  feature: {feature_ids_for_run}")
        if feature_caveat:
            print_main(runtime, f"  caveat: {feature_caveat}")
    print_main(runtime, f"  ds    : {train_cfg.get('deepspeed_config') or 'disabled'}")
    print_main(
        runtime,
        "  batch : "
        f"micro={int(train_cfg.get('per_device_batch_size', 1))}, "
        f"accum={int(train_cfg.get('gradient_accumulation_steps', 1))}"
    )
    print_main(runtime, f"  maxlen: {int(train_cfg.get('max_length', 2048))}")
    if disabled_checkpointing:
        print_main(runtime, "  ckpt  : disabled for ZeRO-3 compatibility")
    if applied_memory_safety:
        print_main(runtime, f"  mem   : {train_cfg.get('memory_safety_override', {}).get('reason')}")
    if applied_global_batch:
        print_main(runtime, f"  gbatch: {train_cfg.get('global_batch_size')}")
    if group_cfg["target"] == "syco":
        weighted_loss = bool(train_cfg.get("weighted_syco_front_loss", False))
        print_main(
            runtime,
            "  syco : "
            f"source=dataset_response, "
            f"loss={'front_weighted' if weighted_loss else 'regular_ce'}"
        )
    print_main(runtime, f"  ranks : {runtime.world_size}")
    print_main(runtime, f"  output: {output_dir}")
    print_main(runtime, "=" * 64)
    logger.info("Resolved model: %s (%s)", model_name, model_path)
    logger.info("Output directory: %s", output_dir)
    if feature_caveat:
        logger.warning(feature_caveat)
    if applied_memory_safety:
        logger.info("Applied memory safety override: %s", train_cfg.get("memory_safety_override"))

    # Materialize/validate the shared split once before all ranks read it. This
    # avoids concurrent cache reconstruction on network filesystems when a
    # path alias or source fingerprint invalidates old split metadata.
    if runtime.distributed:
        if runtime.is_main and group_cfg["target"] == "syco":
            resolve_syco_train_dataset(cfg, model_name)
        barrier(runtime)
    examples = build_training_examples(cfg, args.group, model_name)
    print_main(runtime, "[1/4] Loading model and tokenizer ...")
    model, tokenizer = load_model(model_path, cfg, runtime, ds_config=ds_config)

    max_length = int(train_cfg.get("max_length", 2048))
    dataset = SFTDataset(examples, tokenizer, max_length=max_length, cfg=cfg)
    if runtime.is_main:
        save_dataset_manifest(os.path.join(output_dir, "train_dataset_manifest.jsonl"), examples, dataset)
    print_main(runtime, f"      train examples: {len(dataset)}")

    collator = SFTCollator(tokenizer)
    sampler = DistributedSampler(
        dataset,
        num_replicas=runtime.world_size,
        rank=runtime.rank,
        shuffle=True,
        seed=seed,
        drop_last=False,
    ) if runtime.distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("per_device_batch_size", 1)),
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collator,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )

    lora_cfg = None
    wrapped: List[str] = []
    if tuning_mode == TUNING_MODE_LORA:
        print_main(runtime, "[2/4] Installing LoRA adapters ...")
        for param in model.parameters():
            param.requires_grad_(False)
        lora_cfg = LoRAConfig.from_dict(cfg.get("lora", {}))
        wrapped = inject_lora_adapters(model, lora_cfg)
        set_trainable_lora_only(model)
        if not wrapped:
            raise RuntimeError("No LoRA target modules were wrapped; check lora.target_modules/module_filter")
        logger.info("LoRA wrapped %s modules", len(wrapped))
        print_main(runtime, f"      LoRA modules: {len(wrapped)}")
    else:
        print_main(runtime, "[2/4] Enabling full-parameter fine-tuning ...")
        set_trainable_full_model(model)
    param_summary = trainable_parameter_summary(model)
    logger.info("Parameter summary: %s", param_summary)
    print_main(
        runtime,
        "      trainable params: "
        f"{int(param_summary['trainable']):,} / {int(param_summary['total']):,} "
        f"({100.0 * param_summary['trainable_ratio']:.3f}%)"
    )

    steering_vec = build_train_steering_vector(cfg, group_cfg, model_name)
    hook = None
    if steering_vec is not None:
        layer = int(cfg["sae"]["configs"][model_name]["layer"])
        layer_module = get_layer_module(model, layer)
        hook = TrainTimeSteeringHook(
            layer_module=layer_module,
            steering_vec=steering_vec,
            beta=beta,
            mode=group_cfg.get("injection", "positive"),
            token_scope=token_scope,
        )
        logger.info(
            "Registered train-time steering hook: layer=%s beta=%s norm=%.6f injection=%s token_scope=%s injection_targets=%s",
            layer,
            beta,
            float(steering_vec.norm().item()),
            group_cfg.get("injection"),
            token_scope,
            ",".join(injection_targets),
        )

    total_steps = compute_num_training_steps(len(dataloader), cfg)
    total_microbatches = compute_num_training_microbatches(len(dataloader), cfg)
    warmup_steps = resolve_warmup_steps(cfg, total_steps)
    use_deepspeed = ds_config is not None
    optimizer = None if use_deepspeed else create_optimizer(model, cfg)

    def build_scheduler(optimizer_obj):
        return get_scheduler(
            name=train_cfg.get("lr_scheduler", "cosine"),
            optimizer=optimizer_obj,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    scheduler = build_scheduler if use_deepspeed else build_scheduler(optimizer)
    model, optimizer, scheduler = initialize_deepspeed_engine(
        model,
        optimizer,
        scheduler,
        cfg,
        runtime,
        ds_config,
    )

    input_device = runtime.device if ds_config is not None else primary_input_device(model)
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    amp_dtype = torch.bfloat16 if train_cfg.get("mixed_precision", "bf16") == "bf16" else torch.float16
    use_amp = (
        not use_deepspeed
        and input_device.type == "cuda"
        and train_cfg.get("mixed_precision", "bf16") in {"bf16", "fp16"}
    )
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if use_amp else nullcontext()
    )

    print_main(runtime, "[3/4] Fine-tuning ...")
    model.train()
    start_time = time.time()
    history: List[Dict[str, Any]] = []
    completed_steps = 0
    consumed_microbatches = 0
    running_loss = 0.0
    running_count = 0
    running_beta = 0.0
    injection_microbatches = 0
    injection_token_sum = 0.0
    injection_supervised_token_sum = 0.0
    injection_eligible_supervised_token_sum = 0.0
    injection_beta_sum = 0.0
    progress_path = os.path.join(output_dir, "training_progress.json")
    epochs = max(1, math.ceil(float(train_cfg.get("epochs", 1.0))))
    if runtime.is_main:
        save_training_progress(
            progress_path,
            {
                "status": "training",
                "run_name": args.run_name or os.path.basename(output_dir.rstrip(os.sep)),
                "group": args.group,
                "step": 0,
                "total_steps": total_steps,
                "epoch": 0,
                "epochs": float(train_cfg.get("epochs", 1.0)),
                "loss": None,
                "lr": None,
                "beta": beta,
                "elapsed_sec": 0.0,
            },
        )
    progress = tqdm(
        total=total_steps,
        desc=f"Step4 {args.group}",
        unit="step",
        dynamic_ncols=True,
        disable=not runtime.is_main,
    )
    if not use_deepspeed:
        optimizer.zero_grad(set_to_none=True)

    for epoch_idx in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch_idx)
        for batch_idx, batch in enumerate(dataloader):
            if completed_steps >= total_steps or consumed_microbatches >= total_microbatches:
                break
            batch = move_batch_to_device(batch, input_device)
            assistant_mask = batch.pop("assistant_mask")
            loss_weights = batch.pop("loss_weights")
            target_types = batch.pop("target_types")
            batch.pop("sample_ids", None)
            current_beta = beta
            if hook is not None:
                current_beta = sample_train_beta(group_cfg, beta, beta_rng)
                steering_mask = build_train_time_steering_mask(
                    assistant_mask=assistant_mask,
                    labels=batch["labels"],
                    token_scope=token_scope,
                )
                steering_mask = limit_mask_to_front_positions(
                    steering_mask,
                    injection_front_token_count,
                )
                injection_target_mask = build_injection_target_mask(
                    assistant_mask=assistant_mask,
                    target_types=target_types,
                    injection_targets=injection_targets,
                )
                steering_mask = steering_mask & injection_target_mask
                hook.set_beta(current_beta)
                hook.set_mask(steering_mask)
                injection_microbatches += 1
                injection_token_sum += float(steering_mask.sum().detach().float().item())
                injection_supervised_token_sum += float(
                    (batch["labels"] != IGNORE_INDEX).sum().detach().float().item()
                )
                injection_eligible_supervised_token_sum += float(
                    ((batch["labels"] != IGNORE_INDEX) & injection_target_mask)
                    .sum()
                    .detach()
                    .float()
                    .item()
                )
                injection_beta_sum += float(current_beta)
            if use_deepspeed and hasattr(model, "set_gradient_accumulation_boundary"):
                final_microbatch = consumed_microbatches + 1 >= total_microbatches
                model.set_gradient_accumulation_boundary(
                    ((batch_idx + 1) % grad_accum == 0)
                    or (batch_idx + 1 == len(dataloader))
                    or final_microbatch
                )
            with autocast_context:
                labels = batch["labels"]
                outputs = model(**batch, use_cache=False)
                loss = weighted_causal_lm_loss(outputs.logits, labels, loss_weights)
                scaled_loss = loss / grad_accum
            if use_deepspeed:
                model.backward(loss)
            else:
                scaled_loss.backward()

            loss_value = reduce_mean_scalar(float(loss.detach().float().item()), runtime)
            running_loss += loss_value
            running_count += 1
            running_beta += float(current_beta)
            consumed_microbatches += 1

            should_step = (
                ((batch_idx + 1) % grad_accum == 0)
                or (batch_idx + 1 == len(dataloader))
                or (consumed_microbatches >= total_microbatches)
            )
            if should_step:
                if use_deepspeed:
                    model.step()
                    step_applied = bool(getattr(model, "_step_applied", True))
                else:
                    grad_norm = train_cfg.get("max_grad_norm")
                    if grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad],
                            float(grad_norm),
                        )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step_applied = True
                if not step_applied:
                    continue
                completed_steps += 1
                mean_loss = running_loss / max(1, running_count)
                mean_beta = running_beta / max(1, running_count)
                lr = optimizer_learning_rate(optimizer, scheduler)
                elapsed = time.time() - start_time
                if runtime.is_main:
                    progress_row = {
                        "status": "training",
                        "run_name": args.run_name or os.path.basename(output_dir.rstrip(os.sep)),
                        "group": args.group,
                        "step": completed_steps,
                        "total_steps": total_steps,
                        "epoch": epoch_idx + 1,
                        "epochs": float(train_cfg.get("epochs", 1.0)),
                        "loss": mean_loss,
                        "lr": lr,
                        "beta": mean_beta,
                        "elapsed_sec": round(elapsed, 2),
                    }
                    history.append(
                        {
                            "step": progress_row["step"],
                            "epoch": progress_row["epoch"],
                            "loss": progress_row["loss"],
                            "lr": progress_row["lr"],
                            "beta": progress_row["beta"],
                            "elapsed_sec": progress_row["elapsed_sec"],
                        }
                    )
                    save_training_progress(progress_path, progress_row)
                    progress.set_postfix(loss=f"{mean_loss:.4f}", lr=f"{lr:.2e}", beta=f"{mean_beta:g}")
                    progress.update(1)
                running_loss = 0.0
                running_count = 0
                running_beta = 0.0

        if completed_steps >= total_steps or consumed_microbatches >= total_microbatches:
            break

    progress.close()
    if hook is not None:
        hook.remove()
    global_injection_microbatches = reduce_sum_scalar(injection_microbatches, runtime)
    global_injection_token_sum = reduce_sum_scalar(injection_token_sum, runtime)
    global_injection_supervised_token_sum = reduce_sum_scalar(injection_supervised_token_sum, runtime)
    global_injection_eligible_supervised_token_sum = reduce_sum_scalar(
        injection_eligible_supervised_token_sum,
        runtime,
    )
    global_injection_beta_sum = reduce_sum_scalar(injection_beta_sum, runtime)
    injection_stats = {
        "registered": hook is not None,
        "token_scope": token_scope,
        "front_token_count": injection_front_token_count,
        "targets": list(injection_targets),
        "microbatches": int(global_injection_microbatches),
        "injected_hidden_positions": int(global_injection_token_sum),
        "supervised_tokens_seen": int(global_injection_supervised_token_sum),
        "eligible_supervised_tokens_seen": int(global_injection_eligible_supervised_token_sum),
        "coverage": (
            global_injection_token_sum / global_injection_supervised_token_sum
            if global_injection_supervised_token_sum > 0 else 0.0
        ),
        "eligible_coverage": (
            global_injection_token_sum / global_injection_eligible_supervised_token_sum
            if global_injection_eligible_supervised_token_sum > 0 else 0.0
        ),
        "mean_sampled_beta": (
            global_injection_beta_sum / global_injection_microbatches
            if global_injection_microbatches > 0 else 0.0
        ),
    }
    barrier(runtime)

    print_main(runtime, "[4/4] Saving model/checkpoint and metadata ...")
    save_model = model.module if use_deepspeed and hasattr(model, "module") else model
    export_model = state_dict_export_model(save_model)
    metadata = {
        "tuning_mode": tuning_mode,
        "model_name": model_name,
        "model_path": model_path,
        "group": args.group,
        "run_name": args.run_name or os.path.basename(output_dir.rstrip(os.sep)),
        "auto_run_name": build_auto_run_name(cfg, args, model_name),
        "target": group_cfg["target"],
        "injection": group_cfg.get("injection", "none"),
        "beta": beta,
        "beta_schedule": group_cfg.get("beta_schedule", "fixed"),
        "beta_min": group_cfg.get("beta_min"),
        "beta_max": group_cfg.get("beta_max"),
        "beta_values": group_cfg.get("beta_values"),
        "train_time_injection": injection_stats,
        "feature_caveat": feature_caveat,
        "deepspeed_config": train_cfg.get("deepspeed_config"),
        "per_device_batch_size": int(train_cfg.get("per_device_batch_size", 1)),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 1)),
        "global_effective_batch_size": (
            int(train_cfg.get("per_device_batch_size", 1))
            * int(train_cfg.get("gradient_accumulation_steps", 1))
            * int(runtime.world_size)
        ),
        "requested_global_batch_size": train_cfg.get("global_batch_size"),
        "global_batch_size_override": train_cfg.get("global_batch_size_override"),
        "epochs": float(train_cfg.get("epochs", 1.0)),
        "learning_rate": float(train_cfg.get("learning_rate", 1e-5)),
        "warmup_steps": int(train_cfg.get("warmup_steps", 0)),
        "warmup_ratio": train_cfg.get("warmup_ratio"),
        "linear_attention_fast_path": train_cfg.get("linear_attention_fast_path", "available"),
        "memory_safety_override": train_cfg.get("memory_safety_override"),
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", False)),
        "gradient_checkpointing_disabled_reason": train_cfg.get("gradient_checkpointing_disabled_reason"),
        "world_size": runtime.world_size,
        "lora": lora_cfg.to_dict() if lora_cfg is not None else None,
        "trainable_parameter_summary": param_summary,
        "feature": cfg.get("feature", {}),
        "sae": cfg.get("sae", {}).get("configs", {}).get(model_name, {}),
        "syco_train_dataset": cfg.get("data", {}).get("resolved_syco_train_dataset", {}),
        "syco_training_objective": {
            "weighted_syco_front_loss": bool(train_cfg.get("weighted_syco_front_loss", False)),
            "syco_front_token_count": int(train_cfg.get("syco_front_token_count", 128)),
            "syco_front_token_weight": float(train_cfg.get("syco_front_token_weight", 8.0)),
        },
        "train_examples": len(dataset),
        "wrapped_lora_modules": len(wrapped),
        "total_steps": completed_steps,
        "elapsed_sec": round(time.time() - start_time, 2),
    }
    save_status: Dict[str, Any] = {}
    if tuning_mode == TUNING_MODE_LORA:
        if lora_cfg is None:
            raise RuntimeError("LoRA config unexpectedly missing in LoRA tuning mode")
        save_lora_adapter_distributed(save_model, output_dir, lora_cfg, metadata, runtime)
        save_status = {
            "adapter_model": os.path.join(output_dir, "adapter_model.pt"),
            "adapter_config": os.path.join(output_dir, "adapter_config.json"),
        }
    else:
        save_status = save_full_model_distributed(
            model,
            export_model,
            tokenizer,
            output_dir,
            cfg,
            metadata,
            runtime,
            ds_config,
        )
        metadata["full_model_save"] = save_status
    if runtime.is_main:
        tokenizer.save_pretrained(output_dir)
        with open(os.path.join(output_dir, "step4_config_resolved.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        save_loss_history(os.path.join(output_dir, "loss_history.csv"), history)
        save_training_summary(os.path.join(output_dir, "training_summary.json"), metadata)
        last_progress = history[-1] if history else {}
        save_training_progress(
            progress_path,
            {
                "status": "done",
                "run_name": args.run_name or os.path.basename(output_dir.rstrip(os.sep)),
                "group": args.group,
                "step": int(last_progress.get("step", completed_steps)),
                "total_steps": total_steps,
                "epoch": last_progress.get("epoch", float(train_cfg.get("epochs", 1.0))),
                "epochs": float(train_cfg.get("epochs", 1.0)),
                "loss": last_progress.get("loss"),
                "lr": last_progress.get("lr"),
                "beta": last_progress.get("beta", beta),
                "elapsed_sec": last_progress.get("elapsed_sec", round(time.time() - start_time, 2)),
            },
        )

    if args.save_merged:
        if tuning_mode != TUNING_MODE_LORA:
            raise RuntimeError("--save-merged only applies to LoRA tuning mode")
        if uses_zero3(ds_config):
            raise RuntimeError("--save-merged is not supported with ZeRO-3 in this lightweight Step 4 launcher")
        merged_dir = os.path.join(output_dir, "merged_model")
        if runtime.is_main:
            merged_count = merge_lora_adapters(save_model)
            save_model.save_pretrained(
                merged_dir,
                safe_serialization=True,
                max_shard_size=cfg.get("output", {}).get("max_shard_size", "5GB"),
            )
            tokenizer.save_pretrained(merged_dir)
            logger.info("Saved merged model to %s after merging %s modules", merged_dir, merged_count)

    print_main(runtime, "")
    print_main(runtime, f"Done in {(time.time() - start_time) / 60.0:.1f} min.")
    if tuning_mode == TUNING_MODE_LORA:
        print_main(runtime, f"  Adapter: {os.path.join(output_dir, 'adapter_model.pt')}")
    else:
        if save_status.get("full_model_eval_ready"):
            print_main(runtime, f"  HF model: {output_dir}")
        if save_status.get("deepspeed_checkpoint_dir"):
            print_main(runtime, f"  DS ckpt : {save_status.get('deepspeed_checkpoint_dir')}")
        if save_status.get("hf_model_export_error"):
            print_main(runtime, f"  HF save : failed ({save_status.get('hf_model_export_error')})")
    print_main(runtime, f"  Summary: {os.path.join(output_dir, 'training_summary.json')}")
    print_main(runtime, f"  Losses : {os.path.join(output_dir, 'loss_history.csv')}")
    print_main(runtime, f"  Log    : {log_file}")
    barrier(runtime)


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
