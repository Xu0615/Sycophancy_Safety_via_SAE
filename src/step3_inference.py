"""Step 3 — Steered inference via vLLM with forward hooks.

Uses vLLM with enforce_eager=True to enable PyTorch forward hooks for
SAE feature steering.  The steering hook modifies the residual stream at
the target layer during every forward pass (including each autoregressive
generation step).

Produces the same output format as Step 1 (sample_id, prompt_text,
response_text, thinking_text, full_response) so the downstream judge /
filter / report modules can be reused unchanged.
"""

import logging
import os
from typing import Any, Dict, Optional

import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer

from src.utils import (
    append_checkpoint,
    checkpoint_to_dataframe,
    load_checkpoint,
    save_parquet,
    split_generation_text,
)

logger = logging.getLogger(__name__)

# ====================================================================
# vLLM initialization for steering
# ====================================================================

def load_vllm_for_steering(
    model_path: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 16384,
):
    """Load vLLM engine with enforce_eager=True for forward-hook steering.

    Returns:
        (llm, tokenizer, torch_model) where torch_model is the underlying
        nn.Module extracted from vLLM internals for hook registration.
    """
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import LLM

    logger.info(f"Loading vLLM engine from {model_path}")
    logger.info(f"  enforce_eager=True (required for forward hooks)")
    logger.info(f"  gpu_memory_utilization={gpu_memory_utilization}, "
                f"max_model_len={max_model_len}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True,
    )

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        disable_log_stats=True,
    )

    torch_model = _get_vllm_torch_model(llm)
    logger.info("vLLM engine loaded successfully (enforce_eager=True)")
    return llm, tokenizer, torch_model


def _get_vllm_torch_model(llm) -> nn.Module:
    """Extract the underlying PyTorch nn.Module from a vLLM LLM object.

    Supports vLLM V1 (0.19.x) and older V0 (0.6.x) attribute chains.
    """
    chains = [
        # vLLM V1 (0.19.x)
        ["llm_engine", "model_executor", "driver_worker", "model_runner", "model"],
        # vLLM V1: through InprocClient -> EngineCore
        ["llm_engine", "engine_core", "engine_core", "model_executor",
         "driver_worker", "model_runner", "model"],
        # vLLM V1: WorkerWrapperBase proxies to worker
        ["llm_engine", "model_executor", "driver_worker", "worker",
         "model_runner", "model"],
        # vLLM V0 (0.6.x) legacy chains
        ["llm_engine", "model_executor", "driver_worker", "model_runner",
         "model", "model"],
    ]
    errors = []
    for ch in chains:
        try:
            cur = llm
            for attr in ch:
                cur = getattr(cur, attr)
            if isinstance(cur, nn.Module):
                return cur
        except Exception as e:
            errors.append(f"  Chain {'.'.join(ch)}: {e}")
    raise RuntimeError(
        "Could not locate torch model inside vLLM LLM.\n" + "\n".join(errors)
    )


def get_layer_module(torch_model: nn.Module, layer: int) -> nn.Module:
    """Get the transformer block at a given layer index.

    Handles Qwen3.5 (language_model.model.layers), Qwen2 (model.layers),
    and other common architectures.
    """
    layer = int(layer)
    # Qwen3.5: ConditionalGeneration -> language_model -> model -> layers
    if hasattr(torch_model, "language_model"):
        lm = torch_model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers[layer]
    # Qwen2 / generic: model -> model -> layers
    if hasattr(torch_model, "model") and hasattr(torch_model.model, "layers"):
        return torch_model.model.layers[layer]
    # Direct layers attribute
    if hasattr(torch_model, "layers"):
        return torch_model.layers[layer]
    raise AttributeError("Could not locate transformer layers in model.")


# ====================================================================
# Text cleaning
# ====================================================================

def _clean_and_split(raw: str) -> Dict[str, Any]:
    """Parse generated text into thinking_text and response_text."""
    return split_generation_text(raw)


# ====================================================================
# Steered inference with vLLM
# ====================================================================

def run_steered_inference(
    llm,
    tokenizer,
    df_prompts: pd.DataFrame,
    output_dir: str,
    max_new_tokens: int = 4096,
    temperature: float = 0.6,
    top_p: float = 0.9,
    batch_size: int = 512,
    dry_run: bool = False,
    dry_run_samples: int = 5,
) -> pd.DataFrame:
    """Run inference with the (already-hooked) vLLM model.

    The steering hook must be registered on the model BEFORE calling this.

    Args:
        llm:            vLLM LLM instance (with steering hook active)
        tokenizer:      tokenizer for prompt formatting
        df_prompts:     DataFrame with sample_id, prompt_text, source_dataset
        output_dir:     directory for checkpoints and output parquet
        max_new_tokens: max tokens to generate per sample
        temperature:    sampling temperature
        top_p:          nucleus sampling p
        batch_size:     checkpoint granularity (vLLM handles GPU batching)
        dry_run:        if True, only process a few samples
        dry_run_samples: number of samples for dry run

    Returns:
        DataFrame with inference results
    """
    from vllm import SamplingParams

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "inference_checkpoint.jsonl")
    output_path = os.path.join(output_dir, "model_outputs.parquet")

    if dry_run:
        df_prompts = df_prompts.head(dry_run_samples)
    expected_ids = set(df_prompts["sample_id"])

    done_ids = load_checkpoint(checkpoint_path)
    logger.info(f"Already completed: {len(done_ids)} samples")
    pending = df_prompts[~df_prompts["sample_id"].isin(done_ids)].reset_index(drop=True)
    logger.info(f"Pending: {len(pending)} samples")

    if len(pending) == 0:
        logger.info("All samples already processed, loading from checkpoint")
        df_results = checkpoint_to_dataframe(checkpoint_path)
        df_results = df_results[df_results["sample_id"].isin(expected_ids)].reset_index(drop=True)
        save_parquet(df_results, output_path)
        return df_results

    sp = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )

    n_batches = (len(pending) + batch_size - 1) // batch_size
    pbar = tqdm(total=len(pending), desc="Steered generation", unit="sample")

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(pending))
        batch_rows = pending.iloc[start:end]

        formatted_prompts = []
        for _, row in batch_rows.iterrows():
            messages = [{"role": "user", "content": row["prompt_text"]}]
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=True,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            formatted_prompts.append(text)

        outputs = llm.generate(formatted_prompts, sp, use_tqdm=False)

        for i, row in enumerate(batch_rows.itertuples()):
            raw_text = outputs[i].outputs[0].text
            parsed = _clean_and_split(raw_text)

            record = {
                "sample_id": row.sample_id,
                "source_dataset": row.source_dataset,
                "prompt_text": row.prompt_text,
                "response_text": parsed.get("response_text", ""),
                "thinking_text": parsed.get("thinking_text"),
                "full_response": parsed.get("full_response", ""),
                "input_tokens": len(outputs[i].prompt_token_ids),
                "output_tokens": len(outputs[i].outputs[0].token_ids),
                "error": None,
            }
            for col in (
                "syco_id",
                "domain",
                "eval_probe_type",
                "target_direction",
            ):
                if hasattr(row, col):
                    record[col] = getattr(row, col)
            append_checkpoint(checkpoint_path, record)
            pbar.update(1)

    pbar.close()
    logger.info("Steered inference complete")

    df_results = checkpoint_to_dataframe(checkpoint_path)
    df_results = df_results[df_results["sample_id"].isin(expected_ids)].reset_index(drop=True)
    save_parquet(df_results, output_path)
    return df_results
