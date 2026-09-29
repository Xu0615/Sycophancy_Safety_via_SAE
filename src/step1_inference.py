"""
Step 1 — Local model inference via vLLM.

Loads a local Qwen model with vLLM for high-throughput generation and produces
responses for harmful prompts.  Supports resumable runs via JSONL checkpoints.

Thinking chain: For Qwen3/3.5 models the ``<think>...</think>`` block is
preserved in ``thinking_text`` and also kept in ``full_response``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from src.utils import (
    append_checkpoint,
    checkpoint_to_dataframe,
    load_checkpoint,
    split_generation_text,
    save_parquet,
    setup_logger,
)

logger = setup_logger("step1_inference")


# ====================================================================
# vLLM engine loading
# ====================================================================

def _import_vllm() -> tuple[Any, Any]:
    """Import vLLM lazily so non-vLLM callers can use text parsing helpers."""
    from vllm import LLM, SamplingParams

    return LLM, SamplingParams


def load_vllm_engine(
    model_path: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 16384,
) -> tuple:
    """Load vLLM engine and a standalone tokenizer for prompt formatting."""
    LLM, _ = _import_vllm()
    logger.info(f"Loading vLLM engine from {model_path}")
    logger.info(f"  tensor_parallel_size={tensor_parallel_size}, "
                f"gpu_memory_utilization={gpu_memory_utilization}, "
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
    )

    logger.info("vLLM engine loaded successfully")
    return llm, tokenizer


# ====================================================================
# Text cleaning (backend-agnostic)
# ====================================================================

def _clean_and_split(raw: str) -> Dict[str, Any]:
    """Clean decoded text and split into thinking / response.

    Qwen3/3.5 chat templates put ``<think>\\n`` at the end of the generation
    prompt (i.e. inside the *input* tokens).  After we slice off the input,
    the decoded new tokens typically look like:

        ``thinking content here\\n</think>\\n\\nresponse here``

    So we must handle three cases:
      1. Both ``<think>`` and ``</think>`` present (full pair in output)
      2. Only ``</think>`` present (``<think>`` was in the input prompt)
      3. Only ``<think>`` present (generation was cut off mid-thought)
    """
    return split_generation_text(raw)


# ====================================================================
# vLLM generation
# ====================================================================

def generate_vllm(
    llm: LLM,
    tokenizer: Any,
    prompts: List[str],
    sampling_params: SamplingParams,
    use_tqdm: bool = True,
) -> List[Dict[str, Any]]:
    """Format prompts via chat template, generate with vLLM, parse results.

    Args:
        use_tqdm: Whether vLLM shows its internal progress bar. Set False when
                  the caller already has an outer progress bar.
    """
    # Format each prompt using the chat template
    formatted_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        formatted_prompts.append(input_text)

    # Single vLLM call for all prompts (continuous batching handled internally)
    outputs = llm.generate(formatted_prompts, sampling_params, use_tqdm=use_tqdm)

    results = []
    for output in outputs:
        raw_text = output.outputs[0].text
        result = _clean_and_split(raw_text)
        result["input_tokens"] = len(output.prompt_token_ids)
        result["output_tokens"] = len(output.outputs[0].token_ids)
        results.append(result)

    return results


# ====================================================================
# Main inference runner
# ====================================================================

def run_inference(
    input_path: str,
    output_path: str,
    model_path: str,
    checkpoint_path: str = "outputs/step1_inference_checkpoint.jsonl",
    max_new_tokens: int = 8192,
    temperature: float = 0.6,
    top_p: float = 0.9,
    do_sample: bool = True,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 16384,
    dry_run: bool = False,
    dry_run_samples: int = 5,
) -> pd.DataFrame:
    logger.info(f"Loading prompts from {input_path}")
    df_prompts = pd.read_parquet(input_path)
    logger.info(f"Total prompts: {len(df_prompts)}")

    if dry_run:
        df_prompts = df_prompts.head(dry_run_samples)
        logger.info(f"[DRY RUN] Using only {len(df_prompts)} samples")
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

    _, SamplingParams = _import_vllm()

    # Load vLLM engine
    llm, tokenizer = load_vllm_engine(
        model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )

    # Build sampling params
    sampling_params = SamplingParams(
        temperature=temperature if do_sample else 0,
        top_p=top_p if do_sample else 1.0,
        max_tokens=max_new_tokens,
    )
    logger.info(f"SamplingParams: temperature={sampling_params.temperature}, "
                f"top_p={sampling_params.top_p}, max_tokens={sampling_params.max_tokens}")

    # Generate in batches so that checkpoints are written incrementally.
    # vLLM handles continuous batching *within* each llm.generate() call,
    # so the batch size here only controls checkpoint granularity.
    BATCH_SIZE = 512
    prompts_all = pending["prompt_text"].tolist()
    n_batches = (len(prompts_all) + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(f"Starting vLLM generation for {len(prompts_all)} prompts "
                f"in {n_batches} batches (batch_size={BATCH_SIZE}) ...")

    pbar = tqdm(total=len(prompts_all), desc="Generating", unit="sample")
    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(prompts_all))
        batch_prompts = prompts_all[start:end]

        results = generate_vllm(llm, tokenizer, batch_prompts, sampling_params)

        for i, res in enumerate(results):
            row = pending.iloc[start + i]
            record = {
                "sample_id": row["sample_id"],
                "source_dataset": row["source_dataset"],
                "prompt_text": row["prompt_text"],
                "response_text": res.get("response_text", ""),
                "thinking_text": res.get("thinking_text"),
                "full_response": res.get("full_response", ""),
                "input_tokens": res.get("input_tokens", 0),
                "output_tokens": res.get("output_tokens", 0),
                "model_path": model_path,
                "error": None,
            }
            append_checkpoint(checkpoint_path, record)
            logger.debug(f"sample={row['sample_id']} out_tokens={record['output_tokens']} "
                         f"has_think={record['thinking_text'] is not None}")
            pbar.update(1)

    pbar.close()
    logger.info("Inference complete")
    df_results = checkpoint_to_dataframe(checkpoint_path)
    df_results = df_results[df_results["sample_id"].isin(expected_ids)].reset_index(drop=True)
    save_parquet(df_results, output_path)
    return df_results
