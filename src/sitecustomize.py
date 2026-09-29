"""Process-start hooks used by Step4 evaluation.

This module is imported automatically by Python when ``src/`` is on
``PYTHONPATH``.  Keep it opt-in: most project commands should not mutate vLLM's
global registry just because this repository is importable.
"""

from __future__ import annotations

import os
import sys


def _env_flag(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _register_qwen35_text_vllm_model() -> None:
    try:
        from vllm.model_executor.models.registry import ModelRegistry

        ModelRegistry.register_model(
            "Qwen3_5ForCausalLM",
            "step4_vllm_qwen35_text:Step4Qwen3_5TextForCausalLM",
        )
        ModelRegistry.register_model(
            "Qwen3_5MoeForCausalLM",
            "step4_vllm_qwen35_text:Step4Qwen3_5MoeTextForCausalLM",
        )
    except Exception as exc:  # pragma: no cover - best-effort startup hook.
        if _env_flag("STEP4_QWEN35_VLLM_PATCH_DEBUG"):
            print(
                f"[step4 sitecustomize] failed to register Qwen3.5 vLLM shim: {exc}",
                file=sys.stderr,
            )


if _env_flag("STEP4_ENABLE_QWEN35_VLLM_PATCH"):
    _register_qwen35_text_vllm_model()
