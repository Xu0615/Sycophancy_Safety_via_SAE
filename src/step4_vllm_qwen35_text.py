"""vLLM registration shim for Step4 Qwen3.5 text-only full-SFT exports."""

from collections.abc import Iterable

import torch

from vllm.model_executor.models.interfaces import IsHybrid
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForCausalLM,
)


def _prefix_text_model_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Map Transformers text-only keys onto vLLM's nested text model.

    ``Qwen3_5TextForCausalLM.save_pretrained`` writes keys such as
    ``embed_tokens.weight`` and ``layers.0...``. Tied-embedding exports from
    the outer Qwen3.5 wrapper may instead use ``model.language_model.*``.
    vLLM's text CausalLM nests the text modules directly below ``model``.
    """

    for name, tensor in weights:
        for wrapper_prefix in (
            "model.language_model.model.",
            "model.language_model.",
            "language_model.model.",
            "language_model.",
        ):
            if name.startswith(wrapper_prefix):
                name = name[len(wrapper_prefix) :]
                break
        if not name.startswith(("model.", "lm_head.", "mtp.")):
            name = f"model.{name}"
        yield name, tensor


class Step4Qwen3_5TextForCausalLM(Qwen3_5ForCausalLM, IsHybrid):
    """Expose Qwen3.5 text CausalLM as a hybrid model to vLLM."""

    is_hybrid = True

    get_mamba_state_dtype_from_config = (
        Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config
    )
    get_mamba_state_shape_from_config = (
        Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config
    )
    get_mamba_state_copy_func = Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        return super().load_weights(_prefix_text_model_weights(weights))


class Step4Qwen3_5MoeTextForCausalLM(Qwen3_5MoeForCausalLM, IsHybrid):
    """Expose Qwen3.5 MoE text CausalLM as a hybrid model to vLLM."""

    is_hybrid = True

    get_mamba_state_dtype_from_config = (
        Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config
    )
    get_mamba_state_shape_from_config = (
        Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config
    )
    get_mamba_state_copy_func = Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        return super().load_weights(_prefix_text_model_weights(weights))
