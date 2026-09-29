import pytest
import torch


vllm = pytest.importorskip("vllm")

from src.step4_vllm_qwen35_text import _prefix_text_model_weights


def test_qwen35_vllm_weight_prefix_accepts_text_and_outer_exports():
    tensor = torch.zeros(1)
    weights = [
        ("layers.0.input_layernorm.weight", tensor),
        ("model.layers.1.input_layernorm.weight", tensor),
        ("model.language_model.layers.2.input_layernorm.weight", tensor),
        ("language_model.model.layers.3.input_layernorm.weight", tensor),
        ("model.language_model.lm_head.weight", tensor),
    ]

    names = [name for name, _ in _prefix_text_model_weights(weights)]

    assert names == [
        "model.layers.0.input_layernorm.weight",
        "model.layers.1.input_layernorm.weight",
        "model.layers.2.input_layernorm.weight",
        "model.layers.3.input_layernorm.weight",
        "lm_head.weight",
    ]
