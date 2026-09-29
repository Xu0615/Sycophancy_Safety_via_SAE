"""Step 3 — SAE Feature Steering.

Load SAE decoder weights and create steering vectors for target features.
The steering hook modifies the residual stream at the target layer during
every forward pass (including each autoregressive generation step).

Steering direction:
  enhance  → residual += alpha * steering_vector
  suppress → residual -= alpha * steering_vector
"""

import json
import logging
import os
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


def load_sae_decoder(sae_path: str):
    """Load SAE decoder weight and metadata.

    Returns dict with keys: decoder_weight [d_model, n_features], n_features, d_model.
    """
    logger.info(f"Loading SAE decoder from {sae_path}")
    state = torch.load(sae_path, map_location="cpu", weights_only=False)
    if "decoder.weight" in state:
        decoder_weight = state["decoder.weight"]
    elif "W_dec" in state:
        decoder_weight = state["W_dec"]
    else:
        raise KeyError(
            "Unsupported SAE checkpoint schema: expected decoder.weight or W_dec"
        )
    d_model, n_features = decoder_weight.shape
    logger.info(f"SAE decoder: d_model={d_model}, n_features={n_features}")
    return {
        "decoder_weight": decoder_weight,
        "n_features": n_features,
        "d_model": d_model,
    }


def build_steering_vector(
    decoder_weight: torch.Tensor,
    feature_ids: List[int],
    normalize: bool = False,
) -> torch.Tensor:
    """Build a steering vector by summing SAE decoder columns for target features.

    Args:
        decoder_weight: [d_model, n_features]
        feature_ids:    list of feature indices
        normalize:      if True, divide by number of features

    Returns:
        steering_vec: [d_model] float32
    """
    vecs = [decoder_weight[:, fid].float() for fid in feature_ids]
    steering_vec = torch.stack(vecs, dim=0).sum(dim=0)
    if normalize and len(feature_ids) > 1:
        steering_vec = steering_vec / len(feature_ids)
    logger.info(
        f"Steering vector built from {len(feature_ids)} feature(s) "
        f"{feature_ids}, norm={steering_vec.norm().item():.4f}"
    )
    return steering_vec


class SteeringHook:
    """Forward hook that adds a steering vector to the residual stream.

    Accepts a layer module directly (not the full model), compatible with
    both vLLM-extracted models and HuggingFace models.

    Handles vLLM Qwen-style (mlp_out, residual) tuples, standard
    transformers (hidden_states, ...) tuples, and bare tensor outputs.
    """

    def __init__(
        self,
        layer_module: torch.nn.Module,
        steering_vec: torch.Tensor,
        alpha: float,
        direction: str = "enhance",
    ):
        self.alpha = alpha
        self.direction = direction
        sign = 1.0 if direction == "enhance" else -1.0
        self._delta = (sign * alpha * steering_vec).clone()
        try:
            self._handle = layer_module.register_forward_hook(
                self._fn, always_call=True)
        except TypeError:
            self._handle = layer_module.register_forward_hook(self._fn)
        logger.info(
            f"SteeringHook registered "
            f"(direction={direction}, alpha={alpha}, "
            f"delta_norm={self._delta.norm().item():.4f})"
        )

    def _fn(self, module, inp, out):
        # Qwen-style (mlp_out, residual): both tensors with same shape
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            a0, a1 = out[0], out[1]
            if (torch.is_tensor(a0) and torch.is_tensor(a1)
                    and a0.shape == a1.shape):
                delta = self._delta.to(a0.device, dtype=a0.dtype)
                new_a0 = a0 + delta
                if len(out) == 2:
                    return (new_a0, a1)
                return (new_a0, a1) + tuple(out[2:])

        # Fallback: standard (hidden_states, ...) or bare tensor
        hs = out[0] if isinstance(out, tuple) else out
        delta = self._delta.to(hs.device, dtype=hs.dtype)
        hs = hs + delta
        if isinstance(out, tuple):
            return (hs,) + out[1:]
        return hs

    def remove(self):
        self._handle.remove()


def load_feature_ids_from_step2(
    step2_dir: str,
    model_name: str,
    feature_subdir: str = "syco_feature",
    feature_file: str = "top_features.json",
    top_k: Optional[int] = 10,
    feature_ids: Optional[List[int]] = None,
) -> List[int]:
    """Load sycophancy feature IDs from the current Step 2 output.

    The current Step 2 feature pipeline writes paired sycophancy features to:
      outputs/step2/<model_name>/syco_feature/top_features.json

    That file is a ranked list of feature dicts.  This loader also accepts
    summary.json, which stores the same ranking in top_feature_ids.
    """
    if feature_ids is not None:
        return [int(fid) for fid in feature_ids]

    path = os.path.join(step2_dir, model_name, feature_subdir, feature_file)
    logger.info(f"Loading Step 2 sycophancy feature IDs from {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        ids = [int(row["feature_id"]) for row in data]
    elif isinstance(data, dict) and "top_feature_ids" in data:
        ids = [int(fid) for fid in data["top_feature_ids"]]
    elif isinstance(data, dict) and "features" in data:
        ids = [int(row["feature_id"]) for row in data["features"]]
    else:
        raise ValueError(
            f"Unsupported Step 2 feature file format: {path}. "
            "Expected a list of feature rows, or a dict with top_feature_ids."
        )

    if top_k is not None:
        ids = ids[:int(top_k)]

    logger.info(
        f"Loaded {len(ids)} feature(s) from {model_name}/{feature_subdir}: {ids}")
    return ids
