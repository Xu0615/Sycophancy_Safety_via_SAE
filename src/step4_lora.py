"""Lightweight LoRA utilities for Step 4 sycophancy SFT.

The project environment does not require PEFT.  This module implements the
small subset of LoRA behavior needed by the Step 4 SFT loop:

- wrap selected ``nn.Linear`` modules in the language model blocks;
- save and load adapter-only checkpoints;
- optionally merge adapter weights back into the base model for no-hook eval.

The paper uses rs-LoRA, so the default scaling mode here supports both the
standard LoRA scaling and the rank-stabilized variant.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    """Serializable LoRA configuration."""

    rank: int = 32
    alpha: float = 64.0
    dropout: float = 0.05
    scaling_type: str = "standard"
    target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
    )
    module_filter: str = "language_model.layers"
    lora_dtype: str = "float32"

    @classmethod
    def from_dict(cls, data: Dict) -> "LoRAConfig":
        data = dict(data or {})
        if "target_modules" in data:
            data["target_modules"] = tuple(data["target_modules"])
        return cls(**data)

    def to_dict(self) -> Dict:
        out = asdict(self)
        out["target_modules"] = list(self.target_modules)
        return out


def dtype_from_name(name: str) -> torch.dtype:
    """Convert a config dtype string to a torch dtype."""

    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported LoRA dtype: {name}")


class LoRALinear(nn.Module):
    """A frozen linear layer plus trainable low-rank update."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
        scaling_type: str = "standard",
        lora_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")

        base_device = base.weight.device
        if base_device.type == "meta":
            raise ValueError("Cannot attach LoRA to a module still on the meta device")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling_type = str(scaling_type)
        if self.scaling_type == "standard":
            self.scaling = float(alpha) / float(rank)
        elif self.scaling_type == "rs":
            self.scaling = float(alpha) / math.sqrt(float(rank))
        else:
            raise ValueError(f"Unsupported LoRA scaling type: {self.scaling_type}")
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        in_features = base.in_features
        out_features = base.out_features

        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=base_device, dtype=lora_dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=base_device, dtype=lora_dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        for param in self.base.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        x_lora = self.dropout(x).to(dtype=self.lora_A.dtype)
        update = torch.nn.functional.linear(x_lora, self.lora_A)
        update = torch.nn.functional.linear(update, self.lora_B)
        return base_out + update.to(dtype=base_out.dtype) * self.scaling

    def merged_weight(self) -> torch.Tensor:
        """Return the base weight plus LoRA delta on the base weight device."""

        delta = torch.matmul(self.lora_B, self.lora_A) * self.scaling
        return self.base.weight.data + delta.to(device=self.base.weight.device, dtype=self.base.weight.dtype)


def _matches_target(name: str, target_modules: Iterable[str], module_filter: Optional[str]) -> bool:
    if module_filter and module_filter not in name:
        return False
    return any(name.endswith(target) for target in target_modules)


def _get_parent_module(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora_adapters(model: nn.Module, config: LoRAConfig) -> List[str]:
    """Replace target linear modules with ``LoRALinear`` wrappers.

    Returns the names of modules that were wrapped.
    """

    lora_dtype = dtype_from_name(config.lora_dtype)
    replacements: List[Tuple[str, nn.Linear]] = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and _matches_target(
            name,
            config.target_modules,
            config.module_filter,
        ):
            replacements.append((name, module))

    wrapped_names: List[str] = []
    for name, linear in replacements:
        parent, child_name = _get_parent_module(model, name)
        setattr(
            parent,
            child_name,
            LoRALinear(
                base=linear,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
                scaling_type=config.scaling_type,
                lora_dtype=lora_dtype,
            ),
        )
        wrapped_names.append(name)
    return wrapped_names


def iter_lora_modules(model: nn.Module) -> Iterator[Tuple[str, LoRALinear]]:
    """Yield all LoRA-wrapped modules."""

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return adapter-only state dict."""

    state: Dict[str, torch.Tensor] = {}
    for name, module in iter_lora_modules(model):
        state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
        state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    return state


def load_lora_state_dict(model: nn.Module, state: Dict[str, torch.Tensor], strict: bool = True) -> None:
    """Load adapter-only state dict into an already wrapped model."""

    missing: List[str] = []
    unexpected = set(state.keys())
    for name, module in iter_lora_modules(model):
        for suffix, param in (("lora_A", module.lora_A), ("lora_B", module.lora_B)):
            key = f"{name}.{suffix}"
            if key not in state:
                missing.append(key)
                continue
            param.data.copy_(state[key].to(device=param.device, dtype=param.dtype))
            unexpected.discard(key)

    if strict and (missing or unexpected):
        raise RuntimeError(
            "LoRA checkpoint mismatch: "
            f"missing={missing[:5]} unexpected={sorted(unexpected)[:5]}"
        )


def save_lora_adapter(
    model: nn.Module,
    output_dir: str,
    config: LoRAConfig,
    extra_metadata: Optional[Dict] = None,
) -> None:
    """Save adapter weights and config."""

    os.makedirs(output_dir, exist_ok=True)
    state = lora_state_dict(model)
    adapter_path = os.path.join(output_dir, "adapter_model.pt")
    torch.save(state, adapter_path)

    config_payload = config.to_dict()
    if extra_metadata:
        config_payload["metadata"] = extra_metadata
    with open(os.path.join(output_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, ensure_ascii=False)


def load_lora_adapter(model: nn.Module, adapter_dir: str, strict: bool = True) -> LoRAConfig:
    """Inject wrappers from adapter config, then load adapter weights."""

    with open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8") as f:
        payload = json.load(f)
    config_payload = {k: v for k, v in payload.items() if k != "metadata"}
    config = LoRAConfig.from_dict(config_payload)
    inject_lora_adapters(model, config)
    state = torch.load(os.path.join(adapter_dir, "adapter_model.pt"), map_location="cpu", weights_only=False)
    load_lora_state_dict(model, state, strict=strict)
    return config


def merge_lora_adapters(model: nn.Module) -> int:
    """Merge all LoRA weights into base linear layers in-place.

    Returns the number of merged modules.
    """

    merged = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        parent, child_name = _get_parent_module(model, name)
        module.base.weight.data.copy_(module.merged_weight())
        setattr(parent, child_name, module.base)
        merged += 1
    return merged


def trainable_parameter_summary(model: nn.Module) -> Dict[str, float]:
    """Return trainable and total parameter counts."""

    def logical_numel(param: torch.nn.Parameter) -> int:
        if hasattr(param, "ds_numel"):
            return int(param.ds_numel)
        if hasattr(param, "ds_shape"):
            n = 1
            for dim in tuple(param.ds_shape):
                n *= int(dim)
            return n
        return int(param.numel())

    trainable = sum(logical_numel(p) for p in model.parameters() if p.requires_grad)
    total = sum(logical_numel(p) for p in model.parameters())
    ratio = trainable / total if total else 0.0
    return {
        "trainable": float(trainable),
        "total": float(total),
        "trainable_ratio": ratio,
    }
