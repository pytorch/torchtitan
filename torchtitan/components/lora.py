# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from torchtitan.config import Configurable
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe.moe import GroupedExperts, TokenChoiceTopKRouter
from torchtitan.tools.logging import logger

# Cache for dynamically created LoRA classes
_lora_class_cache: dict[type, type] = {}

# Cache for dynamically created expert LoRA classes
_expert_lora_class_cache: dict[type, type] = {}


def apply_lora(linear: nn.Linear, rank: int, alpha: float) -> nn.Linear:
    parent_cls = type(linear)
    assert issubclass(
        parent_cls, nn.Linear
    ), f"parent_cls must be a subclass of nn.Linear, got {parent_cls}"

    if parent_cls not in _lora_class_cache:

        class LoRALinear(parent_cls):  # type: ignore[valid-type, misc]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("LoRALinear should not be instantiated directly.")

            @classmethod
            def from_linear(
                cls, linear: nn.Linear, rank: int, alpha: float
            ) -> "LoRALinear":
                linear.__class__ = cls
                linear._init_lora(rank, alpha)  # type: ignore[attr-defined]
                return linear  # type: ignore[return-value]

            def _init_lora(
                self,
                rank: int,
                alpha: float,
                device: torch.device | None = None,
                dtype: torch.dtype | None = None,
            ) -> None:
                self._lora_scaling = alpha / rank
                device = device if device is not None else self.weight.device
                dtype = dtype if dtype is not None else self.weight.dtype
                self.lora_a = (
                    Linear.Config(bias=False)
                    .build(in_features=self.in_features, out_features=rank)
                    .to(device=device, dtype=dtype)
                )
                self.lora_b = (
                    Linear.Config(bias=False)
                    .build(in_features=rank, out_features=self.out_features)
                    .to(device=device, dtype=dtype)
                )

            def init_weights(self, **kwargs) -> None:
                super().init_weights(**kwargs)  # pyrefly: ignore [not-callable]
                nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_b.weight)

            def forward(self, input: torch.Tensor) -> torch.Tensor:
                base_out = super().forward(input)
                lora_out = self.lora_b(self.lora_a(input))
                return base_out + self._lora_scaling * lora_out

        LoRALinear.__name__ = f"LoRA{parent_cls.__name__}"
        LoRALinear.__qualname__ = f"LoRA{parent_cls.__name__}"
        _lora_class_cache[parent_cls] = LoRALinear

    # pyrefly: ignore [missing-attribute]
    return _lora_class_cache[parent_cls].from_linear(linear, rank, alpha)


def _compute_expert_lora_delta(
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
    target_weight: nn.Parameter,
) -> torch.Tensor:
    """Compute the LoRA weight delta for expert weights.

    Args:
        lora_a: (E, in, r) — projects input dim to rank.
        lora_b: (E, r, out) — projects rank to output dim.
        scaling: alpha / rank.
        target_weight: The base weight parameter to match DTensor placements.

    Returns:
        delta matching target_weight's shape and placements.
        Math: delta = scaling * B^T @ A^T  →  shape (E, out, in).
    """
    from torch.distributed.tensor import distribute_tensor, DTensor

    delta = scaling * torch.bmm(lora_b.transpose(-2, -1), lora_a.transpose(-2, -1))
    # When the base weight is a DTensor (TP/EP sharded), distribute the delta
    # to match its placements so the in-place add_/sub_ operates on matching shapes.
    if isinstance(target_weight, DTensor) and not isinstance(delta, DTensor):
        delta = distribute_tensor(
            delta, target_weight.device_mesh, target_weight.placements
        )
    return delta


def apply_expert_lora(
    experts: GroupedExperts, rank: int, alpha: float
) -> GroupedExperts:
    """Apply LoRA adapters to a GroupedExperts module via class swapping.

    LoRA parameters are registered as direct parameters on the module. EP partition
    functions that use ``named_parameters(recurse=False)`` with ``Shard(0)`` will
    correctly shard them on the expert dimension. TP/ETP partition functions only
    touch w1/w2/w3 by name and leave LoRA parameters unsharded.

    Forward uses merge-per-forward: LoRA deltas are merged into base weights before
    calling the base forward, then unmerged after. This reuses the base
    GroupedExperts.forward without duplicating its DTensor/EP/padding logic.
    """
    parent_cls = type(experts)
    assert issubclass(
        parent_cls, GroupedExperts
    ), f"parent_cls must be a subclass of GroupedExperts, got {parent_cls}"

    if parent_cls not in _expert_lora_class_cache:

        class LoRAGroupedExperts(parent_cls):  # type: ignore[valid-type, misc]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError(
                    "LoRAGroupedExperts should not be instantiated directly."
                )

            @classmethod
            def from_experts(
                cls, experts: GroupedExperts, rank: int, alpha: float
            ) -> "LoRAGroupedExperts":
                experts.__class__ = cls
                experts._init_expert_lora(rank, alpha)  # type: ignore[attr-defined]
                return experts  # type: ignore[return-value]

            def _init_expert_lora(self, rank: int, alpha: float) -> None:
                self._lora_scaling = alpha / rank
                num_experts = self.num_experts
                # w1: (E, hidden_dim, dim) -> A1: (E, dim, r), B1: (E, r, hidden_dim)
                dim_w1_in = self.w1.shape[2]  # dim
                dim_w1_out = self.w1.shape[1]  # hidden_dim
                # w2: (E, dim, hidden_dim) -> A2: (E, hidden_dim, r), B2: (E, r, dim)
                dim_w2_in = self.w2.shape[2]  # hidden_dim
                dim_w2_out = self.w2.shape[1]  # dim
                # w3: (E, hidden_dim, dim) -> A3: (E, dim, r), B3: (E, r, hidden_dim)
                dim_w3_in = self.w3.shape[2]  # dim
                dim_w3_out = self.w3.shape[1]  # hidden_dim

                device = self.w1.device
                dtype = self.w1.dtype

                self.lora_a_w1 = nn.Parameter(
                    torch.empty(
                        num_experts, dim_w1_in, rank, device=device, dtype=dtype
                    )
                )
                self.lora_b_w1 = nn.Parameter(
                    torch.empty(
                        num_experts, rank, dim_w1_out, device=device, dtype=dtype
                    )
                )
                self.lora_a_w2 = nn.Parameter(
                    torch.empty(
                        num_experts, dim_w2_in, rank, device=device, dtype=dtype
                    )
                )
                self.lora_b_w2 = nn.Parameter(
                    torch.empty(
                        num_experts, rank, dim_w2_out, device=device, dtype=dtype
                    )
                )
                self.lora_a_w3 = nn.Parameter(
                    torch.empty(
                        num_experts, dim_w3_in, rank, device=device, dtype=dtype
                    )
                )
                self.lora_b_w3 = nn.Parameter(
                    torch.empty(
                        num_experts, rank, dim_w3_out, device=device, dtype=dtype
                    )
                )

            def init_weights(self, init_std: float) -> None:
                super().init_weights(init_std)
                for name in ("lora_a_w1", "lora_a_w2", "lora_a_w3"):
                    nn.init.kaiming_uniform_(getattr(self, name), a=math.sqrt(5))
                for name in ("lora_b_w1", "lora_b_w2", "lora_b_w3"):
                    nn.init.zeros_(getattr(self, name))

            def forward(
                self,
                x: torch.Tensor,
                num_tokens_per_expert: torch.Tensor,
            ) -> torch.Tensor:
                # Merge LoRA deltas into base weights, run base forward, unmerge.
                # This reuses all base GroupedExperts logic (DTensor, EP, padding).
                deltas = {}
                for w_name, a_name, b_name in (
                    ("w1", "lora_a_w1", "lora_b_w1"),
                    ("w2", "lora_a_w2", "lora_b_w2"),
                    ("w3", "lora_a_w3", "lora_b_w3"),
                ):
                    lora_a = getattr(self, a_name)
                    lora_b = getattr(self, b_name)
                    w = getattr(self, w_name)
                    delta = _compute_expert_lora_delta(
                        lora_a, lora_b, self._lora_scaling, w
                    )
                    w.data.add_(delta)
                    deltas[w_name] = delta

                try:
                    return super().forward(x, num_tokens_per_expert)
                finally:
                    # Unmerge: subtract deltas to restore original weights
                    for w_name, delta in deltas.items():
                        getattr(self, w_name).data.sub_(delta)

        LoRAGroupedExperts.__name__ = f"LoRA{parent_cls.__name__}"
        LoRAGroupedExperts.__qualname__ = f"LoRA{parent_cls.__name__}"
        _expert_lora_class_cache[parent_cls] = LoRAGroupedExperts

    # pyrefly: ignore [missing-attribute]
    return _expert_lora_class_cache[parent_cls].from_experts(experts, rank, alpha)


class LoRAConverter(Configurable):
    """Apply LoRA adapters to all Linear layers and GroupedExperts in a model."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        rank: int = 8
        """Rank of the LoRA matrices (lora_a: in_features x rank, lora_b: rank x out_features)."""

        alpha: float = 16.0
        """Scaling factor. Output is scaled by alpha/rank."""

        save_format: str = "dcp"
        """Format for saving adapter weights at the last training step.
        "dcp" saves adapter weights via DCP (default, supports resumption).
        "peft" saves adapter_model.safetensors + adapter_config.json for
        compatibility with HuggingFace PEFT."""

    def __init__(self, config: Config, **kwargs):
        self.rank = config.rank
        self.alpha = config.alpha
        self.save_format = config.save_format
        if self.save_format not in ("dcp", "peft"):
            raise ValueError(
                f"LoRA save_format must be 'dcp' or 'peft', got '{self.save_format}'"
            )
        logger.info(f"LoRA training active with rank={self.rank}, alpha={self.alpha}")

    @staticmethod
    def _is_lora_key(key: str) -> bool:
        """Check if a state dict key belongs to a LoRA adapter."""
        return ".lora_a." in key or ".lora_b." in key

    def convert(self, model: nn.Module) -> None:
        model.requires_grad_(False)
        self._replace_linears_with_lora(model)

        # Wire up checkpoint filtering so ModelWrapper knows which keys
        # are adapter keys.
        model.converter_key_filter = self._is_lora_key  # type: ignore[attr-defined]
        model.lora_save_format = self.save_format  # type: ignore[attr-defined]
        model.lora_config = {  # type: ignore[attr-defined]
            "rank": self.rank,
            "alpha": self.alpha,
        }

    def _replace_linears_with_lora(self, module: nn.Module) -> None:
        # Collect router gate linears so we can skip them — routing scores
        # must stay frozen to preserve expert load balancing.
        router_gate_ids: set[int] = set()
        for child in module.modules():
            if isinstance(child, TokenChoiceTopKRouter):
                router_gate_ids.add(id(child.gate))

        for _, child in list(module.named_modules()):
            if isinstance(child, nn.Linear) and id(child) not in router_gate_ids:
                apply_lora(child, self.rank, self.alpha)
            elif isinstance(child, GroupedExperts):
                apply_expert_lora(child, self.rank, self.alpha)

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]) -> None:
        pass


def remap_lora_keys_to_hf(
    adapter_sd: dict[str, Any], from_hf_map: dict[str, str | None]
) -> dict[str, Any]:
    """Remap torchtitan LoRA keys to HF PEFT naming for saving.

    Converts keys like ``layers.0.attention.wq.lora_a.weight`` to
    ``base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight``.
    """
    to_hf_map = {v: k for k, v in from_hf_map.items() if v is not None}
    remapped = {}
    for key, value in adapter_sd.items():
        lora_suffix = None
        base_key = key
        for suffix in (".lora_a.weight", ".lora_b.weight"):
            if key.endswith(suffix):
                lora_suffix = suffix
                base_key = key[: -len(suffix)] + ".weight"
                break
        if lora_suffix is None:
            remapped[key] = value
            continue

        abstract_base = re.sub(r"(?<=\.)\d+(?=\.)", "{}", base_key, count=1)
        layer_match = re.search(r"(?<=\.)\d+(?=\.)", base_key)
        layer_num = layer_match.group(0) if layer_match else None

        if abstract_base in to_hf_map:
            hf_abstract = to_hf_map[abstract_base]
            hf_base = hf_abstract.format(layer_num) if layer_num else hf_abstract
            hf_base_no_weight = hf_base.rsplit(".weight", 1)[0]
            peft_suffix = lora_suffix.replace(".lora_a.", ".lora_A.").replace(
                ".lora_b.", ".lora_B."
            )
            hf_key = f"base_model.model.{hf_base_no_weight}{peft_suffix}"
        else:
            hf_key = f"base_model.model.{key}"
            hf_key = hf_key.replace(".lora_a.", ".lora_A.").replace(
                ".lora_b.", ".lora_B."
            )
            logger.warning(
                f"No HF mapping for base key '{abstract_base}', using '{hf_key}'"
            )
        remapped[hf_key] = value
    return remapped


def remap_lora_keys_from_hf(
    adapter_sd: dict[str, Any], from_hf_map: dict[str, str | None]
) -> dict[str, Any]:
    """Remap HF PEFT keys to torchtitan naming for loading.

    Converts keys like ``base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight``
    to ``layers.0.attention.wq.lora_a.weight``.
    """
    remapped = {}
    for key, value in adapter_sd.items():
        stripped = key
        if stripped.startswith("base_model.model."):
            stripped = stripped[len("base_model.model.") :]
        stripped = stripped.replace(".lora_A.", ".lora_a.").replace(
            ".lora_B.", ".lora_b."
        )

        lora_suffix = None
        hf_base_key = stripped
        for suffix in (".lora_a.weight", ".lora_b.weight"):
            if stripped.endswith(suffix):
                lora_suffix = suffix
                hf_base_key = stripped[: -len(suffix)] + ".weight"
                break
        if lora_suffix is None:
            remapped[stripped] = value
            continue

        abstract_hf = re.sub(r"(?<=\.)\d+(?=\.)", "{}", hf_base_key, count=1)
        layer_match = re.search(r"(?<=\.)\d+(?=\.)", hf_base_key)
        layer_num = layer_match.group(0) if layer_match else None

        if abstract_hf in from_hf_map and from_hf_map[abstract_hf] is not None:
            tt_abstract = from_hf_map[abstract_hf]
            tt_base = tt_abstract.format(layer_num) if layer_num else tt_abstract
            tt_base_no_weight = tt_base.rsplit(".weight", 1)[0]
            tt_key = f"{tt_base_no_weight}{lora_suffix}"
        else:
            tt_key = stripped
            logger.warning(
                f"No torchtitan mapping for HF key '{abstract_hf}', using '{tt_key}'"
            )
        remapped[tt_key] = value
    return remapped
