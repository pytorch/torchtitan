# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torchtitan.config import Configurable
from torchtitan.models.common.linear import Linear
from torchtitan.tools.logging import logger

# Cache for dynamically created LoRA classes
_lora_class_cache: dict[type, type] = {}


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


class LoRAConverter(Configurable):
    """Apply LoRA adapters to all Linear layers in a model."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        rank: int = 8
        """Rank of the LoRA matrices (lora_a: in_features x rank, lora_b: rank x out_features)."""

        alpha: float = 16.0
        """Scaling factor. Output is scaled by alpha/rank."""

        save_adapter_only: bool = True
        """If True, only save LoRA adapter weights in checkpoints.
        Requires base model to be loaded from HF/initial_load_path on resume.
        Set to False to save full model weights for debugging without pretrained base."""

    def __init__(self, config: Config, **kwargs):
        self.rank = config.rank
        self.alpha = config.alpha
        self.save_adapter_only = config.save_adapter_only
        logger.info(f"LoRA training active with rank={self.rank}, alpha={self.alpha}")

    def convert(self, model: nn.Module) -> None:
        model.requires_grad_(False)
        self._replace_linears_with_lora(model)

    def _replace_linears_with_lora(self, module: nn.Module) -> None:
        for _, child in list(module.named_modules()):
            if isinstance(child, nn.Linear):
                apply_lora(child, self.rank, self.alpha)

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]) -> None:
        pass
