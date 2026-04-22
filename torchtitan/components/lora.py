# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from torchtitan.config import Configurable
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger

# Module-protocol-compatible Linear for LoRA adapter layers.
# Uses Module.from_nn_module so that adapters satisfy verify_module_protocol
# and use reset_parameters() for initialization (no param_init Config needed).
_LoRALinear = Module.from_nn_module(nn.Linear)

# Cache for dynamically created LoRA classes
_lora_class_cache: dict[type, type] = {}


def apply_lora(linear: nn.Linear, rank: int, alpha: float) -> nn.Linear:
    """Apply LoRA adapters to a Linear module via dynamic class inheritance.

    Creates (and caches) a LoRA subclass of ``type(linear)`` so that LoRA
    composes with any Linear variant (e.g. FakeQuantizedLinear).
    """
    parent_cls = type(linear)
    if not issubclass(parent_cls, nn.Linear):
        raise ValueError(f"apply_lora expects an nn.Linear subclass, got {parent_cls}")

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

            def _init_lora(self, rank: int, alpha: float) -> None:
                self._lora_scaling = alpha / rank
                device = self.weight.device
                dtype = self.weight.dtype
                self.lora_a = _LoRALinear(
                    self.in_features,
                    rank,
                    bias=False,
                    device=device,
                    dtype=dtype,
                )
                self.lora_b = _LoRALinear(
                    rank,
                    self.out_features,
                    bias=False,
                    device=device,
                    dtype=dtype,
                )
                self._init_weight()

            def _init_weight(self) -> None:
                # pyrefly: ignore [bad-argument-type]
                nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
                nn.init.zeros_(
                    self.lora_b.weight  # pyrefly: ignore [bad-argument-type]
                )

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

        filter_fqns: list[str] = field(default_factory=list)
        """FQNs of Linear modules to skip (substring match).
        Example: filter_fqns=["output"] to exclude the LM head."""

    def __init__(self, config: Config, **kwargs: Any) -> None:
        if config.rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {config.rank}")
        self.rank = config.rank
        self.alpha = config.alpha
        self.filter_fqns = config.filter_fqns
        logger.info(f"LoRA training active with rank={self.rank}, alpha={self.alpha}")

    def convert(self, model: nn.Module) -> None:
        model.requires_grad_(False)
        self._replace_linears_with_lora(model)

    def _replace_linears_with_lora(self, module: nn.Module) -> None:
        # Snapshot the module list before mutation so that newly created
        # lora_a / lora_b children (which are nn.Linear) are not visited.
        for fqn, child in list(module.named_modules()):
            if isinstance(child, nn.Linear) and not any(
                f in fqn for f in self.filter_fqns
            ):
                apply_lora(child, self.rank, self.alpha)

        # Patch init_weights to also reinitialize LoRA adapters
        original_init_weights = getattr(module, "init_weights", None)

        def new_model_init_weights(*args: Any, **kwargs: Any) -> None:
            if original_init_weights is not None and callable(original_init_weights):
                original_init_weights(*args, **kwargs)
            for sub_module in module.modules():
                if hasattr(sub_module, "_lora_scaling"):
                    sub_module._init_weight()  # pyrefly: ignore [not-callable]

        object.__setattr__(module, "init_weights", new_model_init_weights)

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]) -> None:
        pass
