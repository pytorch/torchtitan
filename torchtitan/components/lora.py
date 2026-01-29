# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Union

import torch
import torch.nn as nn

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import register_model_converter

# Cache for dynamically created LoRA classes
_lora_class_cache: dict[type, type] = {}


def create_lora_linear(parent_cls: type) -> type:
    """Create a LoRA-enabled subclass of a Linear layer.

    Args:
        parent_cls: A nn.Linear subclass to extend with LoRA.

    Returns:
        A new class with LoRA adapters that inherits from parent_cls.
    """
    if parent_cls in _lora_class_cache:
        return _lora_class_cache[parent_cls]

    class LoRALinear(parent_cls):  # type: ignore[valid-type, misc]
        def __init__(
            self,
            *args: Any,
            rank: int = 8,
            alpha: float = 16.0,
            dropout: float = 0.0,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self._init_lora(rank, alpha, dropout)

        def _init_lora(self, rank: int, alpha: float, dropout: float) -> None:
            self._lora_scaling = alpha / rank
            self._lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.lora_a = nn.Linear(
                self.in_features,
                rank,
                bias=False,
                device=self.weight.device,
                dtype=self.weight.device,
            )
            self.lora_b = nn.Linear(
                rank,
                self.out_features,
                bias=False,
                device=self.weight.device,
                dtype=self.weight.device,
            )
            nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b.weight)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            base_out = super().forward(x)
            lora_out = self.lora_b(self.lora_a(self._lora_dropout(x)))
            return base_out + self._lora_scaling * lora_out

    LoRALinear.__name__ = f"LoRA{parent_cls.__name__}"
    LoRALinear.__qualname__ = f"LoRA{parent_cls.__name__}"
    _lora_class_cache[parent_cls] = LoRALinear
    return LoRALinear


class LoRAConverter:
    """Converts a model to use LoRA adapters on all Linear layers."""

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.rank = job_config.lora.rank
        self.alpha = job_config.lora.alpha
        self.dropout = job_config.lora.dropout
        self._lora_modules: list[nn.Module] = []

    def convert(self, model: nn.Module) -> None:
        """Apply LoRA to all Linear layers, freezing base model weights."""
        self._apply_lora(model)

        if self._is_meta_device(model):
            self._hook_init_weights(model)

    def _apply_lora(self, module: nn.Module) -> None:
        """Recursively freeze params and wrap Linear layers with LoRA."""
        for name, child in list(module._modules.items()):
            if name in ("lora_a", "lora_b") or child is None:
                continue
            # Freeze direct parameters of this module
            for param in child.parameters(recurse=False):
                param.requires_grad_(False)
            if isinstance(child, nn.Linear):
                setattr(module, name, self._wrap_linear(child))
                self._lora_modules.append(getattr(module, name))
            else:
                self._apply_lora(child)

    def _wrap_linear(self, linear: nn.Linear) -> nn.Module:
        """Wrap a Linear layer with LoRA adapters."""
        lora_cls = create_lora_linear(linear.__class__)
        linear.__class__ = lora_cls

        linear._lora_scaling = self.alpha / self.rank  # type: ignore[attr-defined]
        linear._lora_dropout = (  # type: ignore[attr-defined]
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()
        )
        linear.lora_a = nn.Linear(  # type: ignore[attr-defined]
            linear.in_features,
            self.rank,
            bias=False,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        linear.lora_b = nn.Linear(  # type: ignore[attr-defined]
            self.rank,
            linear.out_features,
            bias=False,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )

        if linear.weight.device.type != "meta":
            self._init_lora_weights(linear)

        return linear

    def _init_lora_weights(self, module: nn.Module) -> None:
        """Initialize LoRA weights and enable gradients."""
        nn.init.kaiming_uniform_(module.lora_a.weight, a=math.sqrt(5))  # type: ignore[union-attr]
        nn.init.zeros_(module.lora_b.weight)  # type: ignore[union-attr]
        module.lora_a.weight.requires_grad_(True)  # type: ignore[union-attr]
        module.lora_b.weight.requires_grad_(True)  # type: ignore[union-attr]

    def _is_meta_device(self, model: nn.Module) -> bool:
        return any(p.device.type == "meta" for p in model.parameters())

    def _hook_init_weights(self, model: nn.Module) -> None:
        """Hook into init_weights to initialize LoRA after materialization."""
        original_init_weights = getattr(model, "init_weights", None)

        def new_init_weights(*args: Any, **kwargs: Any) -> None:
            if callable(original_init_weights):
                original_init_weights(*args, **kwargs)
            device = args[0] if args else kwargs.get("buffer_device", "cuda")
            for module in self._lora_modules:
                module.lora_a.to_empty(device=device, recurse=True)  # type: ignore[union-attr]
                module.lora_b.to_empty(device=device, recurse=True)  # type: ignore[union-attr]
                self._init_lora_weights(module)

        object.__setattr__(model, "init_weights", new_init_weights)

    def post_optimizer_hook(self, model: Union[nn.Module, list[nn.Module]]) -> None:
        pass


register_model_converter(LoRAConverter, "lora")
