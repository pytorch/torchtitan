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
    assert issubclass(
        parent_cls, nn.Linear
    ), f"parent_cls must be a subclass of nn.Linear, got {parent_cls}"

    if parent_cls in _lora_class_cache:
        return _lora_class_cache[parent_cls]

    class LoRALinear(parent_cls):  # type: ignore[valid-type, misc]
        def __init__(
            self,
            *args: Any,
            rank: int = 8,
            alpha: float = 16.0,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self._init_lora(rank, alpha)

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
            self.lora_a = nn.Linear(
                self.in_features,
                rank,
                bias=False,
                device=device,
                dtype=dtype,
            )
            self.lora_b = nn.Linear(
                rank,
                self.out_features,
                bias=False,
                device=device,
                dtype=dtype,
            )
            self._init_weight()

        def _init_weight(self) -> None:
            _super_init_weight = getattr(super(), "_init_weight", None)
            if callable(_super_init_weight):
                _super_init_weight()
            nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b.weight)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            base_out = super().forward(input)
            # Compute LoRA in weight dtype, cast output to match base_out
            lora_out = self.lora_b(self.lora_a(input))
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
        self._lora_modules: list[nn.Module] = []

    def convert(self, model: nn.Module) -> None:
        """Apply LoRA to all Linear layers, freezing base model weights."""
        self._replace_linears_with_lora(model)
        self._override_model_init_weights(model)

    def _replace_linears_with_lora(self, module: nn.Module) -> None:
        """Recursively freeze params and replace Linear layers with LoRA equivalents."""
        for name, child in list(module._modules.items()):
            if name in ("lora_a", "lora_b") or child is None:
                continue
            # Freeze direct parameters of this module
            for param in child.parameters(recurse=False):
                param.requires_grad_(False)
            if isinstance(child, nn.Linear):
                lora_cls = create_lora_linear(type(child))
                # Build kwargs, handling special cases like Float8Linear
                kwargs: dict[str, Any] = {
                    "bias": child.bias is not None,
                    "device": child.weight.device,
                    "dtype": child.weight.dtype,
                    "rank": self.rank,
                    "alpha": self.alpha,
                }
                # Pass through config for Float8Linear and similar classes
                if hasattr(child, "config"):
                    kwargs["config"] = child.config
                lora_layer = lora_cls(
                    child.in_features,
                    child.out_features,
                    **kwargs,
                )
                lora_layer.weight = child.weight
                if child.bias is not None:
                    lora_layer.bias = child.bias
                setattr(module, name, lora_layer)
                self._lora_modules.append(lora_layer)
            else:
                self._replace_linears_with_lora(child)

    def _override_model_init_weights(self, model: nn.Module) -> None:
        """Override model's init_weights to also initialize LoRA adapters."""
        original_init_weights = getattr(model, "init_weights", None)

        def new_init_weights(*args: Any, **kwargs: Any) -> None:
            if callable(original_init_weights):
                original_init_weights(*args, **kwargs)
            for module in self._lora_modules:
                module.weight.requires_grad_(False)
                if module.bias is not None:
                    module.bias.requires_grad_(False)
                # Reinitialize LoRA weights
                _init_weight = getattr(module, "_init_weight", None)
                if callable(_init_weight):
                    _init_weight()

        object.__setattr__(model, "init_weights", new_init_weights)

    def post_optimizer_hook(self, model: Union[nn.Module, list[nn.Module]]) -> None:
        pass


register_model_converter(LoRAConverter, "lora")
