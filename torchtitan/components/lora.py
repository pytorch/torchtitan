# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Callable, cast, Union

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
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("LoRALinear should not be instantiated directly. ")

        def _init_lora(
            self,
            rank: int,
            alpha: float,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ) -> None:
            base = cast(nn.Linear, self)
            self._lora_scaling = alpha / rank
            device = device if device is not None else base.weight.device
            dtype = dtype if dtype is not None else base.weight.dtype
            self.lora_a = nn.Linear(
                base.in_features,
                rank,
                bias=False,
                device=device,
                dtype=dtype,
            )
            self.lora_b = nn.Linear(
                rank,
                base.out_features,
                bias=False,
                device=device,
                dtype=dtype,
            )
            self._init_weight()

        def _init_weight(self) -> None:
            nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b.weight)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            base_out = super().forward(input)
            lora_out = self.lora_b(self.lora_a(input))
            return base_out + self._lora_scaling * lora_out

    LoRALinear.__name__ = f"LoRA{parent_cls.__name__}"
    LoRALinear.__qualname__ = f"LoRA{parent_cls.__name__}"
    _lora_class_cache[parent_cls] = LoRALinear
    return LoRALinear


def apply_lora(linear: nn.Linear, rank: int, alpha: float) -> nn.Linear:
    """Convert an existing Linear layer to a LoRALinear in-place.

    Args:
        linear: The Linear layer to convert.
        rank: LoRA rank.
        alpha: LoRA alpha scaling factor.

    Returns:
        The same linear object, now with LoRA adapters attached.
    """
    lora_cls = create_lora_linear(type(linear))
    # Change class in-place to avoid re-initializing the Linear
    linear.__class__ = lora_cls
    # Initialize LoRA adapters as post-init
    _init_lora = getattr(linear, "_init_lora", None)
    assert _init_lora is not None and callable(_init_lora)
    _init_lora(rank, alpha)
    return linear


def _chain_fns(prev: Callable[[], None], curr: Callable[[], None]) -> Callable[[], None]:
    """Chain two functions together."""
    def combined() -> None:
        prev()
        curr()

    return combined


class LoRAConverter:
    """Converts a model to use LoRA adapters on all Linear layers."""

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.rank = job_config.lora.rank
        self.alpha = job_config.lora.alpha

    def convert(self, model: nn.Module) -> None:
        """Apply LoRA to all Linear layers, freezing base model weights."""
        self._replace_linears_with_lora(model)

    def _replace_linears_with_lora(self, module: nn.Module) -> None:
        """Recursively freeze params and replace Linear layers with LoRA equivalents."""
        chained_init: Callable[[], None] = lambda: None

        # Freeze params and apply LoRA to all Linear layers
        for fqn, child in module.named_modules():
            if fqn.endswith("lora_a") or fqn.endswith("lora_b") or child is None:
                continue
            for param in child.parameters(recurse=False):
                param.requires_grad_(False)
            if isinstance(child, nn.Linear):
                apply_lora(child, self.rank, self.alpha)
                _init_weight = getattr(child, "_init_weight", None)
                if callable(_init_weight):
                    prev = chained_init
                    curr = cast(Callable[[], None], _init_weight)

                    def _chain(p: Callable[[], None] = prev, c: Callable[[], None] = curr) -> None:
                        p()
                        c()

                    chained_init = _chain

        # Override the main model's init_weights to also init LoRA weights
        original_init_weights = getattr(module, "init_weights", None)

        def new_model_init_weights(*args: Any, **kwargs: Any) -> None:
            if original_init_weights is not None and callable(original_init_weights):
                original_init_weights(*args, **kwargs)
            chained_init()

        object.__setattr__(module, "init_weights", new_model_init_weights)

    def post_optimizer_hook(self, model: Union[nn.Module, list[nn.Module]]) -> None:
        pass


register_model_converter(LoRAConverter, "lora")
