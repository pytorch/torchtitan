# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Union

import torch
import torch.nn as nn

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import register_model_converter
from torchtitan.tools.logging import logger

# Cache for dynamically created LoRA classes
_cls_to_lora_cls: dict[type, type] = {}


class LoRAModule:
    """Mixin class that adds LoRA adapter functionality to nn.Linear subclasses.

    This class is injected leftmost in the MRO to override the forward method
    while preserving the original class hierarchy. The original weight and bias
    are accessed directly via self.weight and self.bias (inherited from nn.Linear).
    """

    # LoRA parameters (set by _setup_lora)
    _lora_rank: int
    _lora_alpha: float
    _lora_scaling: float
    _lora_dropout: nn.Module
    lora_a: nn.Linear
    lora_b: nn.Linear

    # Inherited from nn.Linear
    weight: torch.Tensor
    bias: Optional[torch.Tensor]
    in_features: int
    out_features: int

    def _setup_lora(
        self,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        """Initialize LoRA parameters for this module."""
        self._lora_rank = rank
        self._lora_alpha = alpha
        self._lora_scaling = alpha / rank
        self._lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Get dimensions from self (inherited from nn.Linear)
        in_dim = self.in_features  # type: ignore[attr-defined]
        out_dim = self.out_features  # type: ignore[attr-defined]
        dtype = self.weight.dtype  # type: ignore[attr-defined]

        # Create LoRA layers on meta device
        self.lora_a = nn.Linear(in_dim, rank, bias=False, device="meta", dtype=dtype)
        self.lora_b = nn.Linear(rank, out_dim, bias=False, device="meta", dtype=dtype)

    def _initialize_lora_parameters(self) -> None:
        """Initialize LoRA parameters after to_empty."""
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.lora_a.weight.requires_grad = True
        self.lora_b.weight.requires_grad = True

    def _lora_to_empty(
        self, *, device: Optional[Union[str, torch.device, int]], recurse: bool = True
    ) -> None:
        """Move LoRA layers to device."""
        self.lora_a.to_empty(device=device, recurse=recurse)
        self.lora_b.to_empty(device=device, recurse=recurse)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base linear forward using weight and bias directly
        out = nn.functional.linear(x, self.weight, self.bias)

        # LoRA path: out += scale * (x @ A.T @ B.T)
        lora_x = self._lora_dropout(x)
        lora_out = self.lora_b(self.lora_a(lora_x))
        out = out + self._lora_scaling * lora_out

        return out


class LoRAConverter:
    """Model converter that adds LoRA adapters to Linear layers using MRO injection."""

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.rank = job_config.lora.rank
        self.alpha = job_config.lora.alpha
        self.dropout = job_config.lora.dropout

    def convert(self, model: nn.Module) -> None:
        """Inplace conversion of the model to use LoRA adapters via MRO injection."""
        init_weights_fn = model.init_weights
        lora_modules: list[LoRAModule] = []

        for module in list(model.modules()):
            for param in module._parameters.values():
                if param is not None:
                    param.requires_grad_(False)
            for name, child in list(module._modules.items()):
                if isinstance(child, nn.Linear) and name not in ("lora_a", "lora_b"):
                    wrapper = self._create_lora_wrapper(child)
                    setattr(module, name, wrapper)
                    lora_modules.append(wrapper)

        # Wrap init_weights to initialize LoRA parameters
        def new_init_weights(*args, **kwargs):
            if callable(init_weights_fn):
                init_weights_fn(*args, **kwargs)
            # Initialize LoRA parameters and move to device
            for lora_mod in lora_modules:
                lora_mod._lora_to_empty(device=args[0] if args else "cuda")
                lora_mod._initialize_lora_parameters()

        object.__setattr__(model, "init_weights", new_init_weights)

    def _create_lora_wrapper(self, linear: nn.Linear) -> nn.Module:
        """Create a LoRA-wrapped module by injecting LoRAModule leftmost in MRO."""
        cls = linear.__class__

        # Check cache for existing LoRA class
        new_cls = _cls_to_lora_cls.get(cls)
        if not new_cls:
            # Place LoRA leftmost for highest priority in the method resolution order
            new_cls = type(f"LoRA{cls.__name__}", (LoRAModule, cls), {})
            _cls_to_lora_cls[cls] = new_cls

        # Change the class of the linear module (weight and bias are already present)
        linear.__class__ = new_cls

        # Setup LoRA parameters
        linear._setup_lora(self.rank, self.alpha, self.dropout)  # type: ignore[attr-defined]

        return linear  # type: ignore[return-value]

    def post_optimizer_hook(self, model: Union[nn.Module, list[nn.Module]]) -> None:
        """Post-optimizer hook (no-op for LoRA)."""
        pass


# Register the LoRA converter
register_model_converter(LoRAConverter, "lora")
