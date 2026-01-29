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

# Cache for dynamically created LoRA classes
_cls_to_lora_cls: dict[type, type] = {}


def create_lora_linear(parent_cls: type) -> type:
    """Create a LoRA-enabled version of a Linear subclass.
    """
    if parent_cls in _cls_to_lora_cls:
        return _cls_to_lora_cls[parent_cls]

    class LoRALinearChild(parent_cls):
        """Dynamically created LoRA-enabled Linear subclass."""

        def __init__(
            self,
            *args,
            rank: int = 8,
            alpha: float = 16.0,
            dropout: float = 0.0,
            **kwargs,
        ) -> None:
            super().__init__(*args, **kwargs)

            # Get dimensions from self (inherited from nn.Linear)
            in_features = self.in_features  # type: ignore[attr-defined]
            out_features = self.out_features  # type: ignore[attr-defined]
            dtype = self.weight.dtype  # type: ignore[attr-defined]
            device = self.weight.device  # type: ignore[attr-defined]

            # LoRA parameters
            self._lora_rank = rank
            self._lora_alpha = alpha
            self._lora_scaling = alpha / rank
            self._lora_dropout = (
                nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
            )

            # Create LoRA layers
            self.lora_a = nn.Linear(
                in_features, rank, bias=False, device=device, dtype=dtype
            )
            self.lora_b = nn.Linear(
                rank, out_features, bias=False, device=device, dtype=dtype
            )

            # Initialize LoRA parameters
            nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b.weight)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Base forward from parent class
            out = super().forward(x)

            # LoRA path: out += scale * (x @ A.T @ B.T)
            lora_x = self._lora_dropout(x)
            lora_out = self.lora_b(self.lora_a(lora_x))
            out = out + self._lora_scaling * lora_out

            return out

    LoRALinearChild.__name__ = f"LoRA{parent_cls.__name__}"
    LoRALinearChild.__qualname__ = f"LoRA{parent_cls.__name__}"

    _cls_to_lora_cls[parent_cls] = LoRALinearChild
    return LoRALinearChild


class LoRAConverter:
    """Model converter that adds LoRA adapters to Linear layers using create_lora_linear."""

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.rank = job_config.lora.rank
        self.alpha = job_config.lora.alpha
        self.dropout = job_config.lora.dropout

    def convert(self, model: nn.Module) -> None:
        """Inplace conversion of the model to use LoRA adapters."""
        # First pass: freeze all existing parameters
        for param in model.parameters():
            param.requires_grad_(False)

        # Track LoRA modules for later initialization
        lora_modules: list[nn.Module] = []

        # Second pass: replace Linear layers with LoRA-wrapped versions
        self._apply_lora_to_module(model, lora_modules)

        # Check if model is on meta device (weights need to be materialized later)
        is_meta = any(
            p.device.type == "meta"
            for p in model.parameters()
            if p is not None
        )

        if is_meta:
            # Hook into init_weights to initialize LoRA parameters after materialization
            original_init_weights = getattr(model, "init_weights", None)

            def new_init_weights(*args, **kwargs):
                # Call original init_weights first
                if callable(original_init_weights):
                    original_init_weights(*args, **kwargs)

                # Get the device from args (first arg is usually buffer_device)
                device = args[0] if args else kwargs.get("buffer_device", "cuda")

                # Initialize LoRA parameters
                for lora_mod in lora_modules:
                    # Move LoRA layers from meta to actual device
                    lora_mod.lora_a.to_empty(device=device, recurse=True)  # type: ignore[union-attr]
                    lora_mod.lora_b.to_empty(device=device, recurse=True)  # type: ignore[union-attr]

                    # Initialize weights
                    nn.init.kaiming_uniform_(lora_mod.lora_a.weight, a=math.sqrt(5))  # type: ignore[union-attr]
                    nn.init.zeros_(lora_mod.lora_b.weight)  # type: ignore[union-attr]

                    # Ensure requires_grad is True
                    lora_mod.lora_a.weight.requires_grad_(True)  # type: ignore[union-attr]
                    lora_mod.lora_b.weight.requires_grad_(True)  # type: ignore[union-attr]

            object.__setattr__(model, "init_weights", new_init_weights)

    def _apply_lora_to_module(
        self, module: nn.Module, lora_modules: list[nn.Module]
    ) -> None:
        """Recursively apply LoRA to Linear layers in the module."""
        for name, child in list(module._modules.items()):
            # Skip lora_a and lora_b layers, or None children
            if name in ("lora_a", "lora_b") or child is None:
                continue

            if isinstance(child, nn.Linear):
                # Wrap this Linear layer with LoRA
                wrapper = self._create_lora_wrapper(child)
                setattr(module, name, wrapper)
                lora_modules.append(wrapper)
                # Don't recurse into the wrapper - lora_a/lora_b should not be wrapped
            else:
                # Recurse into non-Linear modules
                self._apply_lora_to_module(child, lora_modules)

    def _create_lora_wrapper(self, linear: nn.Linear) -> nn.Module:
        """Create a LoRA-wrapped module using create_lora_linear."""
        cls = linear.__class__

        # Get or create LoRA class for this Linear subclass
        lora_cls = create_lora_linear(cls)

        # Change the class of the linear module (weight and bias are already present)
        linear.__class__ = lora_cls

        # Get dimensions from the existing linear layer
        in_features = linear.in_features
        out_features = linear.out_features
        dtype = linear.weight.dtype
        device = linear.weight.device

        # Setup LoRA parameters
        linear._lora_rank = self.rank  # type: ignore[attr-defined]
        linear._lora_alpha = self.alpha  # type: ignore[attr-defined]
        linear._lora_scaling = self.alpha / self.rank  # type: ignore[attr-defined]
        linear._lora_dropout = (  # type: ignore[attr-defined]
            nn.Dropout(p=self.dropout) if self.dropout > 0.0 else nn.Identity()
        )

        # Create LoRA layers (on same device as linear, which may be meta)
        linear.lora_a = nn.Linear(  # type: ignore[attr-defined]
            in_features, self.rank, bias=False, device=device, dtype=dtype
        )
        linear.lora_b = nn.Linear(  # type: ignore[attr-defined]
            self.rank, out_features, bias=False, device=device, dtype=dtype
        )

        # If NOT on meta device, initialize now
        if device.type != "meta":
            nn.init.kaiming_uniform_(linear.lora_a.weight, a=math.sqrt(5))  # type: ignore[attr-defined]
            nn.init.zeros_(linear.lora_b.weight)  # type: ignore[attr-defined]
            linear.lora_a.weight.requires_grad_(True)  # type: ignore[attr-defined]
            linear.lora_b.weight.requires_grad_(True)  # type: ignore[attr-defined]

        return linear

    def post_optimizer_hook(self, model: Union[nn.Module, list[nn.Module]]) -> None:
        """Post-optimizer hook (no-op for LoRA)."""
        pass


# Register the LoRA converter
register_model_converter(LoRAConverter, "lora")
