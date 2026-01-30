# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import register_model_converter
from torchtitan.tools.logging import logger


class LoRALinear(nn.Module):
    """LoRA wrapper for any linear layer.

    Wraps an existing linear layer and adds LoRA adapters.
    Implements: x -> linear(x) + (alpha / rank) * B @ A @ x

    See: https://arxiv.org/abs/2106.09685

    Args:
        linear: The linear layer to wrap (nn.Linear, Float8Linear, etc.)
        rank: Rank of the low-rank approximation.
        alpha: Scaling factor.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        linear: nn.Module,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Get dtype from the linear layer's weight
        dtype = linear.weight.dtype if hasattr(linear, 'weight') else None

        # LoRA layers on meta device
        self.lora_a = nn.Linear(linear.in_features, rank, bias=False, device="meta", dtype=dtype)
        self.lora_b = nn.Linear(rank, linear.out_features, bias=False, device="meta", dtype=dtype)

    @property
    def weight(self):
        """Expose wrapped linear's weight for compatibility."""
        return self.linear.weight

    @property
    def bias(self):
        """Expose wrapped linear's bias for compatibility."""
        return self.linear.bias

    @property
    def in_features(self):
        """Expose wrapped linear's in_features for compatibility."""
        return self.linear.in_features

    @property
    def out_features(self):
        """Expose wrapped linear's out_features for compatibility."""
        return self.linear.out_features

    def initialize_parameters(self):
        """Initialize LoRA parameters after materialization."""
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def adapter_params(self) -> list[str]:
        """Return names of LoRA adapter parameters."""
        return ["lora_a.weight", "lora_b.weight"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base linear forward (works with nn.Linear, Float8Linear, etc.)
        out = self.linear(x)

        # LoRA path - use modules directly to preserve gradient flow through DTensor
        lora_x = self.dropout(x)
        lora_hidden = self.lora_a(lora_x)  # [batch, seq, rank]
        lora_out = self.lora_b(lora_hidden)  # [batch, seq, out_features]

        # Both out and lora_out are plain tensors (use_local_output=True in TP layer_plan)
        return out + self.scaling * lora_out


class LoRAConverter:
    """Model converter that adds LoRA adapters to Linear layers."""

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.rank = job_config.lora.rank
        self.alpha = job_config.lora.alpha
        self.dropout = job_config.lora.dropout
        self._lora_modules: list[LoRALinear] = []

        logger.info(
            f"LoRA training active with rank={self.rank}, alpha={self.alpha}, "
            f"dropout={self.dropout}"
        )

    def convert(self, model: nn.Module) -> None:
        """Inplace conversion of the model to use LoRA adapters."""
        self._apply_lora(model)
        self._hook_init_weights(model)

        logger.info(f"Converted {len(self._lora_modules)} linear modules to LoRALinear")

    def _apply_lora(self, model: nn.Module) -> None:
        """Replace Linear layers with LoRALinear wrappers."""
        for module in list(model.modules()):
            for name, child in list(module._modules.items()):
                if isinstance(child, nn.Linear) and not isinstance(child, LoRALinear):
                    if name == "output":
                        continue
                    lora_linear = LoRALinear(
                        linear=child,
                        rank=self.rank,
                        alpha=self.alpha,
                        dropout=self.dropout,
                    )
                    setattr(module, name, lora_linear)
                    self._lora_modules.append(lora_linear)

    def _hook_init_weights(self, model: nn.Module) -> None:
        """Hook into init_weights to freeze base params and initialize LoRA."""
        original_init_weights = model.init_weights
        lora_modules = self._lora_modules
        model_ref = [model]

        def new_init_weights(*args, **kwargs):
            if callable(original_init_weights):
                original_init_weights(*args, **kwargs)

            for ll in lora_modules:
                ll.initialize_parameters()

            m = model_ref[0]

            trainable_count = 0
            frozen_count = 0
            for name, param in m.named_parameters():
                if "lora_a" in name or "lora_b" in name:
                    param.requires_grad_(True)
                    trainable_count += 1
                else:
                    param.requires_grad_(False)
                    frozen_count += 1

            total_params = sum(p.numel() for p in m.parameters())
            trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
            logger.info(
                f"LoRA: frozen {frozen_count} params, trainable {trainable_count} params, "
                f"trainable params: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)"
            )

        object.__setattr__(model, "init_weights", new_init_weights)

    def post_optimizer_hook(self, model: Union[nn.Module, list[nn.Module]]) -> None:
        """Post-optimizer hook (no-op for LoRA)."""
        pass


# Register the LoRA converter
register_model_converter(LoRAConverter, "lora")
