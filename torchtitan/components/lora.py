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
        self.in_dim = linear.in_features
        self.out_dim = linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Get dtype from the linear layer's weight
        dtype = linear.weight.dtype if hasattr(linear, 'weight') else None

        # LoRA layers on meta device
        self.lora_a = nn.Linear(self.in_dim, rank, bias=False, device="meta", dtype=dtype)
        self.lora_b = nn.Linear(rank, self.out_dim, bias=False, device="meta", dtype=dtype)

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
        return self.in_dim

    @property
    def out_features(self):
        """Expose wrapped linear's out_features for compatibility."""
        return self.out_dim

    def to_empty(
        self, *, device: Optional[Union[str, torch.device, int]], recurse: bool = True
    ):
        self.lora_a.to_empty(device=device, recurse=recurse)
        self.lora_b.to_empty(device=device, recurse=recurse)
        return self

    def initialize_parameters(self):
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def adapter_params(self) -> list[str]:
        """Return names of LoRA adapter parameters."""
        return ["lora_a.weight", "lora_b.weight"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base linear forward (works with nn.Linear, Float8Linear, etc.)
        out = self.linear(x)

        # LoRA path: out += scale * (x @ A.T @ B.T)
        x = self.dropout(x)
        lora_out = self.lora_b(self.lora_a(x))
        out = out + self.scaling * lora_out

        return out


class LoRAConverter:
    """Model converter that adds LoRA adapters to Linear layers."""

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.rank = job_config.lora.rank
        self.alpha = job_config.lora.alpha
        self.dropout = job_config.lora.dropout

        logger.info(
            f"LoRA training active with rank={self.rank}, alpha={self.alpha}, "
            f"dropout={self.dropout}"
        )

    def convert(self, model: nn.Module) -> None:
        """Inplace conversion of the model to use LoRA adapters."""
        init_weights_fn = model.init_weights
        lora_count = 0

        def make_init_wrapper(prev_fn, ll: LoRALinear | None = None, final_log: bool = False):
            def wrapped(*args, **kwargs):
                if callable(prev_fn):
                    prev_fn(*args, **kwargs)
                if ll is not None:
                    ll.initialize_parameters()
                    ll.lora_a.weight.requires_grad = True
                    ll.lora_b.weight.requires_grad = True
                if final_log:
                    trainable_params = sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    )
                    logger.info(
                        f"LoRA parameters initialized for {lora_count} layers, "
                        f"trainable params: {trainable_params:,}"
                    )
            return wrapped

        for module in list(model.modules()):
            for param in module._parameters.values():
                if param is not None:
                    param.requires_grad_(False)

            for name, child in list(module._modules.items()):
                if isinstance(child, nn.Linear) and not isinstance(child, LoRALinear):
                    lora_linear = LoRALinear(
                        linear=child,
                        rank=self.rank,
                        alpha=self.alpha,
                        dropout=self.dropout,
                    )
                    setattr(module, name, lora_linear)
                    lora_count += 1
                    init_weights_fn = make_init_wrapper(init_weights_fn, lora_linear)

        # Add final logging wrapper
        init_weights_fn = make_init_wrapper(init_weights_fn, final_log=True)
        object.__setattr__(model, "init_weights", init_weights_fn)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Converted {lora_count} linear modules to LoRALinear")
        logger.info(
            f"Trainable params: {trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )

    def post_optimizer_hook(self, model: Union[nn.Module, list[nn.Module]]) -> None:
        """Post-optimizer hook (no-op for LoRA)."""
        pass


# Register the LoRA converter
register_model_converter(LoRAConverter, "lora")
