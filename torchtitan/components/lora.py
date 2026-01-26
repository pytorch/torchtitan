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
    """LoRA linear layer.

    Implements: x -> W_0 @ x + (alpha / rank) * B @ A @ x

    See: https://arxiv.org/abs/2106.09685

    Args:
        in_dim: Input dimension.
        out_dim: Output dimension.
        rank: Rank of the low-rank approximation.
        alpha: Scaling factor.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Setup weight - reuse provided tensor or create on meta device
        if weight is not None:
            self.register_parameter("weight", nn.Parameter(weight, requires_grad=False))
        else:
            self.register_parameter(
                "weight",
                nn.Parameter(
                    torch.empty(out_dim, in_dim, device="meta", dtype=dtype),
                    requires_grad=False,
                ),
            )

        # Setup bias
        if bias is not None:
            self.register_parameter("bias", nn.Parameter(bias, requires_grad=False))
        else:
            self.register_parameter(
                "bias",
                nn.Parameter(
                    torch.empty(out_dim, device="meta", dtype=dtype),
                    requires_grad=False,
                ),
            )

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # LoRA layers on meta device
        self.lora_a = nn.Linear(in_dim, rank, bias=False, device="meta", dtype=dtype)
        self.lora_b = nn.Linear(rank, out_dim, bias=False, device="meta", dtype=dtype)

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
        # Base linear: out = x @ W.T + bias
        out = F.linear(x, self.weight, self.bias)  # type: ignore[arg-type]

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
        num_replaced = 0
        for module in list(model.modules()):
            for child_name, child in module.named_children():
                # TODO: Add support for GroupedExperts.
                if isinstance(child, nn.Linear) and not isinstance(child, LoRALinear):
                    original_weight = child.weight.data
                    original_bias = child.bias.data if child.bias is not None else None

                    # Break reference chain before creating new module
                    child.weight = None  # type: ignore[assignment]
                    child.bias = None  # type: ignore[assignment]

                    lora_linear = LoRALinear(
                        in_dim=child.in_features,
                        out_dim=child.out_features,
                        rank=self.rank,
                        alpha=self.alpha,
                        dropout=self.dropout,
                        weight=original_weight,
                        bias=original_bias,
                        dtype=original_weight.dtype,
                    )
                    setattr(module, child_name, lora_linear)
                    num_replaced += 1

        # Wrap init_weights to also initialize LoRA parameters
        original_init_weights = model.init_weights

        def init_weights_with_lora(*args, **kwargs):
            if callable(original_init_weights):
                original_init_weights(*args, **kwargs)
            self._init_lora_params(model)

        object.__setattr__(model, "init_weights", init_weights_with_lora)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Converted {num_replaced} linear modules to LoRALinear")
        logger.info(
            f"Trainable params: {trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )

    def _set_lora_requires_grad(self, model: nn.Module) -> None:
        """Set requires_grad: True for LoRA params, False for others."""
        for name, param in model.named_parameters():
            param.requires_grad = "lora_a" in name or "lora_b" in name

    def _init_lora_params(self, model: nn.Module) -> None:
        """Initialize LoRA parameters after model initialization."""
        lora_layer_count = 0
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.initialize_parameters()
                lora_layer_count += 1

        self._set_lora_requires_grad(model)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"LoRA parameters initialized for {lora_layer_count} layers, "
            f"trainable params: {trainable_params:,}"
        )

    def post_optimizer_hook(self, model: Union[nn.Module, list[nn.Module]]) -> None:
        """Post-optimizer hook (no-op for LoRA)."""
        pass


# Register the LoRA converter
register_model_converter(LoRAConverter, "lora")
