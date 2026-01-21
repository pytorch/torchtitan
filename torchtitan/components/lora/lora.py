# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import register_model_converter
from torchtitan.tools.logging import logger


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) fine-tuning.

    Args:
        rank: Rank of the low-rank approximation. Default: 8.
        alpha: Scaling factor for the low-rank approximation. Default: 16.0.
        dropout: Dropout probability for LoRA layers. Default: 0.0.
        TODO: add support to layers to apply, e.g. only attention layers or all linear.
    """

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0


def get_lora_config(job_config: JobConfig) -> LoRAConfig:
    """Get LoRA config from job_config, using defaults if not specified."""
    lora_config = job_config.lora
    return LoRAConfig(
        rank=lora_config.rank,
        alpha=lora_config.alpha,
        dropout=lora_config.dropout,
    )


class _LoRALinearFunction(torch.autograd.Function):
    """Memory-efficient LoRA linear computation.

    Forward: out = X @ W.T + bias + scale * (X @ A.T @ B.T)

    Memory optimizations:
    - Only saves X, A, B for backward
    - Uses in-place addmm_ operations
    """

    @staticmethod
    def forward(ctx, X, W, bias, A, B, scale):  # type: ignore[override]
        orig_shape = X.shape
        X_2d = X.view(-1, X.shape[-1]) if X.dim() == 3 else X

        out = torch.empty(X_2d.shape[0], W.shape[0], dtype=X.dtype, device=X.device)
        torch.mm(X_2d, W.t(), out=out)

        if bias is not None:
            out.add_(bias)

        out.addmm_(X_2d @ A.T, B.T, alpha=scale)

        if X.dim() == 3:
            out = out.view(orig_shape[0], orig_shape[1], -1)

        ctx.custom_saved_tensors = (W, scale)
        ctx.save_for_backward(A, B, X)
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, dY):  # type: ignore[override]
        W, scale = ctx.custom_saved_tensors
        A, B, X = ctx.saved_tensors

        batch, seq_len, hd = X.shape
        dY = dY.reshape(-1, dY.shape[-1])
        X = X.reshape(-1, X.shape[-1])
        A, B = A.t(), B.t()

        d_A = torch.empty_like(A)
        d_B = torch.empty_like(B)
        d_A.addmm_(X.t(), dY @ B.t(), alpha=scale, beta=0)
        d_B.addmm_(A.t() @ X.t(), dY, alpha=scale, beta=0)

        dX = dY @ W
        dX.addmm_(dY @ B.t(), A.t(), alpha=scale)
        d_bias = dY.sum(dim=0) if ctx.has_bias else None

        return dX.view(batch, seq_len, hd), None, d_bias, d_A.t(), d_B.t(), None


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
        use_bias: Whether to include bias.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        use_bias: bool = False,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha
        self.use_bias = use_bias
        self.disabled = False

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
        if use_bias:
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
        else:
            self.register_parameter("bias", None)

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
        _lora_a_init_params(self.lora_a)
        _lora_b_init_params(self.lora_b)

    def adapter_params(self) -> list[str]:
        """Return names of LoRA adapter parameters."""
        return ["lora_a.weight", "lora_b.weight"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.disabled:
            return F.linear(x, self.weight, self.bias)  # type: ignore[arg-type]

        return _LoRALinearFunction.apply(
            self.dropout(x),
            self.weight,
            self.bias,
            self.lora_a.weight,
            self.lora_b.weight,
            self.alpha / self.rank,
        )


def _lora_a_init_params(x: nn.Linear) -> None:
    """Initialize LoRA A weight to Kaiming uniform."""
    nn.init.kaiming_uniform_(x.weight, a=math.sqrt(5))


def _lora_b_init_params(x: nn.Linear) -> None:
    """Initialize LoRA B weight to zeros."""
    nn.init.zeros_(x.weight)


class LoRAConverter:
    """Model converter that adds LoRA adapters to Linear layers."""

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        lora_config = get_lora_config(job_config)
        self.rank = lora_config.rank
        self.alpha = lora_config.alpha
        self.dropout = lora_config.dropout
        self._converted_model: Optional[nn.Module] = None

        logger.info(
            f"LoRA training active with rank={self.rank}, alpha={self.alpha}, "
            f"dropout={self.dropout}"
        )

    def convert(self, model: nn.Module) -> None:
        """Inplace conversion of the model to use LoRA adapters."""
        replacements = []
        for module in model.modules():
            for child_name, child in module.named_children():
                if isinstance(child, nn.Linear) and not isinstance(child, LoRALinear):
                    replacements.append((module, child_name, child))

        for parent_module, child_name, child in replacements:
            has_bias = child.bias is not None
            original_weight = child.weight.data
            original_bias = child.bias.data if has_bias else None

            # Break reference chain before creating new module
            child.weight = None  # type: ignore[assignment]
            if has_bias:
                child.bias = None  # type: ignore[assignment]

            lora_linear = LoRALinear(
                in_dim=child.in_features,
                out_dim=child.out_features,
                rank=self.rank,
                alpha=self.alpha,
                dropout=self.dropout,
                use_bias=has_bias,
                weight=original_weight,
                bias=original_bias,
                dtype=child.weight.dtype
                if child.weight is not None
                else original_weight.dtype,
            )
            setattr(parent_module, child_name, lora_linear)

        self._set_lora_requires_grad(model)
        self._converted_model = model

        # Wrap init_weights to also initialize LoRA parameters
        original_init_weights = model.init_weights

        def init_weights_with_lora(*args, **kwargs):
            if callable(original_init_weights):
                original_init_weights(*args, **kwargs)
            self._init_lora_params(model)

        object.__setattr__(model, "init_weights", init_weights_with_lora)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Converted {len(replacements)} linear modules to LoRALinear")
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
