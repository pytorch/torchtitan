# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from typing import List, Optional, Union

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
        apply_to_all_linears: If True, apply LoRA to all Linear layers.
            If False, only apply to attention layers (wq, wk, wv, wo). Default: True.
    """

    rank: int = 8
    """Rank of the low-rank approximation"""

    alpha: float = 16.0
    """Scaling factor for the low-rank approximation"""

    dropout: float = 0.0
    """Dropout probability for LoRA layers"""

    apply_to_all_linears: bool = True
    """If True, apply LoRA to all Linear layers. If False, only apply to attention layers."""


def get_lora_config(job_config: JobConfig) -> LoRAConfig:
    """Get LoRA config from job_config, using defaults if not specified.

    The LoRA config can be specified in the TOML file under [lora] section:
    ```toml
    [lora]
    rank = 8
    alpha = 16.0
    dropout = 0.0
    apply_to_all_linears = true
    ```

    If not specified, default values from LoRAConfig will be used.
    """
    lora_section = job_config.lora
    return LoRAConfig(
        rank=lora_section.rank,
        alpha=lora_section.alpha,
        dropout=lora_section.dropout,
        apply_to_all_linears=lora_section.apply_to_all_linears,
    )


class LoRALinear(nn.Module):
    """LoRA linear layer as introduced in `LoRA: Low-Rank Adaptation of Large Language Models <https://arxiv.org/abs/2106.09685>`_.

    LoRA perturbs a given layer via a low-rank approximation where only
    the rank decomposition matrices are trainable. In a linear layer instead of
    :math:`x \\mapsto W_0x` a LoRALinear layer is defined as
    :math:`x \\mapsto W_0x + (\\alpha / r)BAx`, where :math:`r` is the rank of
    the matrices :math:`A` and :math:`B` and :math:`\\alpha` is a scaling factor.
    As in the original implementation, we support dropout before multiplication
    by the low-rank matrices.

    Args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        rank (int): rank of the low-rank approximation
        alpha (float): scaling factor for the low-rank approximation
        dropout (float): dropout probability. Default: 0.0
        use_bias (bool): whether to include bias in the original linear layer.
            Default: False

    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        use_bias: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha
        self.use_bias = use_bias

        # Setup weight and bias
        linear = nn.Linear(in_features=in_dim, out_features=out_dim, bias=self.use_bias)
        weight = linear.weight
        bias = linear.bias if self.use_bias else None

        # 'self.disabled' is a flag showing whether to turn off LoRA adapters,
        # this can be used in DPO for treating the lora adapters as the policy model
        # and disabling it to treat the base model as the reference model
        self.disabled = False
        self.register_parameter("weight", nn.Parameter(weight))
        self.register_parameter(
            "bias", nn.Parameter(bias) if bias is not None else None
        )
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.lora_a = nn.Linear(in_features=in_dim, out_features=rank, bias=False)
        self.lora_b = nn.Linear(in_features=rank, out_features=out_dim, bias=False)
        self.merged = False
        self.initialize_parameters()

    def to_empty(
        self, *, device: Optional[Union[str, torch.device, int]], recurse: bool = True
    ):
        self.lora_a.to_empty(device=device, recurse=recurse)
        self.lora_b.to_empty(device=device, recurse=recurse)

    def initialize_parameters(self):
        # Initialize as in
        # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L119
        _lora_a_init_params(self.lora_a)
        _lora_b_init_params(self.lora_b)

    def adapter_params(self) -> list[str]:
        """
        Return a list of strings corresponding to the names of the ``nn.Parameter`` s in
        the model coming from the adapter.

        For LoRA this means lora_a.weight and lora_b.weight.
        """
        # NOTE: this function has to be updated if the names of "lora_a" and "lora_b"
        # in this module change.
        adapter_params = ["lora_a.weight", "lora_b.weight"]
        return adapter_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape ``(..., in_dim)``

        Returns:
            torch.Tensor: output tensor with shape ``(..., out_dim)``

        """
        out = F.linear(x, self.weight, self.bias)
        if self.disabled:
            return out
        lora_out = self.lora_a(self.dropout(x))
        lora_out = (self.alpha / self.rank) * self.lora_b(lora_out)
        return out + lora_out


def _lora_a_init_params(x: nn.Linear) -> None:
    """
    Initialize LoRA A weight to Kaiming uniform.
    """
    nn.init.kaiming_uniform_(x.weight, a=math.sqrt(5))


def _lora_b_init_params(x: nn.Linear) -> None:
    """
    Initialize LoRA B weight to zeros.
    """
    nn.init.zeros_(x.weight)


class LoRAConverter:
    """Model converter that adds LoRA adapters to Linear layers.

    This converter replaces nn.Linear layers with LoRALinear layers and sets
    requires_grad=True only for LoRA parameters, freezing all other parameters.

    Configuration can be specified in the TOML file under [lora] section:
    ```toml
    [lora]
    rank = 8
    alpha = 16.0
    dropout = 0.0
    apply_to_all_linears = true
    ```
    """

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        lora_config = get_lora_config(job_config)
        self.rank = lora_config.rank
        self.alpha = lora_config.alpha
        self.dropout = lora_config.dropout
        self.apply_to_all_linears = lora_config.apply_to_all_linears

        logger.info(
            f"LoRA config: rank={self.rank}, alpha={self.alpha}, "
            f"dropout={self.dropout}, apply_to_all_linears={self.apply_to_all_linears}"
        )

    def convert(self, model: nn.Module) -> None:
        """Inplace conversion of the model to use LoRA adapters."""
        # First, freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Collect all Linear layers to replace (to avoid modifying while iterating)
        replacements = []
        for name, module in model.named_modules():
            for child_name, child in module.named_children():
                if isinstance(child, nn.Linear) and not isinstance(child, LoRALinear):
                    replacements.append((module, child_name, child))

        # Replace Linear layers with LoRALinear
        for parent_module, child_name, child in replacements:
            lora_linear = LoRALinear(
                in_dim=child.in_features,
                out_dim=child.out_features,
                rank=self.rank,
                alpha=self.alpha,
                dropout=self.dropout,
                use_bias=child.bias is not None,
            )
            # First move to the same device and dtype as the original weights
            lora_linear = lora_linear.to(
                device=child.weight.device, dtype=child.weight.dtype
            )
            # Then copy the original weights (after dtype conversion)
            lora_linear.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                lora_linear.bias.data.copy_(child.bias.data)
            # Replace the module
            setattr(parent_module, child_name, lora_linear)

        # Enable gradients only for LoRA parameters
        for name, param in model.named_parameters():
            if "lora_a" in name or "lora_b" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Log the number of trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        logger.info(
            f"LoRA adapters added. Trainable parameters: {trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]) -> None:
        """Post-optimizer hook (no-op for LoRA)."""
        pass


# Register the LoRA converter
register_model_converter(LoRAConverter, "lora")
