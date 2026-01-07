# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) module for efficient fine-tuning.
    
    Creates a low-rank decomposition that runs in parallel with a base linear layer.
    The output is: base_output + scale * lora_B(lora_A(x))
    
    Args:
        in_features: Input dimension
        out_features: Output dimension  
        rank: LoRA rank (bottleneck dimension)
        alpha: LoRA scaling factor (scale = alpha / rank)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the LoRA delta: scale * lora_B(lora_A(x))
        
        Note: This returns only the delta, not base + delta.
        The caller is responsible for adding this to the base output.
        """
        return self.lora_B(self.lora_A(x)) * self.scale
    
    def init_weights(self) -> None:
        """
        Initialize LoRA weights following standard practice:
        - A: Kaiming uniform initialization
        - B: Zeros (so LoRA starts as identity)
        """
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
