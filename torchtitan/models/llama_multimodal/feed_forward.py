# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Llama 2 is licensed under the LLAMA 2 Community License,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


from typing import Callable, Optional

import torch.nn.functional as F
from torch import nn


class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (Optional[float]): Custom multiplier for hidden dimension. Defaults to None.
        activation:
        dropout
        enable_w3 (bool): Whether to enable the third linear layer. Defaults to True.

    Attributes:
        w1 (Linear): Linear transformation for the first layer, which projects input from input dim to
            hidden dim, and multiplies by the projection from w3 for activation and second layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Optional[Linear]): Linear transformation for the first layer to be multiplied by the
            projection of w1 as well.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        activation: Callable = F.silu,
        dropout: float = 0.0,
        enable_w3: bool = True,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.activation = activation
        self.dropout = dropout
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False) if enable_w3 else None

    def forward(self, x):
        h = self.activation(self.w1(x))
        if self.w3:
            h = self.w3(x) * h
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.w2(h)

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            if linear:
                nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
