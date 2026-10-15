# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchtitan.models.base.module import Module


class BatchedLinear(Module):
    """Per-head linear map with a flattened ``[H * D_out, D_in]`` weight."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        n_heads: int
        in_features: int
        out_features: int
        param_init: dict | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.out_features = config.out_features
        self.n_batches = config.n_heads
        self.weight = nn.Parameter(
            torch.empty(
                config.n_heads * config.out_features,
                config.in_features,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *prefix, H, D_in = x.shape
        weight = self.weight.view(H, self.out_features, D_in)
        x_h = x.reshape(-1, H, D_in).transpose(0, 1)
        out = torch.bmm(x_h, weight.transpose(-2, -1))
        return out.transpose(0, 1).reshape(*prefix, H, -1)
