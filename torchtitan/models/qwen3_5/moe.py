# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5-specific MoE components."""

from dataclasses import dataclass

import torch
import torch_remat as remat

from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear


class SigmoidGatedFeedForward(FeedForward):
    """Qwen3.5 shared FFN with a per-token sigmoid output gate."""

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        gate: Linear.Config

    def __init__(self, config: Config):
        super().__init__(config)
        self.gate = config.gate.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_TD = super().forward(x)
        gate_out_TD = remat.region(
            self.gate,
            self.remat_region_name("gate"),
            recompute=self.remat_should_recompute("gate"),
        )(x)
        remat.recompute_needs_tensor(out_TD, gate_out_TD)
        return torch.sigmoid(gate_out_TD) * out_TD


__all__ = ["SigmoidGatedFeedForward"]
