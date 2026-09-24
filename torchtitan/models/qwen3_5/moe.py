# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5-specific MoE components."""

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch_remat as remat

from torchtitan.distributed.spmd_types import spmd_dense_sp_enabled, spmd_sparse_mesh
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import _redistribute_tp


class SigmoidGatedFeedForward(FeedForward):
    """Qwen3.5 shared FFN with a per-token sigmoid output gate."""

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        gate: Linear.Config

    def __init__(self, config: Config):
        super().__init__(config)
        self.gate = config.gate.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if spmd_sparse_mesh() is not None:
            src = spmd.S(0) if spmd_dense_sp_enabled() else spmd.I
            x = _redistribute_tp(x, src=src, dst=spmd.R)

        out_TD = super().forward(x)
        gate_out_TD = remat.region(
            self.gate,
            self.remat_region_name("gate"),
            recompute=self.remat_should_recompute("gate"),
        )(x)
        if spmd_sparse_mesh() is not None and spmd_dense_sp_enabled():
            gate_out_TD = _redistribute_tp(
                gate_out_TD,
                src=spmd.R,
                dst=spmd.S(0),
            )
        remat.recompute_needs_tensor(out_TD, gate_out_TD)
        return torch.sigmoid(gate_out_TD) * out_TD


__all__ = ["SigmoidGatedFeedForward"]
