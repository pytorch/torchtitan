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

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import (
    spmd_dense_sp_enabled,
    spmd_mesh_group,
    spmd_sparse_mesh,
)
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
        ep_enabled = spmd_sparse_mesh() is not None
        sp_enabled = spmd_dense_sp_enabled()
        tp_group = spmd_mesh_group(MeshAxisName.TP)
        if ep_enabled and tp_group is not None:
            src = spmd.S(0) if sp_enabled else spmd.I
            if sp_enabled:
                x = remat.region(
                    spmd.redistribute,
                    self.remat_region_name("tp_communication.input_gather"),
                    recompute=self.remat_should_recompute("tp_communication"),
                )(
                    x,
                    tp_group,
                    src=src,
                    dst=spmd.R,
                    backward_options={"op_dtype": x.dtype},
                )
            else:
                x = spmd.redistribute(
                    x,
                    tp_group,
                    src=src,
                    dst=spmd.R,
                    backward_options={"op_dtype": x.dtype},
                )

        out_TD = super().forward(x)
        gate_out_TD = remat.region(
            self.gate,
            self.remat_region_name("gate"),
            recompute=self.remat_should_recompute("gate"),
        )(x)
        if ep_enabled and sp_enabled and tp_group is not None:
            gate_out_TD = spmd.redistribute(
                gate_out_TD,
                tp_group,
                src=spmd.R,
                dst=spmd.S(0),
                backward_options={"op_dtype": gate_out_TD.dtype},
            )
        remat.recompute_needs_tensor(out_TD, gate_out_TD)
        return torch.sigmoid(gate_out_TD) * out_TD


__all__ = ["SigmoidGatedFeedForward"]
