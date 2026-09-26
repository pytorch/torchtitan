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

    def _gather_shared_input_and_compute_projections(
        self, x_TD: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather the shared input once, then compute both input projections."""
        ep_enabled = spmd_sparse_mesh() is not None
        sp_enabled = spmd_dense_sp_enabled()
        tp_group = spmd_mesh_group(MeshAxisName.TP)
        if ep_enabled and tp_group is not None:
            src = spmd.S(0) if sp_enabled else spmd.I
            x_TD = spmd.redistribute(
                x_TD,
                tp_group,
                src=src,
                dst=spmd.R,
                backward_options={"op_dtype": x_TD.dtype},
            )

        gate_up_T2F = self.w13(x_TD)
        gate_out_T1 = self.gate(x_TD)
        if ep_enabled and sp_enabled and tp_group is not None:
            gate_out_T1 = spmd.redistribute(
                gate_out_T1,
                tp_group,
                src=spmd.R,
                dst=spmd.S(0),
                backward_options={"op_dtype": gate_out_T1.dtype},
            )
        return gate_up_T2F, gate_out_T1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up_T2F, gate_out_T1 = remat.region(
            self._gather_shared_input_and_compute_projections,
            self.remat_region_name("input_projections"),
            recompute=self.remat_should_recompute("input_projections"),
        )(x)
        remat.recompute_needs_tensor(gate_up_T2F)
        gate_TF, up_TF = gate_up_T2F.unbind(-2)
        out_TD = remat.region(
            self.w2,
            self.remat_region_name("w2"),
            recompute=self.remat_should_recompute("w2"),
        )(self.activation_fn(gate_TF, up_TF))
        remat.recompute_needs_tensor(out_TD, gate_out_T1)
        return torch.sigmoid(gate_out_T1) * out_TD


__all__ = ["SigmoidGatedFeedForward"]
