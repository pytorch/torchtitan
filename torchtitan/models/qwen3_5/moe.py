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
from torchtitan.distributed.spmd_types import _per_axis_types, spmd_mesh_group
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
        tp_group = spmd_mesh_group(MeshAxisName.TP)
        if tp_group is not None:
            sharding_config = self._sharding_config
            assert sharding_config is not None
            assert sharding_config.in_src_shardings is not None
            input_layout = sharding_config.in_src_shardings["x"]
            input_tp_type = _per_axis_types(input_layout).get(MeshAxisName.TP)
            assert input_tp_type is not None
            # w13 and gate both consume x, so gather once at their common
            # module boundary instead of once per projection.
            x = spmd.redistribute(
                x,
                tp_group,
                src=input_tp_type,
                dst=spmd.R,
                backward_options={"op_dtype": x.dtype},
            )

        out_TD = super().forward(x)
        gate_out_TD = remat.region(
            self.gate,
            self.remat_region_name("gate"),
            recompute=self.remat_should_recompute("gate"),
        )(x)
        remat.recompute_needs_tensor(out_TD, gate_out_TD)
        return torch.sigmoid(gate_out_TD) * out_TD


__all__ = ["SigmoidGatedFeedForward"]
