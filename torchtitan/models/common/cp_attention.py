# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel attention kernels.

Tensor suffixes: ``T`` tokens, ``H`` heads, ``K`` qk head dim, ``V`` v head dim.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.tensor.experimental._context_parallel import flex_cp_allgather

from torchtitan.distributed.spmd_types import current_spmd_mesh

from torchtitan.models.common.attention import FlexAttention

__all__ = [
    "ContextParallelKernel",
    "AllGatherCPFlexAttention",
]

_SEQ_DIM = 0


class ContextParallelKernel:
    """Mixin for attention kernels that own their CP collectives."""

    @property
    def cp_group(self) -> dist.ProcessGroup:
        """Return the active multi-rank CP process group."""
        mesh = current_spmd_mesh()
        if mesh is None:
            raise RuntimeError(
                f"{type(self).__name__} requires an active SPMD mesh context."
            )
        mesh_axis_names = mesh.mesh_dim_names or ()
        cp_group = mesh.get_group("cp") if "cp" in mesh_axis_names else None
        if cp_group is None or cp_group.size() == 1:
            raise RuntimeError(f"{type(self).__name__} requires an active CP mesh.")
        return cp_group


class AllGatherCPFlexAttention(ContextParallelKernel, FlexAttention):
    """FlexAttention with sharded Q and all-gathered K/V."""

    @dataclass(kw_only=True, slots=True)
    class Config(FlexAttention.Config):
        pass

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # TODO(fegin): replace flex_cp_allgather with spmd_types.redistribute.
        pg_name = dist._get_process_group_name(self.cp_group)
        k_THK, v_THV = flex_cp_allgather(
            k_THK.contiguous(), v_THV.contiguous(), _SEQ_DIM, pg_name
        )
        return super().forward(q_THK, k_THK, v_THV, **kwargs)
