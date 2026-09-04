# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel attention kernels.

Tensor suffixes: ``T`` tokens, ``H`` heads, ``K`` qk head dim, ``V`` v head dim.
"""

from dataclasses import dataclass
from typing import ClassVar, Literal

import spmd_types as spmd

import torch
import torch.distributed as dist

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.spmd_types import current_spmd_mesh

from torchtitan.models.common.attention import FlexAttention

__all__ = [
    "ContextParallelKernel",
    "AllGatherCPFlexAttention",
    "UlyssesCPFlexAttention",
]

_SEQ_DIM = 0
_HEAD_DIM = 1


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
        reduce_dtype: Literal["float32", "bfloat16"] | None = None
        """Dtype of the backward reduce-scatter. None keeps the input dtype."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.reduce_dtype = (
            TORCH_DTYPE_MAP[config.reduce_dtype] if config.reduce_dtype else None
        )

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        cp_group = self.cp_group
        k_THK, v_THV = (
            spmd.redistribute(
                x,
                cp_group,
                src=spmd.S(_SEQ_DIM),
                dst=spmd.R,
                backward_options={"op_dtype": self.reduce_dtype or x.dtype},
            )
            for x in (k_THK, v_THV)
        )
        return super().forward(q_THK, k_THK, v_THV, **kwargs)


class UlyssesCPFlexAttention(ContextParallelKernel, FlexAttention):
    """Run FlexAttention with sequence-to-head all-to-all redistribution."""

    @dataclass(kw_only=True, slots=True)
    class Config(FlexAttention.Config):
        shard_attention_mask: ClassVar[bool] = False
        shard_attention_heads: ClassVar[bool] = True

    @staticmethod
    def _reshard(
        x: torch.Tensor, cp_group: dist.ProcessGroup, *, src: int, dst: int
    ) -> torch.Tensor:
        """Move the CP sharding of ``x`` from tensor dim ``src`` to ``dst``."""
        return spmd.redistribute(
            x.contiguous(),
            cp_group,
            src=spmd.S(src),
            dst=spmd.S(dst),
        )

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        cp_group = self.cp_group
        # Shard heads instead of tokens: (T/cp, H, *) -> (T, H/cp, *).
        q_THK, k_THK, v_THV = (
            self._reshard(x, cp_group, src=_SEQ_DIM, dst=_HEAD_DIM)
            for x in (q_THK, k_THK, v_THV)
        )
        out_THV = super().forward(q_THK, k_THK, v_THV, **kwargs)
        # Back to sharded tokens: (T, H/cp, V) -> (T/cp, H, V).
        return self._reshard(out_THV, cp_group, src=_HEAD_DIM, dst=_SEQ_DIM)
