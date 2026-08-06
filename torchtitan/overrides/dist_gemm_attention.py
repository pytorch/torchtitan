# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Attention components that use the fused TP+SP linear primitives.

:class:`AllGatherFusedQKVLinear` and :class:`ReduceScatterLinear` are drop-in
replacements for the stock QKV and output projections. They keep the stock
parameter layouts and only move the TP collective into the GEMM, using
:class:`~torchtitan.distributed.dist_linear.AllGatherLinear` and
:class:`~torchtitan.distributed.dist_linear.LinearReduceScatter`.

Those primitives live under ``torchtitan/distributed`` rather than here, because
nothing about them is attention-specific: FFN and MoE projections can use the
same pair. What stays in this module is the wiring -- the QKV-specific reshaping
around the collective, and the override registrations.

The ``dist_gemm_attention`` override only wires these blocks into
:class:`GQAttention` and removes the parent attention-boundary all-gather.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

from torchtitan.config import derive, override
from torchtitan.distributed.dist_linear import (
    AllGatherLinear,
    LinearReduceScatter,
    reserve_for_input,
)
from torchtitan.models.common.attention import FusedQKVLinear, GQAttention
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.sharding import ShardingConfig

if TYPE_CHECKING:
    from torchtitan.distributed.parallel_dims import ParallelDims


def to_local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def tp_mesh_axis(dtensor: DTensor) -> int | None:
    mesh_dim_names = dtensor.device_mesh.mesh_dim_names
    if mesh_dim_names is None or "tp" not in mesh_dim_names:
        return None
    return tuple(mesh_dim_names).index("tp")


def is_tp_sequence_sharded(tensor: torch.Tensor) -> bool:
    if not isinstance(tensor, DTensor):
        return False
    tp_axis = tp_mesh_axis(tensor)
    if tp_axis is None:
        return False
    placement = tensor.placements[tp_axis]
    return isinstance(placement, Shard) and placement.dim == 1


def is_tp_feature_sharded(tensor: torch.Tensor) -> bool:
    if not isinstance(tensor, DTensor):
        return False
    tp_axis = tp_mesh_axis(tensor)
    if tp_axis is None:
        return False
    placement = tensor.placements[tp_axis]
    if not isinstance(placement, Shard):
        return False
    dim = placement.dim + tensor.ndim if placement.dim < 0 else placement.dim
    return dim == tensor.ndim - 1


def tp_head_placements(input_dtensor: DTensor) -> tuple:
    placements = list(input_dtensor.placements)
    tp_axis = tp_mesh_axis(input_dtensor)
    if tp_axis is None:
        raise RuntimeError("AllGatherFusedQKVLinear requires a named TP mesh axis")
    placements[tp_axis] = Shard(2)
    return tuple(placements)


def tp_sequence_placements(input_dtensor: DTensor) -> tuple:
    placements = list(input_dtensor.placements)
    tp_axis = tp_mesh_axis(input_dtensor)
    if tp_axis is None:
        raise RuntimeError("ReduceScatterLinear requires a named TP mesh axis")
    placements[tp_axis] = Shard(1)
    return tuple(placements)


class AllGatherFusedQKVLinear(FusedQKVLinear):
    """Fused QKV projection whose forward all-gathers the TP sequence shard."""

    @dataclass(kw_only=True, slots=True)
    class Config(FusedQKVLinear.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.tp_group: dist.ProcessGroup | None = None

    def parallelize(self, parallel_dims: "ParallelDims") -> None:
        tp_mesh = parallel_dims.get_optional_mesh("tp")
        if tp_mesh is not None:
            self.tp_group = tp_mesh.get_group("tp")
        super().parallelize(parallel_dims)

    def forward(  # pyrefly: ignore[bad-override]
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.tp_group is None or not is_tp_sequence_sharded(x):
            return super().forward(x)

        assert isinstance(x, DTensor)
        x_local = x.to_local()
        bsz, _, dim = x_local.shape
        # The all-gather concatenates whole per-rank blocks, so the rows it
        # produces are ordered (rank, batch, seq_local). Flattening [B, S/W, D]
        # directly would therefore be reinterpreted as (batch, seq) and mix
        # batches together for bsz > 1. Put the sequence outermost first so the
        # gathered rows really are (seq, batch) row-major.
        reserve_for_input(self.tp_group, x_local, self.wqkv.weight.shape[0])
        x_seq_major = x_local.transpose(0, 1).reshape(-1, dim).contiguous()
        qkv_flat = AllGatherLinear.apply(
            x_seq_major,
            to_local(self.wqkv.weight),
            to_local(self.wqkv.bias) if self.wqkv.bias is not None else None,
            self.tp_group,
            self.tp_group.group_name,
        )

        full_seqlen = qkv_flat.shape[0] // bsz
        qkv = qkv_flat.view(
            full_seqlen,
            bsz,
            -1,
            self.r_dim,
            self.head_dim,
        ).transpose(0, 1)
        xq, xk, xv = torch.split(qkv, [self.heads_per_kv, 1, 1], dim=-2)
        xq = xq.reshape(bsz, full_seqlen, -1, self.head_dim).contiguous()
        xk = xk.reshape(bsz, full_seqlen, -1, self.head_dim).contiguous()
        xv = xv.reshape(bsz, full_seqlen, -1, self.head_dim).contiguous()

        placements = tp_head_placements(x)
        return (
            DTensor.from_local(xq, x.device_mesh, placements, run_check=False),
            DTensor.from_local(xk, x.device_mesh, placements, run_check=False),
            DTensor.from_local(xv, x.device_mesh, placements, run_check=False),
        )


class ReduceScatterLinear(Linear):
    """Rowwise linear whose forward performs matmul + reduce-scatter."""

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.tp_group: dist.ProcessGroup | None = None

    def parallelize(self, parallel_dims: "ParallelDims") -> None:
        tp_mesh = parallel_dims.get_optional_mesh("tp")
        if tp_mesh is not None:
            self.tp_group = tp_mesh.get_group("tp")
        super().parallelize(parallel_dims)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.tp_group is None or not is_tp_feature_sharded(input):
            return super().forward(input)

        assert isinstance(input, DTensor)
        input_local = input.to_local()
        bsz, seqlen, k_local = input_local.shape
        world_size = self.tp_group.size()
        # This rank receives seqlen // world_size sequence positions back.
        reserve_for_input(
            self.tp_group,
            input_local,
            self.weight.shape[0],
            tokens_per_rank=(bsz * seqlen) // world_size,
        )
        # Reduce-scatter splits the flattened rows, so put the sequence outermost
        # first or the split would cut across batches instead of the sequence.
        # Feeding 2D with scatter_dim=0 is also what lets the operator take its
        # fused schedules, which it declines for a 3D input.
        x_seq_major = input_local.transpose(0, 1).reshape(-1, k_local).contiguous()
        y_flat = LinearReduceScatter.apply(
            x_seq_major,
            to_local(self.weight),
            to_local(self.bias) if self.bias is not None else None,
            self.tp_group,
            self.tp_group.group_name,
        )
        y_local = (
            y_flat.view(seqlen // world_size, bsz, -1).transpose(0, 1).contiguous()
        )
        return DTensor.from_local(
            y_local,
            input.device_mesh,
            tp_sequence_placements(input),
            run_check=False,
        )


class DistGemmGQAttention(GQAttention):
    """Stock GQA attention wired to distributed QKV and WO projections."""

    @dataclass(kw_only=True, slots=True)
    class Config(GQAttention.Config):
        qkv_linear: AllGatherFusedQKVLinear.Config
        wo: ReduceScatterLinear.Config

    def parallelize(self, parallel_dims: "ParallelDims") -> None:
        # AllGatherFusedQKVLinear owns the attention input all-gather explicitly.
        self._sharding_config = None
        super().parallelize(parallel_dims)


@override(
    target=FusedQKVLinear.Config,
    description="Use symm_mem fused all-gather matmul for fused-QKV projection.",
    exact=True,
)
def all_gather_fused_qkv(cfg: FusedQKVLinear.Config) -> AllGatherFusedQKVLinear.Config:
    return derive(cfg, AllGatherFusedQKVLinear.Config)


@override(
    target=Linear.Config,
    description="Use symm_mem fused matmul reduce-scatter for rowwise linear.",
    exact=True,
)
def reduce_scatter_linear(cfg: Linear.Config) -> ReduceScatterLinear.Config:
    base = cfg.sharding_config
    state_shardings = base.state_shardings if base is not None else {}
    return derive(
        cfg,
        ReduceScatterLinear.Config,
        sharding_config=ShardingConfig(state_shardings=state_shardings),
    )


@override(
    target=GQAttention.Config,
    description="Use distributed QKV and WO projections inside GQA attention.",
    exact=True,
)
def dist_gemm_attention(cfg: GQAttention.Config) -> DistGemmGQAttention.Config:
    if not isinstance(cfg.qkv_linear, FusedQKVLinear.Config):
        raise TypeError(
            "dist_gemm_attention requires GQAttention.qkv_linear to be "
            f"FusedQKVLinear.Config, got {type(cfg.qkv_linear).__name__}"
        )
    return derive(
        cfg,
        DistGemmGQAttention.Config,
        sharding_config=None,
        qkv_linear=all_gather_fused_qkv(cfg.qkv_linear),
        wo=reduce_scatter_linear(cfg.wo),
    )


__all__ = [
    "AllGatherFusedQKVLinear",
    "DistGemmGQAttention",
    "ReduceScatterLinear",
    "all_gather_fused_qkv",
    "dist_gemm_attention",
    "reduce_scatter_linear",
]
