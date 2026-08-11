# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Attention components that use the fused TP+SP linear primitives.

:class:`AllGatherFusedQKVLinear` and :class:`AttentionOutputLinear` are drop-in
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

from collections.abc import Iterator
from dataclasses import dataclass, fields, is_dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

from torchtitan.config import derive, override
from torchtitan.distributed.dist_linear import (
    AllGatherLinear,
    LinearReduceScatter,
    reserve_symm_mem_workspace,
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
        raise RuntimeError("AttentionOutputLinear requires a named TP mesh axis")
    placements[tp_axis] = Shard(1)
    return tuple(placements)


def reserve_for_layer(
    tp_group: dist.ProcessGroup,
    *,
    tokens_per_rank: int | None,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
) -> None:
    """Reserve this layer's share of the workspace, if the token count is known.

    ``tokens_per_rank`` is None when the model config was never updated from a
    runtime config (inference-only callers, unit tests). Reserving is then simply
    skipped and the ops size the workspace lazily on first use, as they always
    have -- correct, just without the graph-capture guarantee.
    """
    if tokens_per_rank is None:
        return
    reserve_symm_mem_workspace(
        tp_group,
        tokens_per_rank=tokens_per_rank // tp_group.size(),
        features=max(in_features, out_features),
        dtype=dtype,
    )


def maybe_update_dist_gemm_config(model_config: object, config: object) -> None:
    """Fill dist-GEMM linear configs from the runtime config.

    The reservation needs the token count per step, which only the runtime config
    knows, so it is stamped onto the module configs here and consumed later in
    ``parallelize``. Mirrors ``maybe_update_minimal_async_ep_config``.
    """
    cfgs = [
        cfg
        for cfg in walk_configs(model_config)
        if isinstance(
            cfg, (AllGatherFusedQKVLinear.Config, AttentionOutputLinear.Config)
        )
    ]
    if not cfgs:
        return

    from torchtitan.config import TORCH_DTYPE_MAP
    from torchtitan.trainer import Trainer

    if not isinstance(config, Trainer.Config):
        # Inference-only callers have no fixed token count per step.
        return

    tokens_per_rank = config.training.local_batch_size * config.training.seq_len
    dtype = TORCH_DTYPE_MAP[config.training.mixed_precision_param]
    for cfg in cfgs:
        cfg.tokens_per_rank = tokens_per_rank
        cfg.param_dtype = dtype


def walk_configs(root: object) -> Iterator[object]:
    """Yield every dataclass reachable from ``root``, itself included."""
    seen: set[int] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if is_dataclass(node) and not isinstance(node, type):
            yield node
            stack.extend(getattr(node, f.name, None) for f in fields(node))
        elif isinstance(node, (list, tuple)):
            stack.extend(node)
        elif isinstance(node, dict):
            stack.extend(node.values())


class AllGatherFusedQKVLinear(FusedQKVLinear):
    """Fused QKV projection whose forward all-gathers the TP sequence shard."""

    @dataclass(kw_only=True, slots=True)
    class Config(FusedQKVLinear.Config):
        # Filled in by maybe_update_dist_gemm_config from the runtime config.
        tokens_per_rank: int | None = None
        param_dtype: torch.dtype | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.tp_group: dist.ProcessGroup | None = None
        self.tokens_per_rank = config.tokens_per_rank
        self.param_dtype = config.param_dtype

    def parallelize(self, parallel_dims: "ParallelDims") -> None:
        tp_mesh = parallel_dims.get_optional_mesh("tp")
        if tp_mesh is not None:
            self.tp_group = tp_mesh.get_group("tp")
            # Before super() shards the weight, so shape[0] is the global size.
            out_features, in_features = self.wqkv.weight.shape
            reserve_for_layer(
                self.tp_group,
                tokens_per_rank=self.tokens_per_rank,
                in_features=in_features,
                out_features=out_features // self.tp_group.size(),
                dtype=self.param_dtype or self.wqkv.weight.dtype,
            )
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


class AttentionOutputLinear(Linear):
    """Attention output projection: matmul fused with the TP reduce-scatter.

    Named for the role it fills rather than the collective it performs, so it does
    not read like the :class:`LinearReduceScatter` autograd Function it calls. The
    class itself is a plain rowwise linear and would work for any row-parallel
    projection; today it is only wired in as ``wo``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        # Filled in by maybe_update_dist_gemm_config from the runtime config.
        tokens_per_rank: int | None = None
        param_dtype: torch.dtype | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.tp_group: dist.ProcessGroup | None = None
        self.tokens_per_rank = config.tokens_per_rank
        self.param_dtype = config.param_dtype

    def parallelize(self, parallel_dims: "ParallelDims") -> None:
        tp_mesh = parallel_dims.get_optional_mesh("tp")
        if tp_mesh is not None:
            self.tp_group = tp_mesh.get_group("tp")
            # Before super() shards the weight, so shape[1] is the global size.
            out_features, in_features = self.weight.shape
            reserve_for_layer(
                self.tp_group,
                tokens_per_rank=self.tokens_per_rank,
                in_features=in_features // self.tp_group.size(),
                out_features=out_features,
                dtype=self.param_dtype or self.weight.dtype,
            )
        super().parallelize(parallel_dims)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.tp_group is None or not is_tp_feature_sharded(input):
            return super().forward(input)

        assert isinstance(input, DTensor)
        input_local = input.to_local()
        bsz, seqlen, k_local = input_local.shape
        world_size = self.tp_group.size()
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
        qkv_linear: AllGatherFusedQKVLinear.Config  # pyrefly: ignore[bad-override]
        wo: AttentionOutputLinear.Config  # pyrefly: ignore[bad-override]

    def parallelize(self, parallel_dims: "ParallelDims") -> None:
        # AllGatherFusedQKVLinear owns the attention input all-gather explicitly.
        self._sharding_config = None

        # Redeclare wo's output contract, which has to happen here rather than in
        # the override factory.
        #
        # A stock rowwise linear produces a Partial sum over its slice of K and
        # lets the framework reduce-scatter it, so `set_gqa_attention_sharding`
        # gives wo a `rowwise_config()`: out_src=Partial, out_dst=Shard(1).
        # AttentionOutputLinear collapses those two steps -- the reduce-scatter
        # happens inside the fused op -- so its forward returns Shard(1) directly
        # and never produces a Partial. Left alone, the module fails its own
        # out_src check with "output DTensor has placements (Shard(dim=1),), but
        # out_src_shardings expects (Partial(sum),)".
        #
        # The override factory does install a corrected config, but the model's
        # sharding setup runs afterwards and overwrites it (last writer wins), so
        # the correction has to be applied here, after super() would have consumed
        # it. Only the parameter shardings are kept: with the output already in its
        # final layout there is nothing left to check or redistribute.
        wo_sharding = getattr(self.wo, "_sharding_config", None)
        if wo_sharding is not None:
            self.wo._sharding_config = ShardingConfig(
                state_shardings=wo_sharding.state_shardings
            )

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
    description="Fuse the TP reduce-scatter into the attention output projection.",
    exact=True,
)
def attention_output_linear(cfg: Linear.Config) -> AttentionOutputLinear.Config:
    base = cfg.sharding_config
    state_shardings = base.state_shardings if base is not None else {}
    return derive(
        cfg,
        AttentionOutputLinear.Config,
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
        wo=attention_output_linear(cfg.wo),
    )


__all__ = [
    "AllGatherFusedQKVLinear",
    "DistGemmGQAttention",
    "AttentionOutputLinear",
    "all_gather_fused_qkv",
    "dist_gemm_attention",
    "maybe_update_dist_gemm_config",
    "attention_output_linear",
]
