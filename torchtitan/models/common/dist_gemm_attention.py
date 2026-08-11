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
around the collective.

:class:`DistGemmGQAttention` wires these blocks into :class:`GQAttention` and
removes the parent attention-boundary all-gather. It is selected by passing
``gemm_backend="dist_gemm"`` to ``make_gqa_config``; see
``torchtitan/models/common/config_utils.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

from torchtitan.distributed.dist_linear import (
    AllGatherLinear,
    LinearReduceScatter,
    reserve_symm_mem_workspace,
)
from torchtitan.models.common.attention import FusedQKVLinear, GQAttention
from torchtitan.models.common.linear import Linear

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
) -> None:
    """Reserve this layer's share of the workspace, if the token count is known.

    ``tokens_per_rank`` is None when the model config was never updated from a
    runtime config (inference-only callers, unit tests). Reserving is then simply
    skipped and the ops size the workspace lazily on first use, as they always
    have -- correct, just without the graph-capture guarantee.

    Sized as float32 rather than the layer's actual dtype. Only under-reserving is
    dangerous -- it puts growth back inside graph capture -- and the collective
    runs in ``training.mixed_precision_param``, which is independent of the dtype
    the weight happens to hold here. Both are ``bfloat16 | float32``, so float32
    is an upper bound over every combination and needs nothing plumbed in.
    """
    if tokens_per_rank is None:
        return
    reserve_symm_mem_workspace(
        tp_group,
        tokens_per_rank=tokens_per_rank // tp_group.size(),
        features=max(in_features, out_features),
        dtype=torch.float32,
    )


def maybe_update_dist_gemm_config(model_config: object, config: object) -> None:
    """Fill dist-GEMM attention configs from the runtime config.

    The reservation needs the token count per step, which only the runtime config
    knows, so it is stamped onto the module configs here and consumed later in
    ``parallelize``. Mirrors ``update_ep_token_dispatcher_config``.
    """
    cfgs = []
    for layer_cfg in getattr(model_config, "layers", []):
        attn_cfg = getattr(layer_cfg, "attention", None)
        if isinstance(attn_cfg, DistGemmGQAttention.Config):
            cfgs.extend((attn_cfg.qkv_linear, attn_cfg.wo))
    if not cfgs:
        return

    from torchtitan.trainer import Trainer

    if not isinstance(config, Trainer.Config):
        # Inference-only callers have no fixed token count per step.
        return

    tokens_per_rank = config.training.local_batch_size * config.training.seq_len
    for cfg in cfgs:
        cfg.tokens_per_rank = tokens_per_rank


class AllGatherFusedQKVLinear(FusedQKVLinear):
    """Fused QKV projection whose forward all-gathers the TP sequence shard."""

    @dataclass(kw_only=True, slots=True)
    class Config(FusedQKVLinear.Config):
        # The workspace reservation needs the token count, and it is the one
        # dimension the module cannot recover on its own: K and N come off the
        # weight, but M is batch x seq_len, known only to the runtime config,
        # which `parallelize` is not handed. So `maybe_update_dist_gemm_config`
        # stamps it here at config-construction time, where both are in scope,
        # and `parallelize` reads it back off the built module.
        tokens_per_rank: int | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.tp_group: dist.ProcessGroup | None = None
        self.tokens_per_rank = config.tokens_per_rank

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
        # See AllGatherFusedQKVLinear.Config: M is the one dimension the module
        # cannot recover from its own weight, so the runtime config stamps it here.
        tokens_per_rank: int | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.tp_group: dist.ProcessGroup | None = None
        self.tokens_per_rank = config.tokens_per_rank

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

    # No parallelize override: both contracts this block needs -- no
    # attention-boundary all-gather, and a wo that emits its final Shard(1)
    # rather than a Partial -- are declared up front by
    # ``set_gqa_attention_sharding``.


__all__ = [
    "AllGatherFusedQKVLinear",
    "AttentionOutputLinear",
    "DistGemmGQAttention",
    "maybe_update_dist_gemm_config",
]
