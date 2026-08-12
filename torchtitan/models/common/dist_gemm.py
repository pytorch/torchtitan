# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Attention components that use the fused TP+SP linear primitives.

:class:`AllGatherFusedQKVLinear` and :class:`RowParallelLinear` are drop-in
replacements for the stock QKV and output projections. They keep the stock
parameter layouts and only move the TP collective into the GEMM, using
:class:`~torchtitan.distributed.dist_linear.AllGatherLinear` and
:class:`~torchtitan.distributed.dist_linear.LinearReduceScatter`.

Those primitives live under ``torchtitan/distributed`` rather than here, because
nothing about them is attention-specific: FFN and MoE projections can use the
same pair. What stays in this module is the wiring -- the QKV-specific reshaping
around the collective.

They are selected by passing ``gemm_backend="dist_gemm"`` to ``make_gqa_config``
(see ``torchtitan/models/common/config_utils.py``), which also drops the parent
attention-boundary all-gather that ``AllGatherFusedQKVLinear`` takes over. No
attention subclass is needed: the stock ``GQAttention`` forward handles a QKV that
changes the sequence length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torchtitan.distributed.dist_linear import (
    AllGatherLinear,
    AllGatherLinearMulti,
    LinearReduceScatter,
    reserve_symm_mem_workspace,
)

from torchtitan.distributed.spmd_types import current_spmd_mesh

from torchtitan.models.common.attention import FusedQKVLinear
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear

if TYPE_CHECKING:
    from torchtitan.config import ParallelismConfig, TrainingConfig
    from torchtitan.distributed.parallel_dims import ParallelDims


def tp_group_from_context() -> dist.ProcessGroup | None:
    """The TP process group from the current spmd_types mesh context, or None.

    Resolved per forward rather than captured at parallelize time. The mesh
    context is only entered inside the trainer's ``train_context``, so it is
    unavailable during ``__init__`` and ``parallelize`` -- and reading it here
    means these modules need no ``parallelize`` override and hold no group state.

    None means "run the stock projection": either no mesh context (non-spmd_types
    caller) or TP is degree 1, in which case there is no collective to fuse.
    """
    mesh = current_spmd_mesh()
    if mesh is None or "tp" not in (mesh.mesh_dim_names or ()):
        return None
    tp_group = mesh.get_group("tp")
    return tp_group if tp_group.size() > 1 else None


def local_param(param: torch.Tensor | None) -> torch.Tensor | None:
    """Unwrap a parameter that a legacy backend may still hand over as a DTensor.

    TODO: delete once full_dtensor and the legacy backend are removed; under
    ``spmd_types`` parameters are already local. Mirrors the same unwrap in
    ``Embedding.forward``.
    """
    return param.to_local() if isinstance(param, DTensor) else param


def reserve_dist_gemm_workspace(
    model: torch.nn.Module,
    parallel_dims: "ParallelDims",
    training: "TrainingConfig",
    parallelism: "ParallelismConfig",
) -> None:
    """Size the symmetric-memory workspace for every dist-GEMM layer, in one call.

    Call this from a model's ``parallelize_fn`` *before* the weights are sharded:
    the dims read here are global, and the workspace has to reach its final size
    before any layer runs. Growing it later re-rendezvouses -- a collective, and
    rejected outright during CUDA graph capture -- and growth also frees the old
    buffer while its address may already be baked into a captured graph, which
    turns an earlier graph into a use-after-free on replay.

    One workspace serves every layer, sized to the widest, so this is a max over
    layers rather than a sum. A no-op unless some layer selected
    ``gemm_backend="dist_gemm"``.

    Sized as float32 rather than the layers' actual dtype. Only under-reserving is
    dangerous, and the collective runs in ``training.mixed_precision_param``, which
    is independent of the dtype the weights hold here. Both are
    ``bfloat16 | float32``, so float32 is an upper bound over every combination.
    """
    qkv_linears = [m for m in model.modules() if isinstance(m, AllGatherFusedQKVLinear)]
    out_linears = [m for m in model.modules() if isinstance(m, RowParallelLinear)]
    if not qkv_linears and not out_linears:
        return

    tp_mesh = parallel_dims.get_optional_mesh("tp")
    if tp_mesh is None:
        # TP disabled: the fused modules fall back to the stock projections and
        # never touch symmetric memory.
        return
    tp_group = tp_mesh.get_group("tp")
    tp_size = tp_group.size()

    # Preconditions. Neither is detectable from inside the modules: under
    # spmd_types an activation is a plain local tensor, so there are no placements
    # to inspect at runtime.
    if parallelism.spmd_backend != "spmd_types":
        raise ValueError(
            "gemm_backend='dist_gemm' requires parallelism.spmd_backend='spmd_types', "
            f"got {parallelism.spmd_backend!r}. The fused modules take and return "
            "plain local tensors; the DTensor backends are being deprecated and are "
            "not supported."
        )
    if not parallelism.enable_sequence_parallel:
        raise ValueError(
            "gemm_backend='dist_gemm' requires parallelism.enable_sequence_parallel; "
            "the fused GEMMs replace the SP all-gather and reduce-scatter, so there "
            "is nothing for them to fuse with SP disabled."
        )

    # Widest (K, N) any layer's fused GEMMs will see once sharded. Weights are
    # still global here, hence the explicit division.
    features = 0
    for qkv in qkv_linears:
        out_f, in_f = qkv.wqkv.weight.shape
        features = max(features, in_f, out_f // tp_size)
    for out_linear in out_linears:
        out_f, in_f = out_linear.weight.shape
        features = max(features, in_f // tp_size, out_f)

    reserve_symm_mem_workspace(
        tp_group,
        tokens_per_rank=(training.local_batch_size * training.seq_len) // tp_size,
        features=features,
        dtype=torch.float32,
    )


class AllGatherFusedQKVLinear(FusedQKVLinear):
    """Fused QKV projection whose forward all-gathers the TP sequence shard."""

    @dataclass(kw_only=True, slots=True)
    class Config(FusedQKVLinear.Config):
        """Same fields as the stock fused QKV. The subclass exists because it is
        what binds ``Config.build()`` to this module rather than the stock one."""

    def forward(  # pyrefly: ignore[bad-override]
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tp_group = tp_group_from_context()
        if tp_group is None:
            return super().forward(x)

        bsz, _, dim = x.shape
        # The all-gather concatenates whole per-rank blocks, so the rows it
        # produces are ordered (rank, batch, seq_local). Flattening [B, S/W, D]
        # directly would therefore be reinterpreted as (batch, seq) and mix
        # batches together for bsz > 1. Put the sequence outermost first so the
        # gathered rows really are (seq, batch) row-major.
        x_seq_major = x.transpose(0, 1).reshape(-1, dim).contiguous()
        qkv_flat = AllGatherLinear.apply(
            x_seq_major,
            local_param(self.wqkv.weight),
            local_param(self.wqkv.bias),
            tp_group,
            tp_group.group_name,
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
        return (
            xq.reshape(bsz, full_seqlen, -1, self.head_dim).contiguous(),
            xk.reshape(bsz, full_seqlen, -1, self.head_dim).contiguous(),
            xv.reshape(bsz, full_seqlen, -1, self.head_dim).contiguous(),
        )


class RowParallelLinear(Linear):
    """Attention output projection: matmul fused with the TP reduce-scatter.

    Named for the role it fills rather than the collective it performs, so it does
    not read like the :class:`LinearReduceScatter` autograd Function it calls. The
    class itself is a plain rowwise linear and would work for any row-parallel
    projection; today it is only wired in as ``wo``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        """Same fields as a stock Linear. The subclass exists because it is what
        binds ``Config.build()`` to this module rather than the stock one."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tp_group = tp_group_from_context()
        if tp_group is None:
            return super().forward(input)

        bsz, seqlen, k_local = input.shape
        world_size = tp_group.size()
        # Reduce-scatter splits the flattened rows, so put the sequence outermost
        # first or the split would cut across batches instead of the sequence.
        # Feeding 2D with scatter_dim=0 is also what lets the operator take its
        # fused schedules, which it declines for a 3D input.
        x_seq_major = input.transpose(0, 1).reshape(-1, k_local).contiguous()
        y_flat = LinearReduceScatter.apply(
            x_seq_major,
            local_param(self.weight),
            local_param(self.bias),
            tp_group,
            tp_group.group_name,
        )
        return y_flat.view(seqlen // world_size, bsz, -1).transpose(0, 1).contiguous()


class AllGatherFusedFeedForward(FeedForward):
    """SwiGLU feed-forward with both TP collectives folded into its GEMMs.

    ``w1`` and ``w3`` share an input, so one all-gather feeds both
    (:class:`AllGatherLinearMulti`); ``w2`` is row-parallel and reduce-scatters
    back to a sequence shard (:class:`LinearReduceScatter`). Parameter layout and
    checkpoint FQNs are the stock ``w1``/``w2``/``w3``.

    Falls back to the stock forward when TP is off, or when ``w1``/``w3`` carry a
    bias: the multi-weight gather takes no per-weight bias (torchtitan's dense FFN
    builds these with ``bias=False``).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        """Same fields as the stock FFN. The subclass exists because it is what
        binds ``Config.build()`` to this module rather than the stock one."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tp_group = tp_group_from_context()
        if tp_group is None or self.w1.bias is not None or self.w3.bias is not None:
            return super().forward(x)

        bsz, _, dim = x.shape
        # Sequence outermost before flattening, so the gathered rows really are
        # (seq, batch) row-major. See AllGatherFusedQKVLinear for why.
        x_seq_major = x.transpose(0, 1).reshape(-1, dim).contiguous()
        h1, h3 = AllGatherLinearMulti.apply(
            x_seq_major,
            tp_group,
            tp_group.group_name,
            local_param(self.w1.weight),
            local_param(self.w3.weight),
        )
        # Elementwise on feature-sharded activations: no collective.
        h = F.silu(h1) * h3
        y_flat = LinearReduceScatter.apply(
            h,
            local_param(self.w2.weight),
            local_param(self.w2.bias),
            tp_group,
            tp_group.group_name,
        )
        # y_flat is [S_local * B, dim], sequence-major. Shape comes from y_flat
        # rather than from x: the collectives change the row count.
        return y_flat.view(-1, bsz, y_flat.shape[-1]).transpose(0, 1).contiguous()


__all__ = [
    "AllGatherFusedFeedForward",
    "AllGatherFusedQKVLinear",
    "RowParallelLinear",
    "reserve_dist_gemm_workspace",
]
