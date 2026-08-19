# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model components that fold the TP collectives into their GEMMs.

:class:`AllGatherFusedQKVLinear`, :class:`RowParallelLinear` and
:class:`AllGatherFusedFeedForward` are drop-in replacements for the stock QKV,
output and SwiGLU projections. They keep the stock parameter layouts and only
move the TP collective into the GEMM, over the autograd Functions in
``torchtitan/distributed/linear.py``. ``RowParallelLinear`` serves both attention's ``wo`` and the
FFN's ``w2``; nothing about the primitives is attention-specific, and MoE
projections could use the same pair.

What lives here is the wiring -- the reshaping around each collective, and the
fallbacks -- while ``torchtitan/distributed/linear.py`` holds the collective+GEMM math itself.

Selected by passing ``tp_gemm_backend="dist_gemm"`` to ``make_gqa_config`` or
``make_ffn_config`` (see ``config_utils.py``), which also drops the boundary
all-gather these modules take over. No attention or FFN subclass is needed beyond
the projections: the stock ``GQAttention`` forward handles a QKV that changes the
sequence length.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F

from torchtitan.distributed.linear import (
    AllGatherLinear,
    AllGatherLinearMulti,
    LinearReduceScatter,
)

from torchtitan.distributed.spmd_types import current_spmd_mesh
from torchtitan.distributed.utils import get_spmd_backend

from torchtitan.models.common.attention import FusedQKVLinear
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.observability import tensor_logging
from torchtitan.tools.logging import logger


_WARNED_NO_TP = False


def _warn_once_unfused() -> None:
    """Say so when the fused modules were selected but TP is not on.

    Otherwise the fallback is indistinguishable from the feature working: the run
    succeeds, the loss looks fine, and nothing ran fused. The preconditions cover
    the wrong-backend and SP-disabled cases with hard errors, but TP=1 has to stay
    runnable, so it warns instead.
    """
    global _WARNED_NO_TP
    if not _WARNED_NO_TP:
        _WARNED_NO_TP = True
        logger.warning(
            "tp_gemm_backend='dist_gemm' selected but tensor parallelism is not "
            "active; running the stock projections. Nothing is fused."
        )


def _tp_group_from_context() -> dist.ProcessGroup | None:
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


def validate_dist_gemm_preconditions(*, enable_sp: bool) -> None:
    """Reject configurations the fused modules cannot serve.

    Called from the sharding setup, which is the first point that sees both the
    selected modules and the parallelism settings. Neither condition is detectable
    from inside a module at runtime: under spmd_types an activation is a plain
    local tensor with no placements to inspect.
    """
    backend = get_spmd_backend()
    if backend != "spmd_types":
        raise ValueError(
            "tp_gemm_backend='dist_gemm' requires "
            f"parallelism.spmd_backend='spmd_types', got {backend!r}. The fused "
            "modules take and return plain local tensors; the DTensor backends are "
            "being deprecated and are not supported."
        )
    if not enable_sp:
        raise ValueError(
            "tp_gemm_backend='dist_gemm' requires "
            "parallelism.enable_sequence_parallel; the fused GEMMs replace the SP "
            "all-gather and reduce-scatter, so there is nothing for them to fuse "
            "with SP disabled."
        )


class AllGatherFusedQKVLinear(FusedQKVLinear):
    """Fused QKV projection whose forward all-gathers the TP sequence shard."""

    @dataclass(kw_only=True, slots=True)
    class Config(FusedQKVLinear.Config):
        """Same fields as the stock fused QKV. The subclass exists because it is
        what binds ``Config.build()`` to this module rather than the stock one, so
        it cannot be deleted as empty."""

    def forward(  # pyrefly: ignore[bad-override]
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tp_group = _tp_group_from_context()
        if tp_group is None:
            _warn_once_unfused()
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
            self.wqkv.weight,
            self.wqkv.bias,
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
        binds ``Config.build()`` to this module rather than the stock one, so it
        cannot be deleted as empty."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tp_group = _tp_group_from_context()
        if tp_group is None:
            _warn_once_unfused()
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
            self.weight,
            self.bias,
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
        binds ``Config.build()`` to this module rather than the stock one, so it
        cannot be deleted as empty."""

        def __post_init__(self) -> None:
            # Rejected at config construction rather than falling back silently in
            # forward: a silent fallback means asking for the fused FFN, getting
            # the stock one, and having no way to tell. The multi-weight gather
            # takes no per-weight bias, and torchtitan's dense FFN builds these
            # with bias=False, so this is a misconfiguration rather than a gap.
            if self.w1.bias or self.w3.bias:
                raise ValueError(
                    "AllGatherFusedFeedForward does not support a bias on w1/w3; "
                    "the fused all-gather takes no per-weight bias. Use the stock "
                    "FeedForward, or build w1/w3 with bias=False."
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tp_group = _tp_group_from_context()
        if tp_group is None:
            _warn_once_unfused()
            return super().forward(x)

        bsz, _, dim = x.shape
        # Sequence outermost before flattening, so the gathered rows really are
        # (seq, batch) row-major. See AllGatherFusedQKVLinear for why.
        x_seq_major = x.transpose(0, 1).reshape(-1, dim).contiguous()
        h1, h3 = AllGatherLinearMulti.apply(
            x_seq_major,
            self.w1.weight,
            self.w3.weight,
            tp_group,
            tp_group.group_name,
        )
        # Elementwise on feature-sharded activations: no collective.
        h = F.silu(h1) * h3
        tensor_logging.log_fwd_bwd_stats(self, act_out=h)
        y_flat = LinearReduceScatter.apply(
            h,
            self.w2.weight,
            self.w2.bias,
            tp_group,
            tp_group.group_name,
        )
        # y_flat is [S_local * B, dim], sequence-major. Shape comes from y_flat
        # rather than from x: the collectives change the row count.
        return y_flat.view(-1, bsz, y_flat.shape[-1]).transpose(0, 1).contiguous()


__all__ = [
    "validate_dist_gemm_preconditions",
    "AllGatherFusedFeedForward",
    "AllGatherFusedQKVLinear",
    "RowParallelLinear",
]
