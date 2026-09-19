# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model components that fold the TP collectives into their GEMMs.

:class:`AsyncColumnParallelLinear` and :class:`AsyncRowParallelLinear` are drop-in
replacements for stock linear projections. They move the TP collective into
the GEMM through the autograd Functions colocated in this module.
``AsyncRowParallelLinear`` serves both attention's ``wo`` and the FFN's ``w2``;
``AsyncColumnParallelLinear`` serves both attention's ``wqkv`` and the FFN's
``w13``. Nothing about the primitives is attention-specific, and MoE
projections could use the same pair.

Selected by passing ``tp_gemm_backend="dist_gemm"`` to ``make_gqa_config`` or
``make_ffn_config`` (see ``config_utils.py``), which also drops the boundary
all-gather these modules take over.

The operations assume they are the only symmetric-memory operation in
flight on their process group. Each operation uses barriers, but concurrent
operations on separate streams would still alias the shared workspace.
"""

from __future__ import annotations

import logging

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.distributed as dist

from torchtitan.distributed.spmd_types import current_spmd_mesh
from torchtitan.models.common.linear import Linear


logger = logging.getLogger(__name__)

# Shape suffix legend:
#   T = token dimensions, F = feed-forward hidden dimension


_WARNED_NO_TP = False


def ensure_symm_mem_ops():
    """Import the symmetric-memory module and return it.

    ``torch.ops.symm_mem.*`` is registered as a side effect of this import, so
    anything reaching for those ops has to do it first.
    """
    import torch.distributed._symmetric_memory as symm_mem

    return symm_mem


class AsyncAllGatherLinear(torch.autograd.Function):
    """All-gather the sequence shard, then apply a column-parallel linear.

    Over ``R`` ranks, with ``M`` rows of sequence-major tokens:

        x_shard_m  [M / R, K]   this rank's slice of the sequence
        w_shard_n  [N / R, K]   weight sharded over its output features
        y_shard_n  [M, N / R]   full sequence, features still sharded
        x_shard_k  [M, K / R]   full sequence, features sharded; the slice of the
                                gathered x that forward saves for wgrad

        forward    y_shard_n  = all_gather(x_shard_m) [M, K] @ w_shard_n.T
        dgrad      dx_shard_m = reduce_scatter(dy_shard_n @ w_shard_n)
                                the dual of the forward gather
        wgrad      dw_shard_n = (all_gather(x_shard_k.T) [K, M] @ dy_shard_n).T
                                see below
        dbias         [N / R]  = dy_shard_n.sum(0), already complete because
                                dy_shard_n holds the full sequence

    Saving the gathered ``x`` for wgrad would cost ``R`` times the activation
    memory, so forward saves ``x_shard_k`` instead -- the same number of elements
    the input already had -- and backward re-gathers it along K.

    Why backward transposes it first. wgrad needs ``x`` gathered along K, i.e.
    ``[K, M]`` from a local ``[M, K / R]``. ``fused_all_gather_matmul`` can only
    gather dim 0 of its input (``gather_dim=0`` is the sole case with a fused
    schedule; see ``_fused_all_gather_matmul_impl``, which moves any other
    gather_dim to the front and flattens, i.e. copies). In ``x_shard_k`` the
    sharded axis K is dim 1, so gathering it directly would take that copy path.
    Transposing to ``[K / R, M]`` puts K on dim 0, so the same gather runs on the
    fused path with no pre-copy -- and ``[K, M]`` is the orientation wgrad wants
    anyway.
    """

    @staticmethod
    def spmd_typecheck(
        result: torch.Tensor,
        *,
        x_shard_m: torch.Tensor,
        w_shard_n: torch.Tensor,
        bias_shard_n: torch.Tensor | None,
        group_name: str,
    ) -> None:
        """SPMD type: x S(0)@TP, w S(0)@TP -> y S(1)@TP."""
        spmd.assert_type(x_shard_m, {group_name: spmd.S(0)})
        # Torch stores weight as [N, K], so column-parallel output-feature
        # sharding is dimension 0 of the stored weight.
        spmd.assert_type(w_shard_n, {group_name: spmd.S(0)})
        if bias_shard_n is not None:
            spmd.assert_type(bias_shard_n, {group_name: spmd.S(0)})
        spmd.assert_local_type_like(
            result,
            x_shard_m,
            {group_name: spmd.S(1)},  # pyrefly: ignore [bad-argument-type]
        )

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx,
        x_shard_m: torch.Tensor,
        w_shard_n: torch.Tensor,
        bias_shard_n: torch.Tensor | None,
        group: dist.ProcessGroup,
        group_name: str,
    ) -> torch.Tensor:
        ensure_symm_mem_ops()
        if not x_shard_m.is_contiguous():
            x_shard_m = x_shard_m.contiguous()

        x_full, outputs = torch.ops.symm_mem.fused_all_gather_matmul(
            x_shard_m,
            [w_shard_n.T],
            0,
            group_name,
        )
        y_shard_n = outputs[0]
        if bias_shard_n is not None:
            y_shard_n = y_shard_n + bias_shard_n

        rank = group.rank()
        world_size = group.size()
        # Keep only a K-shard of the gathered x for wgrad: same memory as the
        # input, and backward all-gathers it back along K.
        x_shard_k = torch.chunk(x_full, world_size, dim=1)[rank].contiguous()

        ctx.save_for_backward(x_shard_k, w_shard_n)
        ctx.group_name = group_name
        ctx.has_bias = bias_shard_n is not None
        return y_shard_n

    @staticmethod
    def backward(ctx, grad_y_shard_n: torch.Tensor):  # pyrefly: ignore[bad-override]
        x_shard_k, w_shard_n = ctx.saved_tensors
        if not grad_y_shard_n.is_contiguous():
            grad_y_shard_n = grad_y_shard_n.contiguous()

        grad_x_shard_m = torch.ops.symm_mem.fused_matmul_reduce_scatter(
            grad_y_shard_n,
            w_shard_n,
            "sum",
            0,
            ctx.group_name,
        )

        # Gather X_k.T along K for the weight gradient. Keep return_A enabled:
        # disabling it may select a schedule that reserves the full gathered
        # buffer and uses substantially more symmetric memory for this shape.
        _, grad_w_outputs = torch.ops.symm_mem.fused_all_gather_matmul(
            x_shard_k.T.contiguous(),
            [grad_y_shard_n],
            0,
            ctx.group_name,
        )
        grad_w_shard_n = grad_w_outputs[0].T.contiguous()
        grad_bias = grad_y_shard_n.sum(dim=0) if ctx.has_bias else None
        return grad_x_shard_m, grad_w_shard_n, grad_bias, None, None


class AsyncLinearReduceScatter(torch.autograd.Function):
    """Apply a row-parallel linear, then reduce-scatter over the sequence.

    Over ``R`` ranks:

        x_shard_k  [M, K / R]   full sequence, features sharded
        w_shard_k  [N, K / R]   weight sharded over its input features
        y_shard_m  [M / R, N]   sequence sharded again, features complete

        forward    y_shard_m  = reduce_scatter(x_shard_k @ w_shard_k.T)
        dgrad      dx_shard_k = all_gather(dy_shard_m) @ w_shard_k
        wgrad      dw_shard_k = all_gather(dy_shard_m).T @ x_shard_k
        dbias         [N]      = all_reduce(dy_shard_m.sum(0))
    """

    @staticmethod
    def spmd_typecheck(
        result: torch.Tensor,
        *,
        x_shard_k: torch.Tensor,
        w_shard_k: torch.Tensor,
        bias: torch.Tensor | None,
        group_name: str,
    ) -> None:
        """SPMD type: x S(1)@TP, w S(1)@TP, bias R@TP -> y S(0)@TP."""
        spmd.assert_type(x_shard_k, {group_name: spmd.S(1)})
        # Torch stores weight as [N, K], so row-parallel input-feature sharding
        # is dimension 1 of the stored weight.
        spmd.assert_type(w_shard_k, {group_name: spmd.S(1)})
        if bias is not None:
            spmd.assert_type(bias, {group_name: spmd.R})
        spmd.assert_local_type_like(
            result,
            x_shard_k,
            {group_name: spmd.S(0)},  # pyrefly: ignore [bad-argument-type]
        )

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx,
        x_shard_k: torch.Tensor,
        w_shard_k: torch.Tensor,
        bias: torch.Tensor | None,
        group: dist.ProcessGroup,
        group_name: str,
    ) -> torch.Tensor:
        ensure_symm_mem_ops()
        if not x_shard_k.is_contiguous():
            x_shard_k = x_shard_k.contiguous()

        y_shard_m = torch.ops.symm_mem.fused_matmul_reduce_scatter(
            x_shard_k,
            w_shard_k.T,
            "sum",
            0,
            group_name,
        )
        if bias is not None:
            y_shard_m = y_shard_m + bias

        ctx.save_for_backward(x_shard_k, w_shard_k)
        ctx.group = group
        ctx.group_name = group_name
        ctx.has_bias = bias is not None
        return y_shard_m

    @staticmethod
    def backward(ctx, grad_y_shard_m: torch.Tensor):  # pyrefly: ignore[bad-override]
        x_shard_k, w_shard_k = ctx.saved_tensors
        if not grad_y_shard_m.is_contiguous():
            grad_y_shard_m = grad_y_shard_m.contiguous()

        grad_y, outputs = torch.ops.symm_mem.fused_all_gather_matmul(
            grad_y_shard_m,
            [w_shard_k],
            0,
            ctx.group_name,
        )
        grad_x_shard_k = outputs[0]

        # Wgrad sums over all token dimensions, so fold any leading batch
        # dimensions before the matrix multiplication.
        grad_y_2d = grad_y.flatten(0, -2)
        x_2d = x_shard_k.flatten(0, -2)
        grad_w_shard_k = torch.mm(grad_y_2d.T, x_2d, out_dtype=torch.float32)
        if grad_w_shard_k.dtype != w_shard_k.dtype:
            grad_w_shard_k = grad_w_shard_k.to(dtype=w_shard_k.dtype)

        grad_bias = None
        if ctx.has_bias:
            reduce_dims = tuple(range(grad_y_shard_m.ndim - 1))
            grad_bias = grad_y_shard_m.sum(dim=reduce_dims)
            dist.all_reduce(grad_bias, group=ctx.group)

        return grad_x_shard_k, grad_w_shard_k, grad_bias, None, None


def _warn_once_no_tp_overlap() -> None:
    """Say so when the dist-GEMM modules were selected but TP is not on.

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
            "active; running the standard projection path without collective "
            "overlap."
        )


def _tp_group_from_context() -> dist.ProcessGroup | None:
    """The TP process group from the current spmd_types mesh context, or None.

    Resolved per forward rather than captured at parallelize time. The mesh
    context is only entered inside the trainer's ``train_context``, so it is
    unavailable during ``__init__`` and ``parallelize`` -- and reading it here
    means these modules need no ``parallelize`` override and hold no group state.

    None means "run the stock projection": either no mesh context or TP is degree
    1, in which case there is no collective to fuse.
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
    if not enable_sp:
        raise ValueError(
            "tp_gemm_backend='dist_gemm' requires "
            "parallelism.enable_sequence_parallel; the fused GEMMs replace the SP "
            "all-gather and reduce-scatter, so there is nothing for them to fuse "
            "with SP disabled."
        )


class AsyncColumnParallelLinear(Linear):
    """Column-parallel linear that all-gathers its TP sequence shard."""

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tp_group = _tp_group_from_context()
        if tp_group is None:
            _warn_once_no_tp_overlap()
            return super().forward(input)

        weight, bias = self._flatten_weight_and_bias()
        output = AsyncAllGatherLinear.apply(
            input,
            weight,
            bias,
            tp_group,
            tp_group.group_name,
        )
        return self._unflatten_output(output)


class AsyncRowParallelLinear(Linear):
    """Attention output projection: matmul fused with the TP reduce-scatter.

    Named for the role it fills rather than the collective it performs, so it does
    not read like the :class:`AsyncLinearReduceScatter` autograd Function it
    calls. The class itself is a plain rowwise linear and would work for any
    row-parallel projection; today it is only wired in as ``wo``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        """Same fields as a stock Linear. The subclass exists because it is what
        binds ``Config.build()`` to this module rather than the stock one, so it
        cannot be deleted as empty."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tp_group = _tp_group_from_context()
        if tp_group is None:
            _warn_once_no_tp_overlap()
            return super().forward(input)

        weight, bias = self._flatten_weight_and_bias()
        output = AsyncLinearReduceScatter.apply(
            input,
            weight,
            bias,
            tp_group,
            tp_group.group_name,
        )
        return self._unflatten_output(output)


__all__ = [
    "AsyncAllGatherLinear",
    "AsyncColumnParallelLinear",
    "AsyncLinearReduceScatter",
    "AsyncRowParallelLinear",
    "validate_dist_gemm_preconditions",
]
