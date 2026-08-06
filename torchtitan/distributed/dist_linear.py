# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Linear projections that fuse the TP collective into the GEMM.

These are the reusable halves of async tensor parallelism, independent of any
particular model component: a column-parallel linear that all-gathers its
sequence shard first, and a row-parallel linear that reduce-scatters its output.
Attention, FFN and MoE projections can all be built on them.

They are kept out of ``torchtitan/models`` on purpose. Both call
``torch.ops.symm_mem``, which is CUDA-only and not present in every build, so
the symmetric-memory import stays local to the functions that need it rather
than sitting on the import path of every model.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


RESERVED_WORKSPACES: set[tuple[str, int]] = set()


def ensure_symm_mem_ops():
    """Import the symmetric-memory module and return it.

    ``torch.ops.symm_mem.*`` is registered as a side effect of this import, so
    anything reaching for those ops has to do it first. The import is kept lazy
    because it is CUDA-only.
    """
    import torch.distributed._symmetric_memory as symm_mem

    return symm_mem


def reserve_symm_mem_workspace(
    group: dist.ProcessGroup,
    *,
    tokens_per_rank: int,
    features: int,
    dtype: torch.dtype,
) -> None:
    """Size the symmetric-memory workspace up front for a given layer shape.

    The fused all-gather/reduce-scatter ops allocate this workspace lazily and
    grow it whenever a call needs more, and growing it re-rendezvouses, which is
    a collective and is rejected outright during CUDA graph capture. Reserving
    the worst case before the first step keeps growth off the hot path and makes
    capture safe by construction rather than by relying on warmup having
    happened to touch every shape.

    The reduce-scatter schedule is the larger consumer, asking for twice the
    output chunk, so that is what is reserved here.
    """
    symm_mem = ensure_symm_mem_ops()

    min_size = (
        2 * tokens_per_rank * features * torch.empty((), dtype=dtype).element_size()
    )
    key = (group.group_name, min_size)
    if key in RESERVED_WORKSPACES:
        return
    symm_mem.get_symm_mem_workspace(group.group_name, min_size=min_size)
    RESERVED_WORKSPACES.add(key)


def reserve_for_input(
    group: dist.ProcessGroup,
    x_local: torch.Tensor,
    out_features: int,
    *,
    tokens_per_rank: int | None = None,
) -> None:
    if tokens_per_rank is None:
        tokens_per_rank = x_local.numel() // x_local.shape[-1]
    reserve_symm_mem_workspace(
        group,
        tokens_per_rank=tokens_per_rank,
        features=max(x_local.shape[-1], out_features),
        dtype=x_local.dtype,
    )


def mm_with_optional_fp32_out(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    try:
        return torch.mm(a, b, out_dtype=torch.float32)
    except TypeError:
        return torch.mm(a.float(), b.float())


class AllGatherLinear(torch.autograd.Function):
    """All-gather the sequence shard, then apply a column-parallel linear.

    Over ``W`` ranks, with ``M`` rows of sequence-major tokens:

        x_shard_m  [M / W, K]   this rank's slice of the sequence
        w_shard_n  [N / W, K]   weight sharded over its output features
        y_shard_n  [M, N / W]   full sequence, features still sharded

        forward   y  [M, N / W] = all_gather(x) [M, K] @ w.T [K, N / W]
        dgrad     dx [M / W, K] = reduce_scatter(dy [M, N / W] @ w [N / W, K])
                                  the dual of the forward gather
        wgrad     dw [N / W, K] = (all_gather(x_k.T) [K, M] @ dy [M, N / W]).T
                                  see below
        dbias        [N / W]    = dy.sum(0), already complete because dy holds
                                  the full sequence

    The weight gradient needs the gathered ``x``, but holding that until
    backward would cost ``W`` times the activation memory. Forward instead saves
    the K-sharded slice of it, ``[M, K / W]``, which is the same size as the
    input this rank already had, and backward rebuilds ``x.T`` by all-gathering
    that slice along K.
    """

    @staticmethod
    def forward(
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
        x_shard_k = torch.chunk(x_full, world_size, dim=1)[rank].contiguous()

        ctx.save_for_backward(x_shard_k, w_shard_n)
        ctx.group_name = group_name
        ctx.has_bias = bias_shard_n is not None
        return y_shard_n

    @staticmethod
    def backward(ctx, grad_y_shard_n: torch.Tensor):
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

        # AG(X_k.T) @ dY produces dW.T. This mirrors the usual AG-linear wgrad
        # dual without depending on a higher-level distributed-linear package.
        _, grad_w_outputs = torch.ops.symm_mem.fused_all_gather_matmul(
            x_shard_k.T.contiguous(),
            [grad_y_shard_n],
            0,
            ctx.group_name,
        )
        grad_w_shard_n = grad_w_outputs[0].T.contiguous()
        grad_bias = grad_y_shard_n.sum(dim=0) if ctx.has_bias else None
        return grad_x_shard_m, grad_w_shard_n, grad_bias, None, None


class LinearReduceScatter(torch.autograd.Function):
    """Apply a row-parallel linear, then reduce-scatter over the sequence.

    The mirror image of :class:`AllGatherLinear`:

        x_shard_k  [M, K / W]   full sequence, features sharded
        w_shard_k  [N, K / W]   weight sharded over its input features
        y_shard_m  [M / W, N]   sequence sharded again, features complete

        forward   y  [M / W, N] = reduce_scatter(x [M, K / W] @ w.T [K / W, N])
                                  every rank holds a partial sum
        dgrad     dx [M, K / W] = all_gather(dy) [M, N] @ w [N, K / W]
                                  the dual of the forward scatter
        wgrad     dw [N, K / W] = all_gather(dy).T [N, M] @ x [M, K / W]
                                  accumulated in fp32
        dbias        [N]        = dy.sum(0) then all-reduce, since each rank
                                  only sees its own slice of the sequence

    Callers flatten sequence-major, so scattering dim 0 splits the sequence
    instead of cutting across batches.
    """

    @staticmethod
    def forward(
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
    def backward(ctx, grad_y_shard_m: torch.Tensor):
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

        grad_y_2d = grad_y.flatten(0, -2)
        x_2d = x_shard_k.flatten(0, -2)
        grad_w_shard_k = mm_with_optional_fp32_out(grad_y_2d.T, x_2d)
        if grad_w_shard_k.dtype != w_shard_k.dtype:
            grad_w_shard_k = grad_w_shard_k.to(dtype=w_shard_k.dtype)

        grad_bias = None
        if ctx.has_bias:
            reduce_dims = tuple(range(grad_y_shard_m.ndim - 1))
            grad_bias = grad_y_shard_m.sum(dim=reduce_dims)
            dist.all_reduce(grad_bias, group=ctx.group)

        return grad_x_shard_k, grad_w_shard_k, grad_bias, None, None
