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

Both assume they are the only symmetric-memory op in flight on their group. Each
op brackets itself with barriers, so ranks cannot run ahead of each other, but
every op carves its buffers from offset 0 of the one workspace the group shares.
Issuing two of them concurrently on separate streams would alias those bytes.
Sequential module forwards and autograd backward are single-stream and therefore
safe; deliberate overlap would need distinct workspace offsets, not a barrier
here.
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
    """Size the symmetric-memory workspace for one layer, before it ever runs.

    There is one workspace per process group for the whole process, grown
    monotonically to the largest size any layer has asked for, so the cost is a
    max over layers rather than a sum and nothing is freed between layers. What
    hurts is *when* it grows. Growth re-rendezvouses, which is a collective and
    is rejected outright during CUDA graph capture. Worse, growth frees the old
    buffer while its address stays baked into any graph already captured against
    it, so a later growth turns an earlier graph into a use-after-free on replay.

    Calling this for every layer at parallelize time is what makes that safe: the
    workspace reaches its final size before any layer runs, so it can never grow
    behind a captured graph's back. Doing it from ``forward`` instead would only
    ever be as good as warmup having already covered every layer.

    ``features`` and ``tokens_per_rank`` are the local, post-sharding values the
    collective will see. The reduce-scatter schedule is the larger consumer,
    asking for twice the output chunk, so that is what is reserved.
    """
    symm_mem = ensure_symm_mem_ops()

    min_size = (
        2 * tokens_per_rank * features * torch.empty(0, dtype=dtype).element_size()
    )
    key = (group.group_name, min_size)
    if key in RESERVED_WORKSPACES:
        return
    symm_mem.get_symm_mem_workspace(group.group_name, min_size=min_size)
    RESERVED_WORKSPACES.add(key)


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
    that slice along K. Transposing it first is what makes that cheap: K is
    dim 1 of the slice but dim 0 of ``[K / W, M]``, so the gather stays on
    dim 0, the only dim the fused all-gather-matmul can consume in place.
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
        # Keep only a K-shard of the gathered x for wgrad: same memory as the
        # input, and backward all-gathers it back along K.
        x_shard_k = torch.chunk(x_full, world_size, dim=1)[rank].contiguous()

        ctx.save_for_backward(x_shard_k, w_shard_n)
        ctx.group_name = group_name
        ctx.has_bias = bias_shard_n is not None
        return y_shard_n

    @staticmethod
    def backward(ctx, grad_y_shard_n: torch.Tensor):
        # dgrad and wgrad are independent, and each is a fused collective+matmul
        # in its own right: dgrad is the dual of the forward gather (matmul then
        # reduce-scatter back to a sequence shard), wgrad gathers the saved
        # K-shard of x instead of the sequence.
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
        # return_A=False matters beyond saving the copy: asking for the gathered
        # tensor back disqualifies the op's multimem fast path, which is the only
        # path that beats an unfused all-gather + mm at small token counts.
        _, grad_w_outputs = torch.ops.symm_mem.fused_all_gather_matmul(
            x_shard_k.T.contiguous(),
            [grad_y_shard_n],
            0,
            ctx.group_name,
            return_A=False,
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
                                  the local matmul is a partial sum over K; the
                                  reduce-scatter completes it and shards over M
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
        # One gather serves both grads: the fused all-gather-matmul returns the
        # full dy alongside dgrad, so wgrad is a plain local matmul on it. dbias
        # is the only term needing a second collective, since each rank's dy
        # shard covers just its slice of the sequence.
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

        # wgrad sums over every token, so any leading batch dims fold into M.
        # torch.mm is strictly 2D; the flatten is a no-op when M is already one
        # dim, which is the shape callers actually pass.
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
