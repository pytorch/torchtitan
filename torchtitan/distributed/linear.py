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

import spmd_types as spmd
import torch
import torch.distributed as dist


def ensure_symm_mem_ops():
    """Import the symmetric-memory module and return it.

    ``torch.ops.symm_mem.*`` is registered as a side effect of this import, so
    anything reaching for those ops has to do it first. The import is kept lazy
    because it is CUDA-only.
    """
    import torch.distributed._symmetric_memory as symm_mem

    return symm_mem


class AllGatherLinear(torch.autograd.Function):
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
        """SPMD type: x S(0)@TP, w S(0)@TP -> y S(1)@TP.

        The gather consumes the row shard, so the result is full on rows; the
        weight's output-feature shard survives the GEMM. Non-TP axes pass through
        from x.
        """
        spmd.assert_type(x_shard_m, {group_name: spmd.S(0)})
        # S(0), not S(1), even though this is the column-parallel direction: torch
        # stores the weight as [N, K] while the mental model of the GEMM is [K, N],
        # so sharding the output features N is dim 0 of what is actually stored.
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
        #
        # return_A is left at its default of True and the returned tensor is
        # discarded. Passing False would let the op select
        # _multimem_all_gather_matmul, which reserves the *full* gathered buffer
        # (K * tokens_global) rather than a shard of it -- ranks times more
        # symmetric memory for this one call. Its heuristic is
        # `local_M * group_size <= 2048`, which for this call evaluates to
        # `K <= 2048`: tuned for a gathered token dim, not a gathered K, so it
        # would fire here on a dimension it was not designed around. Keeping both
        # directions on the decomposed schedule also makes forward and backward
        # performance consistent.
        _, grad_w_outputs = torch.ops.symm_mem.fused_all_gather_matmul(
            x_shard_k.T.contiguous(),
            [grad_y_shard_n],
            0,
            ctx.group_name,
        )
        grad_w_shard_n = grad_w_outputs[0].T.contiguous()
        grad_bias = grad_y_shard_n.sum(dim=0) if ctx.has_bias else None
        return grad_x_shard_m, grad_w_shard_n, grad_bias, None, None


class AllGatherLinearMulti(torch.autograd.Function):
    """One all-gather feeding a pair of column-parallel linears on the same input.

    The SwiGLU case: ``w1`` and ``w3`` both consume the same activation, so
    gathering once and running two GEMMs off it halves the collectives versus
    applying :class:`AllGatherLinear` twice.

        x_shard_m   [M / R, K]      this rank's slice of the sequence
        wa, wb      [N / R, K] x 2  weights, each sharded over its out-features
        ya, yb      [M, N / R] x 2  full sequence, features still sharded

        forward     one all-gather, two GEMMs (the op takes a list of ``Bs``)
        dgrad       reduce_scatter(cat(dy, dim=1) @ cat(w, dim=0))
                    the cats express the sum over outputs as a single product, so
                    dgrad stays one collective; they cost one dy-sized and one
                    weight-sized copy, cheaper than a second reduce-scatter
        wgrad       one all_gather of x_shard_k.T feeding both GEMMs, same trick

    Fixed at two weights rather than variadic, even though the underlying op takes
    any number of ``Bs``: Dynamo cannot trace an ``autograd.Function`` whose tensor
    inputs arrive through ``*args``, which breaks ``--compile.enable`` and anything
    else that traces the model. Add a third explicitly if a caller ever needs one.

    No bias: torchtitan's dense FFN builds its projections with ``bias=False``, so
    threading optional per-weight biases through here is not worth it. Callers with
    a bias should use the stock projection.
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx,
        x_shard_m: torch.Tensor,
        wa_shard_n: torch.Tensor,
        wb_shard_n: torch.Tensor,
        group: dist.ProcessGroup,
        group_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ensure_symm_mem_ops()
        if not x_shard_m.is_contiguous():
            x_shard_m = x_shard_m.contiguous()

        x_full, outputs = torch.ops.symm_mem.fused_all_gather_matmul(
            x_shard_m,
            [wa_shard_n.T, wb_shard_n.T],
            0,
            group_name,
        )

        rank = group.rank()
        world_size = group.size()
        # See AllGatherLinear: keep only a K-shard of the gathered x for wgrad.
        x_shard_k = torch.chunk(x_full, world_size, dim=1)[rank].contiguous()

        ctx.save_for_backward(x_shard_k, wa_shard_n, wb_shard_n)
        ctx.group_name = group_name
        return outputs[0], outputs[1]

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx, grad_ya_shard_n: torch.Tensor, grad_yb_shard_n: torch.Tensor
    ):
        x_shard_k, wa_shard_n, wb_shard_n = ctx.saved_tensors
        # Spelled out rather than a genexpr over the pair: Dynamo traces this
        # backward as well as the forward, and it cannot trace a generator.
        if not grad_ya_shard_n.is_contiguous():
            grad_ya_shard_n = grad_ya_shard_n.contiguous()
        if not grad_yb_shard_n.is_contiguous():
            grad_yb_shard_n = grad_yb_shard_n.contiguous()
        grad_ys = [grad_ya_shard_n, grad_yb_shard_n]

        # dx = dya @ wa + dyb @ wb, expressed as one concatenated product so dgrad
        # stays a single collective. The cats cost one dy-sized and one
        # weight-sized copy, cheaper than a second reduce-scatter.
        grad_x_shard_m = torch.ops.symm_mem.fused_matmul_reduce_scatter(
            torch.cat(grad_ys, dim=1),
            torch.cat((wa_shard_n, wb_shard_n), dim=0),
            "sum",
            0,
            ctx.group_name,
        )

        # One gather of x_shard_k.T feeds both wgrads. return_A left at its
        # default; see AllGatherLinear.backward.
        _, grad_w_outputs = torch.ops.symm_mem.fused_all_gather_matmul(
            x_shard_k.T.contiguous(),
            grad_ys,
            0,
            ctx.group_name,
        )
        return (
            grad_x_shard_m,
            grad_w_outputs[0].T.contiguous(),
            grad_w_outputs[1].T.contiguous(),
            None,
            None,
        )

    @staticmethod
    def spmd_typecheck(
        results: tuple[torch.Tensor, torch.Tensor],
        *,
        x_shard_m: torch.Tensor,
        wa_shard_n: torch.Tensor,
        wb_shard_n: torch.Tensor,
        group_name: str,
    ) -> None:
        """SPMD type: x S(0)@TP, both w S(0)@TP -> both y S(1)@TP."""
        spmd.assert_type(x_shard_m, {group_name: spmd.S(0)})
        # S(0) for the column-parallel direction; see AllGatherLinear for why the
        # stored [N, K] layout inverts the dim you would expect.
        spmd.assert_type(wa_shard_n, {group_name: spmd.S(0)})
        spmd.assert_type(wb_shard_n, {group_name: spmd.S(0)})
        for result in results:
            spmd.assert_local_type_like(
                result,
                x_shard_m,
                {group_name: spmd.S(1)},  # pyrefly: ignore [bad-argument-type]
            )


class LinearReduceScatter(torch.autograd.Function):
    """Apply a row-parallel linear, then reduce-scatter over the sequence.

    The mirror image of :class:`AllGatherLinear`:

        x_shard_k  [M, K / R]   full sequence, features sharded
        w_shard_k  [N, K / R]   weight sharded over its input features
        y_shard_m  [M / R, N]   sequence sharded again, features complete

        forward    y_shard_m  = reduce_scatter(x_shard_k @ w_shard_k.T)
                                the local matmul is a partial sum over K; the
                                reduce-scatter completes it and shards over M
        dgrad      dx_shard_k = all_gather(dy_shard_m) [M, N] @ w_shard_k
                                the dual of the forward scatter
        wgrad      dw_shard_k = all_gather(dy_shard_m).T [N, M] @ x_shard_k
                                accumulated in fp32
        dbias         [N]      = dy_shard_m.sum(0) then all-reduce, since each
                                rank only sees its own slice of the sequence

    Callers flatten sequence-major, so scattering dim 0 splits the sequence
    instead of cutting across batches.
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
        """SPMD type: x S(1)@TP, w S(1)@TP, bias R@TP -> y S(0)@TP.

        The local matmul is a partial sum over the sharded K; the reduce-scatter
        completes it and shards rows instead. Non-TP axes pass through from x.
        """
        spmd.assert_type(x_shard_k, {group_name: spmd.S(1)})
        # S(1), the mirror of AllGatherLinear's S(0): torch stores the weight as
        # [N, K] while the mental model of the GEMM is [K, N], so sharding the
        # input features K -- the row-parallel direction -- is dim 1 of what is
        # actually stored.
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
