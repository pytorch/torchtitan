# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Grouped-P2P backend for the selective gather (forward + backward).

Uses only ``torch.distributed.batch_isend_irecv`` -- no CuTeDSL, no nccl4py --
so it runs anywhere NCCL/RCCL point-to-point works, including AMD via RCCL.

Deadlock-safety: ``batch_isend_irecv`` coalesces the whole batch (NCCL
``ncclGroupStart/End``), which schedules an arbitrary *matched* pattern without
ordering deadlock. Matching is guaranteed because both sides derive their ops
from the SAME all-gathered plans in the same ``(batch, entry)`` order -- NCCL
matches send/recv per (peer, direction) by post order (it ignores tags), so the
per-pair sequences line up.

Only the plain attributes of ``ctx`` are used (``pg``, ``cp_rank``, ``cp_size``,
``batch_size``, ``blocks_per_rank``, ``block_numel``, ``dtype``), so any object
exposing the same attributes works.

Peers are CP-group-local ranks, passed to ``P2POp`` via ``group_peer=`` (its
positional ``peer`` is a GLOBAL rank), so this is correct for a real CP subgroup,
not only ``pg == WORLD``.
"""

import torch
import torch.distributed as dist

from .topology import PlanMetadata
from .transport import check_ctx_meta


def run_p2p_gather(
    ctx,
    meta: PlanMetadata,
    kv_local: torch.Tensor,
    out: torch.Tensor,
    *,
    copy_own: bool = True,
) -> None:
    """Selective gather via grouped P2P (forward).

    ``kv_local`` (shard_numel) is this rank's shard; ``out`` (cp_size*shard_numel)
    receives the needed blocks at their affine global positions. Remote blocks are
    recv'd DIRECTLY into their destination slice in ``out`` (no temps), and own
    blocks are copied with one contiguous slice when own is the full range.
    ``copy_own=False`` skips the own-copy (transport-only measurement; not a
    usable attention input, since a standard kernel needs own materialized).

    ``meta`` carries every rank's plan as host-side nested lists, so building the
    op lists below stays sync-free -- no device-to-host copy per step. It also
    carries the validation ``build_plan_metadata`` ran, which the entries below
    are turned straight into sends and receives on.
    """
    check_ctx_meta(ctx, meta, kv_local, out)
    if not out.is_contiguous():
        raise ValueError(
            "out must be contiguous: the gather writes through a reshape of it."
        )
    all_src_rank, all_src_block = meta.all_src_rank, meta.all_src_block
    me, cp = ctx.cp_rank, ctx.cp_size
    B, bpr, bn = ctx.batch_size, ctx.blocks_per_rank, ctx.block_numel
    cap = meta.plan.capacity
    kv = kv_local.contiguous().reshape(B, bpr, bn)
    o = out.reshape(B, cp * bpr, bn)
    my_sr = all_src_rank[me]
    my_sb = all_src_block[me]

    ops = []
    # Send the blocks peers name from me, iterating each peer's plan in
    # (batch, entry) order so the per-pair sequences match its recvs.
    for R in range(cp):
        if R == me:
            continue
        rsr = all_src_rank[R]
        rsb = all_src_block[R]
        for b in range(B):
            for e in range(cap):
                if rsr[b][e] == me:
                    ops.append(
                        dist.P2POp(
                            dist.isend, kv[b, rsb[b][e]], group=ctx.pg, group_peer=R
                        )
                    )
    # Recv the remote blocks my plan names DIRECTLY into their destination slice
    # in ``out`` (each o[b, g] is a distinct contiguous slice) -- no recv temps,
    # no placement copy.
    for b in range(B):
        for e in range(cap):
            sr = my_sr[b][e]
            if sr < 0 or sr == me:
                continue
            g = sr * bpr + my_sb[b][e]
            ops.append(dist.P2POp(dist.irecv, o[b, g], group=ctx.pg, group_peer=sr))

    if ops:
        for w in dist.batch_isend_irecv(ops):
            w.wait()

    # Own blocks: one contiguous slice when own is the full range (fancy-index
    # free); else per-block. Skipped for copy_own=False.
    if copy_own:
        own = [
            (b, my_sb[b][e]) for b in range(B) for e in range(cap) if my_sr[b][e] == me
        ]
        if len(own) == B * bpr:
            o[:, me * bpr : (me + 1) * bpr, :] = kv
        else:
            for b, sb in own:
                o[b, me * bpr + sb].copy_(kv[b, sb])


def run_p2p_gather_backward(
    ctx,
    meta: PlanMetadata,
    d_out: torch.Tensor,
    d_kv_local: torch.Tensor,
) -> None:
    """Backward via grouped P2P (transpose of the forward gather).

    Transpose of the forward: this rank sends its grad for each remote block it
    read back to that block's owner, and sums the grads its consumers computed
    for this rank's blocks. Own-attention grad is the contiguous own region of
    ``d_out``.

    Requires the forward to have materialized the own region (``copy_own=True``,
    the default and the only mode the autograd path uses), so the plan has to
    name every own block. ``copy_own=False`` leaves that region unpopulated, so
    it has no meaningful gradient; such metadata is rejected here.

    Relies on the plan's unique-entry invariant, which
    ``build_plan_metadata`` checked: a duplicated block would have its gradient
    sent and summed twice.
    """
    check_ctx_meta(ctx, meta, d_kv_local, d_out)
    if not d_kv_local.is_contiguous():
        raise ValueError(
            "d_kv_local must be contiguous: the reduction writes "
            "through a reshape of it."
        )
    if meta.ranks_missing_own:
        raise ValueError(
            "the backward reads a rank's whole own output region, so every "
            f"rank's plan must name the blocks it owns; ranks "
            f"{list(meta.ranks_missing_own)} do not. A plan built with "
            "include_own=False is forward-only."
        )
    all_src_rank, all_src_block = meta.all_src_rank, meta.all_src_block
    me, cp = ctx.cp_rank, ctx.cp_size
    B, bpr, bn = ctx.batch_size, ctx.blocks_per_rank, ctx.block_numel
    cap = meta.plan.capacity
    do = d_out.contiguous().reshape(B, cp * bpr, bn)
    my_sr = all_src_rank[me]
    my_sb = all_src_block[me]

    # Accumulate in at least float32, so a bf16/fp16 sum does not lose its tail,
    # but never below the input's own precision.
    acc_dtype = torch.promote_types(ctx.dtype, torch.float32)
    acc = do[:, me * bpr : (me + 1) * bpr].to(acc_dtype, copy=True)  # (B, bpr, bn)

    ops = []
    # Send my grad for each remote block I read, to its owner.
    for b in range(B):
        for e in range(cap):
            sr = my_sr[b][e]
            if sr < 0 or sr == me:
                continue
            g = sr * bpr + my_sb[b][e]
            ops.append(
                dist.P2POp(
                    dist.isend, do[b, g].contiguous(), group=ctx.pg, group_peer=sr
                )
            )
    # Recv each consumer's grad for my blocks into a temp; remember its target.
    recv_meta = []  # (temp, batch, block)
    for R in range(cp):
        if R == me:
            continue
        rsr = all_src_rank[R]
        rsb = all_src_block[R]
        for b in range(B):
            for e in range(cap):
                if rsr[b][e] == me:
                    tmp = torch.empty(bn, dtype=ctx.dtype, device=d_out.device)
                    ops.append(dist.P2POp(dist.irecv, tmp, group=ctx.pg, group_peer=R))
                    recv_meta.append((tmp, b, rsb[b][e]))

    if ops:
        for w in dist.batch_isend_irecv(ops):
            w.wait()

    # Reduce: own + every consumer's contribution.
    for tmp, b, sb in recv_meta:
        acc[b, sb] += tmp.to(acc_dtype)
    d_kv_local.reshape(B, bpr, bn).copy_(acc)
