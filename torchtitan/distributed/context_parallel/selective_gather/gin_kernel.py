# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""GIN CuTeDSL selective-gather kernel -- forward and backward (inter-node CP).

The GIN counterpart of the LSA kernels. The nccl4py device API is PUSH-only
(``Gin.put`` + ``Gin.wait_signal``; there is no remote get), so the GIN gather
is the transpose of the LSA pull: each rank PUSHES the blocks its consumers need
into their gathered windows, and waits (via a GIN signal) for the blocks its own
plan names to arrive. This is the device-side version of the ``p2p.py`` send
loop, using ``gin.put`` with a per-put completion signal instead of
``batch_isend_irecv``.

Used only when the CP group spans nodes (``backend == "gin"``); an intra-node CP
group uses the LSA kernels. Per the design, a CP group is never a mix of the two.

Signal model: every put carries ``ncclGinSignalInc`` (op 0, "+1 per put",
accumulate-never-reset; ``ncclGin_SignalSet`` is unsupported in NCCL 2.30.7).
Each producer's put increments the consumer's signal slot 0 by 1, so the consumer
waits until the slot reaches the cumulative count of blocks it expects. Overflow
is handled by ``waitSignal``'s rolling comparison.

Buffer reuse: the wait gates on incoming data arriving, not on this rank's own
puts' sources being drained. For a tight loop, gate shard reuse across steps with
a group barrier (the test does this); a device-side ``flush`` / double-buffering
is the perf follow-up, mirroring the LSA reverse-ack. That is the SOURCE
(shard) side; the DESTINATION (``gathered_buf``) has the same hazard -- a peer's
next-step put must not overwrite blocks this rank is still reading. Training gates
that with the backward's per-step barrier; a forward-only / inference loop must
add its own barrier (or double-buffer ``gathered_buf``) between steps.

Scaling: both kernels use a cooperative launch with one CTA per pushed block
(``grid = num_send + 1``). Cooperative launches co-reside all CTAs, so a very
dense plan on a large CP group can exceed the resident-CTA ceiling
(``cudaErrorCooperativeLaunchTooLarge``). TODO: batch multiple puts per CTA to
lift the ceiling for dense plans.
"""

import cutlass
import cutlass.cute as cute
import nccl.core.device.cute as nccl_cute
import nccl.core.device.cute.barrier as nccl_barrier
import torch
import torch.distributed as dist
from nccl.core.device.cute.types import GinFenceLevel, MemoryOrder

from .lsa_kernel import _compile_reduce, _pick_ctas, _THREADS as _REDUCE_THREADS
from .topology import BlockGatherPlan, GINMetadata

_THREADS = 32  # one warp per CTA; a put/wait is issued cooperatively by the CTA.
_SIGNAL_FWD = 0  # GIN signal slot for the forward gather.
_SIGNAL_BWD = 1  # GIN signal slot for the backward grad push.
_SIGNAL_INC = 0  # ncclGinSignalOp: Inc (+1 per put, arg ignored/accumulates).

# Forward is a pure copy (any 16-byte-alignable dtype). Backward accumulates in
# FP32 registers in the reused LSA reduce, so it needs a float cute type.
_SUPPORTED = {torch.int64, torch.int32, torch.bfloat16, torch.float16, torch.float32}
_CUTE_DT = {
    torch.int64: cutlass.Int64,
    torch.int32: cutlass.Int32,
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
    torch.float32: cutlass.Float32,
}
_REDUCE_DT = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
    torch.float32: cutlass.Float32,
}

_GATHER_COMPILED: dict = {}
_PUSH_COMPILED: dict = {}
_REDUCE_COMPILED: dict = {}
_DRAIN_COMPILED: dict = {}


def _compile_drain(example_args):
    """Device drain: a world GIN barrier with a PUT fence.

    ``GinFenceLevel.PUT`` drains this rank's outstanding puts (on GIN context 0,
    the same context the gather used) before the barrier releases -- so after
    this kernel the shard window is safe to overwrite. Launched as a SEPARATE
    kernel after the gather so the kernel boundary orders all the gather's puts
    before the fence (there is no in-kernel grid barrier). One CTA per rank
    drives the team barrier.
    """

    @cute.kernel
    def _drain_kernel(dev_comm):
        dc = nccl_cute.DevComm(dev_comm)
        gin = dc.gin(nccl_cute.GinBackendMask.ALL, 0)
        coop = nccl_cute.cta()
        sess = nccl_barrier.world_gin(coop, gin, dc, 0)
        sess.sync(coop, MemoryOrder.ACQ_REL, GinFenceLevel.PUT)

    @cute.jit
    def _launch(dev_comm: cutlass.Int64):
        _drain_kernel(dev_comm).launch(
            grid=[1, 1, 1], block=[_THREADS, 1, 1], cooperative=True
        )

    return cute.compile(_launch, *example_args)


def _compile_gather(
    num_send,
    blocks_per_rank,
    blocks_out,
    block_numel,
    block_bytes,
    my_rank,
    cute_dt,
    example_args,
):
    """One cooperative kernel: ``num_send`` put CTAs + one wait CTA.

    CTA ``bidx < num_send`` pushes send-list entry ``bidx`` to its consumer;
    CTA ``bidx == num_send`` waits on this rank's forward signal. Cooperative
    launch co-resides all CTAs so the puts get issued while the wait spins
    (else the wait could starve the producers and deadlock the group).
    """

    @cute.kernel
    def _gather_kernel(
        dev_comm, shard_h, gathered_h, sp_ptr, ssb_ptr, sbat_ptr, expected
    ):
        bidx, _, _ = cute.arch.block_idx()
        dc = nccl_cute.DevComm(dev_comm)
        shard_win = nccl_cute.Window(shard_h)
        gathered_win = nccl_cute.Window(gathered_h)
        team = dc.team_world
        gin = dc.gin(nccl_cute.GinBackendMask.ALL, 0)
        coop = nccl_cute.cta()

        # cutlass.const_expr forces a COMPILE-TIME branch: a plain ``if`` inside a
        # @cute.kernel is rewritten to a traced if-region that traces BOTH sides,
        # so make_layout(num_send) would still hit size 0. const_expr evaluates
        # the Python constant and traces only the taken branch. A rank with no
        # consumers (e.g. the last rank of a causal sliding-window plan) then
        # launches only the wait CTA.
        if cutlass.const_expr(num_send > 0):
            if bidx < num_send:
                sp = cute.make_tensor(
                    cute.make_ptr(cutlass.Int32, sp_ptr), cute.make_layout(num_send)
                )
                ssb = cute.make_tensor(
                    cute.make_ptr(cutlass.Int32, ssb_ptr), cute.make_layout(num_send)
                )
                sbat = cute.make_tensor(
                    cute.make_ptr(cutlass.Int32, sbat_ptr), cute.make_layout(num_send)
                )
                peer = sp[bidx]
                s_blk = ssb[bidx]
                b = sbat[bidx]
                # Int64 block index before byte scaling: buffers can exceed 2 GB.
                src_off = cutlass.Int64(b * blocks_per_rank + s_blk) * block_bytes
                dst_off = (
                    cutlass.Int64(b * blocks_out + my_rank * blocks_per_rank + s_blk)
                    * block_bytes
                )
                src = shard_win.tensor(
                    cute_dt, cute.make_layout(block_numel), offset=src_off
                )
                dst = gathered_win.tensor(
                    cute_dt, cute.make_layout(block_numel), offset=dst_off
                )
                # Push my block to the peer's gathered window; +1 its signal.
                gin.put(
                    team,
                    peer,
                    gathered_win,
                    dst,
                    shard_win,
                    src,
                    coop,
                    is_signal=True,
                    signal_id=_SIGNAL_FWD,
                    signal_op=_SIGNAL_INC,
                    signal_op_arg=1,
                )
            else:
                gin.wait_signal(coop, signal=_SIGNAL_FWD, least=expected)
        else:
            gin.wait_signal(coop, signal=_SIGNAL_FWD, least=expected)

    @cute.jit
    def _launch(
        dev_comm: cutlass.Int64,
        shard_h: cutlass.Int64,
        gathered_h: cutlass.Int64,
        sp_ptr: cutlass.Int64,
        ssb_ptr: cutlass.Int64,
        sbat_ptr: cutlass.Int64,
        expected: cutlass.Int64,
    ):
        _gather_kernel(
            dev_comm, shard_h, gathered_h, sp_ptr, ssb_ptr, sbat_ptr, expected
        ).launch(grid=[num_send + 1, 1, 1], block=[_THREADS, 1, 1], cooperative=True)

    return cute.compile(_launch, *example_args)


def run_gin_gather(
    ctx,
    plan: BlockGatherPlan,
    kv_local: torch.Tensor,
    gin_meta: GINMetadata,
    *,
    copy_own: bool = True,
    drain: bool = True,
    synchronize: bool = True,
) -> torch.Tensor:
    """Push-based selective gather over GIN; fills ``ctx.gathered_buf``.

    ``kv_local`` is this rank's shard (``shard_numel`` elements). Remote blocks
    are pushed to peers while this rank waits for the blocks its plan names to
    arrive. Returns ``ctx.gathered_buf`` (the registered destination window).

    ``drain``: when True (default), a device PUT-fence barrier drains this rank's
    outgoing puts before returning, so the shard window is safe to overwrite next
    step (source-reuse safety). p2p gets this from its ``wait()``; GIN needs it
    explicitly because it pushes from a persistent registered window. Only set
    False if the caller guarantees reuse safety another way (e.g. double-buffering
    or a trailing barrier).

    ``copy_own``: keep True (default) to feed a standard attention kernel -- it
    needs ONE contiguous full-sequence K/V, so this rank's own blocks must be
    materialized in the gathered buffer at their global positions. Set False ONLY
    when the gather is fused into an attention kernel that reads own K/V in place
    from ``kv_local`` (own region left untouched); with a normal attention kernel
    False leaves a hole and is incorrect. To avoid the own-copy while staying
    contiguous, allocate ``kv_local`` as the own-region slice of the gathered
    buffer upstream instead.
    """
    if ctx.dtype not in _SUPPORTED:
        raise NotImplementedError(f"dtype {ctx.dtype} not supported yet.")
    if plan.batch_size != ctx.batch_size:
        raise ValueError(
            f"plan batch {plan.batch_size} != ctx.batch_size {ctx.batch_size}."
        )
    if plan.block_numel != ctx.block_numel:
        raise ValueError(
            f"plan block_numel {plan.block_numel} != ctx.block_numel "
            f"{ctx.block_numel}."
        )
    if not getattr(ctx, "enable_gin", False):
        raise RuntimeError(
            "GIN kernels need a GIN-enabled context (backend='gin', or "
            "enable_gin=True); this one reserved no GIN connections."
        )

    me = ctx.cp_rank
    B, bpr, bn = ctx.batch_size, ctx.blocks_per_rank, ctx.block_numel
    blocks_out = ctx.cp_size * bpr
    itemsize = torch.empty((), dtype=ctx.dtype).element_size()
    block_bytes = bn * itemsize
    cute_dt = _CUTE_DT[ctx.dtype]

    if kv_local.numel() != ctx.shard_numel:
        raise ValueError(
            f"kv_local has {kv_local.numel()} elements, expected "
            f"ctx.shard_numel ({ctx.shard_numel})."
        )
    kv_view = kv_local.reshape(B, bpr, bn)
    shard_view = ctx.shard_buf[: ctx.shard_numel].view(B, bpr, bn)

    # Copy-in: only the blocks this rank pushes need to be staged into the
    # registered window. A contiguous whole-shard copy when every block is sent
    # (full plan); otherwise index just the pushed blocks (a sliding window
    # pushes a few, so this avoids copying the whole shard).
    ci_b, ci_sb = gin_meta.copyin_batch, gin_meta.copyin_src_block
    if ci_b.numel() == B * bpr:
        shard_view.copy_(kv_view)
    elif ci_b.numel() > 0:
        shard_view[ci_b, ci_sb] = kv_view[ci_b, ci_sb]

    # Own blocks: copy from kv into the gathered buffer (disjoint from the
    # incoming puts, which only write remote producers' regions). A contiguous
    # slice when own is the full range (fancy indexing on the whole own region is
    # ~8x slower); skipped entirely when the caller reads own in place.
    if copy_own:
        gathered = ctx.gathered_buf.view(B, blocks_out, bn)
        ob, osb = gin_meta.own_batch, gin_meta.own_src_block
        if ob.numel() == B * bpr:
            gathered[:, me * bpr : (me + 1) * bpr, :] = kv_view
        elif ob.numel() > 0:
            gathered[ob, me * bpr + osb] = kv_view[ob, osb]

    # Cumulative signal threshold: each step adds num_recv incoming puts; the slot
    # is never reset, so wait for the running total.
    ctx._gin_fwd_total = getattr(ctx, "_gin_fwd_total", 0) + gin_meta.num_recv
    expected = ctx._gin_fwd_total

    num_send = int(gin_meta.send_peer.numel())
    sp = gin_meta.send_peer.to(torch.int32).contiguous()
    ssb = gin_meta.send_src_block.to(torch.int32).contiguous()
    sbat = gin_meta.send_batch.to(torch.int32).contiguous()

    key = (num_send, bpr, blocks_out, bn, ctx.dtype, me)
    args = (
        ctx.dev_comm_gpu.data_ptr(),
        ctx.shard_window.handle,
        ctx.gathered_window.handle,
        sp.data_ptr(),
        ssb.data_ptr(),
        sbat.data_ptr(),
        expected,
    )
    if key not in _GATHER_COMPILED:
        _GATHER_COMPILED[key] = _compile_gather(
            num_send, bpr, blocks_out, bn, block_bytes, me, cute_dt, args
        )
    _GATHER_COMPILED[key](*args)

    # Drain outgoing puts (source-reuse safety) via a device PUT-fence barrier,
    # a separate launch so the kernel boundary orders all puts before the fence.
    if drain:
        dc_ptr = ctx.dev_comm_gpu.data_ptr()
        if "drain" not in _DRAIN_COMPILED:
            _DRAIN_COMPILED["drain"] = _compile_drain((dc_ptr,))
        _DRAIN_COMPILED["drain"](dc_ptr)

    if synchronize:
        torch.cuda.synchronize()
    return ctx.gathered_buf


# ---------------------------------------------------------------------------
# Backward: push grads into producer staging slots, then the reused LSA reduce.
# ---------------------------------------------------------------------------
def _compile_push(
    num_send,
    batch_size,
    blocks_per_rank,
    blocks_out,
    block_numel,
    block_bytes,
    cute_dt,
    example_args,
):
    """``num_send`` grad-push CTAs + one wait CTA (transpose of the forward).

    Each push CTA copies one grad block out of ``gathered_win`` (which holds
    d_out) into producer ``peer``'s staging window at this rank's slot, and
    signals the backward slot. The wait CTA waits for this rank's own staging to
    be filled by its consumers.
    """

    @cute.kernel
    def _push_kernel(
        dev_comm, gathered_h, stage_h, gp_ptr, gsb_ptr, gbat_ptr, gslot_ptr, expected
    ):
        bidx, _, _ = cute.arch.block_idx()
        dc = nccl_cute.DevComm(dev_comm)
        gathered_win = nccl_cute.Window(gathered_h)
        stage_win = nccl_cute.Window(stage_h)
        team = dc.team_world
        gin = dc.gin(nccl_cute.GinBackendMask.ALL, 0)
        coop = nccl_cute.cta()

        # Compile-time guard (see the forward kernel): a rank that read nothing
        # remote has no grads to push, so it launches only the wait CTA and the
        # push branch -- with cute.make_layout(num_send) -- is never traced.
        if cutlass.const_expr(num_send > 0):
            if bidx < num_send:
                gp = cute.make_tensor(
                    cute.make_ptr(cutlass.Int32, gp_ptr), cute.make_layout(num_send)
                )
                gsb = cute.make_tensor(
                    cute.make_ptr(cutlass.Int32, gsb_ptr), cute.make_layout(num_send)
                )
                gbat = cute.make_tensor(
                    cute.make_ptr(cutlass.Int32, gbat_ptr), cute.make_layout(num_send)
                )
                gslot = cute.make_tensor(
                    cute.make_ptr(cutlass.Int32, gslot_ptr), cute.make_layout(num_send)
                )
                peer = gp[bidx]
                s_blk = gsb[bidx]
                b = gbat[bidx]
                slot = gslot[bidx]
                # Source: producer 'peer's grad block in d_out (held in gathered).
                src_off = (
                    cutlass.Int64(b * blocks_out + peer * blocks_per_rank + s_blk)
                    * block_bytes
                )
                # Dest: my slot in peer's staging [max_consumers, B, bpr, blk].
                stage_blk = (slot * batch_size + b) * blocks_per_rank + s_blk
                dst_off = cutlass.Int64(stage_blk) * block_bytes
                src = gathered_win.tensor(
                    cute_dt, cute.make_layout(block_numel), offset=src_off
                )
                dst = stage_win.tensor(
                    cute_dt, cute.make_layout(block_numel), offset=dst_off
                )
                gin.put(
                    team,
                    peer,
                    stage_win,
                    dst,
                    gathered_win,
                    src,
                    coop,
                    is_signal=True,
                    signal_id=_SIGNAL_BWD,
                    signal_op=_SIGNAL_INC,
                    signal_op_arg=1,
                )
            else:
                gin.wait_signal(coop, signal=_SIGNAL_BWD, least=expected)
        else:
            gin.wait_signal(coop, signal=_SIGNAL_BWD, least=expected)

    @cute.jit
    def _launch(
        dev_comm: cutlass.Int64,
        gathered_h: cutlass.Int64,
        stage_h: cutlass.Int64,
        gp_ptr: cutlass.Int64,
        gsb_ptr: cutlass.Int64,
        gbat_ptr: cutlass.Int64,
        gslot_ptr: cutlass.Int64,
        expected: cutlass.Int64,
    ):
        _push_kernel(
            dev_comm,
            gathered_h,
            stage_h,
            gp_ptr,
            gsb_ptr,
            gbat_ptr,
            gslot_ptr,
            expected,
        ).launch(grid=[num_send + 1, 1, 1], block=[_THREADS, 1, 1], cooperative=True)

    return cute.compile(_launch, *example_args)


def run_gin_gather_backward(
    ctx,
    plan: BlockGatherPlan,
    d_out: torch.Tensor,
    d_kv_local: torch.Tensor,
    gin_meta: GINMetadata,
    *,
    drain: bool = True,
    synchronize: bool = True,
) -> None:
    """Push-based staging backward over GIN, then the local FP32 reduce.

    ``d_out``: grad of the gathered buffer (``cp_size * shard_numel``).
    ``d_kv_local``: output shard grad (``shard_numel``). Each consumer pushes its
    grad blocks into the producer's staging slots; the producer then reduces its
    own d_out region plus its staging slots in FP32 (the reused LSA reduce kernel).

    ``drain``: when True (default), a device PUT-fence barrier drains the outgoing
    grad puts before the reduce, so ``gathered_buf`` (which held d_out and was the
    push source) is safe to overwrite next call -- the backward analog of the
    forward's source-reuse drain.

    ``synchronize=False`` skips only the trailing sync; the pre-push zero-staging
    protocol still does one host-side barrier (unlike the forward, which stays
    device-side via the drain).
    """
    if ctx.dtype not in _REDUCE_DT:
        raise NotImplementedError(f"backward dtype {ctx.dtype} not supported yet.")
    if plan.batch_size != ctx.batch_size:
        raise ValueError(
            f"plan batch {plan.batch_size} != ctx.batch_size {ctx.batch_size}."
        )
    if plan.block_numel != ctx.block_numel:
        raise ValueError(
            f"plan block_numel {plan.block_numel} != ctx.block_numel "
            f"{ctx.block_numel}."
        )
    if ctx.block_numel % _REDUCE_THREADS != 0:
        raise ValueError(
            f"block_numel ({ctx.block_numel}) must be a multiple of "
            f"{_REDUCE_THREADS}: the reused LSA reduce tiles it across threads "
            "with no tail mask, so a non-multiple would write out of bounds."
        )
    if d_out.numel() != ctx.cp_size * ctx.shard_numel:
        raise ValueError("d_out must have cp_size * shard_numel elements.")
    if d_kv_local.numel() != ctx.shard_numel:
        raise ValueError("d_kv_local must have shard_numel elements.")
    if not getattr(ctx, "enable_gin", False):
        raise RuntimeError(
            "GIN kernels need a GIN-enabled context (backend='gin', or "
            "enable_gin=True); this one reserved no GIN connections."
        )

    me = ctx.cp_rank
    B, bpr, bn = ctx.batch_size, ctx.blocks_per_rank, ctx.block_numel
    blocks_out = ctx.cp_size * bpr
    itemsize = torch.empty((), dtype=ctx.dtype).element_size()
    block_bytes = bn * itemsize
    cute_dt = _REDUCE_DT[ctx.dtype]

    # Stage d_out into the registered gathered window: it is both the push source
    # (peers must read from a window) and the reduce's own-region source. Zero my
    # staging so unwritten consumer slots reduce to 0; a group barrier makes both
    # visible before any consumer pushes into a producer's staging.
    ctx.gathered_buf.copy_(d_out.reshape(-1))
    ctx.grad_stage.zero_()
    torch.cuda.synchronize()
    dist.barrier(group=ctx.pg)

    # Push grads into producers; wait for my consumers' grads (cumulative count =
    # my forward send count, since every block I sent gets a grad pushed back).
    ctx._gin_bwd_total = getattr(ctx, "_gin_bwd_total", 0) + int(
        gin_meta.send_peer.numel()
    )
    expected = ctx._gin_bwd_total

    num_gsend = int(gin_meta.grad_send_peer.numel())
    gp = gin_meta.grad_send_peer.to(torch.int32).contiguous()
    gsb = gin_meta.grad_send_src_block.to(torch.int32).contiguous()
    gbat = gin_meta.grad_send_batch.to(torch.int32).contiguous()
    gslot = gin_meta.grad_send_slot.to(torch.int32).contiguous()
    pkey = (num_gsend, B, bpr, blocks_out, bn, ctx.dtype)
    pargs = (
        ctx.dev_comm_gpu.data_ptr(),
        ctx.gathered_window.handle,
        ctx.grad_stage_window.handle,
        gp.data_ptr(),
        gsb.data_ptr(),
        gbat.data_ptr(),
        gslot.data_ptr(),
        expected,
    )
    if pkey not in _PUSH_COMPILED:
        _PUSH_COMPILED[pkey] = _compile_push(
            num_gsend, B, bpr, blocks_out, bn, block_bytes, cute_dt, pargs
        )
    _PUSH_COMPILED[pkey](*pargs)

    # Drain outgoing grad puts (source-reuse safety for gathered_buf) before the
    # reduce reads it; a separate launch orders the pushes before the fence.
    if drain:
        dc_ptr = ctx.dev_comm_gpu.data_ptr()
        if "drain" not in _DRAIN_COMPILED:
            _DRAIN_COMPILED["drain"] = _compile_drain((dc_ptr,))
        _DRAIN_COMPILED["drain"](dc_ptr)

    # Reduce (reused LSA reduce kernel): own d_out region + staging slots -> d_kv, FP32.
    rcpb = _pick_ctas(B * bpr, bn, _REDUCE_THREADS)
    rkey = (B, bpr, ctx.cp_size, bn, ctx.max_consumers, rcpb, ctx.dtype, me)
    rargs = (
        ctx.gathered_buf.data_ptr(),
        ctx.grad_stage.data_ptr(),
        d_kv_local.data_ptr(),
    )
    if rkey not in _REDUCE_COMPILED:
        _REDUCE_COMPILED[rkey] = _compile_reduce(
            B,
            bpr,
            ctx.cp_size,
            bn,
            ctx.max_consumers,
            _REDUCE_THREADS,
            rcpb,
            me,
            cute_dt,
            rargs,
        )
    _REDUCE_COMPILED[rkey](*rargs)

    if synchronize:
        torch.cuda.synchronize()
