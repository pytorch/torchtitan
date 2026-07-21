# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""LSA CuTeDSL selective-gather kernels -- forward and backward (the shipping pair).

Forward (``run_lsa_gather``): batched (B>=1) signal-pad gather. A cooperative
launch publishes a per-rank readiness epoch; consumers spin on peers' signals
(own blocks skip the wait) and copy each needed block over NVLink with a 128-bit
vectorized copy into a full-sized ``out`` buffer. Double-buffering + a reverse-ack
(a producer waits its consumers ``>= step-1`` before overwriting a half) make
buffer reuse safe for causal plans without a barrier.

Backward (``run_lsa_gather_backward``): the transpose, as a staging-reduce (no
atomics). Each consumer vectorized-copies its grad blocks into its own slot of
the producer's staging buffer; the owner then reduces its own-region of ``d_out``
plus the staging slots in FP32 registers, writing ``d_kv`` directly.

Both are driven by a ``SelectiveGatherContext`` (its registered windows + signal
pad) and a ``BlockGatherPlan``.

Scaling: the forward and backward-push use a cooperative launch with grid
``batch_size * capacity * ctas_per_block`` (the reduce uses ``batch_size *
blocks_per_rank``). Cooperative launches co-reside all CTAs, so a very large
``batch_size * capacity`` (a dense full_plan at scale, or a long-context selective
plan) can exceed the resident-CTA ceiling and fail the launch. TODO: tile several
plan entries per CTA to lift the ceiling for large plans. The kernels also index
registered buffers with int32 offsets; ``SelectiveGatherContext`` fails fast when
a buffer would exceed the ~2GB int32 range (int64 offsets are the follow-up).

Streams: all kernels launch on the caller's current CUDA stream and rely on
stream-ordered execution (the reverse-ack's step-1 threshold, the sync-free
``synchronize=False`` path); a future CUDA-graph capture must preserve that.
"""

import cutlass
import cutlass.cute as cute
import nccl.core.device.cute as nccl_cute
import torch

from .topology import PlanMetadata
from .transport import check_ctx_meta

_THREADS = 256
_TARGET_CTAS = 512
_WORD = cutlass.Int128
_WORD_BYTES = 16
_READY_OFF = 4  # backward signal slot 1 ("staging ready")
_DONE_OFF = 8  # backward signal slot 2 ("grads pushed, done")

# Forward is a pure copy, so any 16-byte-block-aligned dtype works; backward
# accumulates in FP32 registers, so it needs a float cute type per dtype.
_SUPPORTED = {torch.int64, torch.int32, torch.bfloat16, torch.float16, torch.float32}
_CUTE_DT = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
    torch.float32: cutlass.Float32,
}

_WAIT_COMPILED: dict = {}  # forward reverse-ack
_GATHER_COMPILED: dict = {}  # forward gather
_PUSH_COMPILED: dict = {}  # backward push
_DONE_COMPILED = None  # backward stage_done publish (shape-independent; cached once)
_WAITDONE_COMPILED: dict = {}  # backward wait-done
_REDUCE_COMPILED: dict = {}  # backward fused reduce


def _pick_ctas(num_units, per_unit, threads):
    """CTAs per unit of work, targeting ~_TARGET_CTAS total and even division."""
    want = max(1, _TARGET_CTAS // num_units)
    cpb = min(want, max(1, per_unit // threads))
    while cpb > 1 and per_unit % (cpb * threads) != 0:
        cpb -= 1
    return cpb


# ---------------------------------------------------------------------------
# Forward: batched signal-pad gather + reverse-ack (double-buffered).
# ---------------------------------------------------------------------------
def _compile_wait(num_consumers, example_args):
    """Reverse-ack: spin until every consumer published sig_write >= step-1."""
    zero32 = cutlass.Int32(0)
    one32 = cutlass.Int32(1)

    @cute.kernel
    def _wait_kernel(signal_h, dst_ptr, step):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        if bidx == 0:
            if tidx == 0:
                signal_win = nccl_cute.Window(signal_h)
                dst_t = cute.make_tensor(
                    cute.make_ptr(cutlass.Int32, dst_ptr),
                    cute.make_layout(num_consumers),
                )
                thresh = step - one32
                for i in cutlass.range_constexpr(num_consumers):
                    c = dst_t[i]
                    peer = cute.make_ptr(cutlass.Int32, signal_win.lsa_pointer(0, c))
                    val = cute.arch.atomic_add(peer, zero32, sem="acquire", scope="sys")
                    while val < thresh:
                        val = cute.arch.atomic_add(
                            peer, zero32, sem="acquire", scope="sys"
                        )

    @cute.jit
    def _launch(signal_h: cutlass.Int64, dst_ptr: cutlass.Int64, step: cutlass.Int32):
        _wait_kernel(signal_h, dst_ptr, step).launch(grid=[1, 1, 1], block=[32, 1, 1])

    return cute.compile(_launch, *example_args)


def _compile_gather(
    capacity,
    batch_size,
    blocks_per_rank,
    cp_size,
    block_bytes,
    words_per_block,
    shard_half_bytes,
    threads,
    ctas_per_block,
    my_rank,
    example_args,
):
    chunk = words_per_block // ctas_per_block
    total_words = batch_size * cp_size * blocks_per_rank * words_per_block
    blocks_out = cp_size * blocks_per_rank  # global blocks per batch in out
    zero32 = cutlass.Int32(0)
    two32 = cutlass.Int32(2)

    @cute.kernel
    def _gather_kernel(shard_h, signal_h, sr_ptr, sb_ptr, out_ptr, step):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        g = bidx // ctas_per_block
        sub = bidx % ctas_per_block
        b = g // capacity  # batch index
        entry = g % capacity  # plan entry within the batch

        shard_win = nccl_cute.Window(shard_h)
        signal_win = nccl_cute.Window(signal_h)
        half_off = (step % two32) * shard_half_bytes

        if bidx == 0:
            if tidx == 0:
                cute.arch.fence_acq_rel_sys()
                my_sig = cute.make_ptr(cutlass.Int32, signal_win.local_pointer(0))
                cute.arch.atomic_max(my_sig, step, sem="release", scope="sys")

        sr_t = cute.make_tensor(
            cute.make_ptr(cutlass.Int32, sr_ptr),
            cute.make_layout(batch_size * capacity),
        )
        sb_t = cute.make_tensor(
            cute.make_ptr(cutlass.Int32, sb_ptr),
            cute.make_layout(batch_size * capacity),
        )
        idx = b * capacity + entry
        sr = sr_t[idx]
        sb = sb_t[idx]
        if sr >= 0:
            if sr != my_rank:
                if tidx == 0:
                    peer_sig = cute.make_ptr(
                        cutlass.Int32, signal_win.lsa_pointer(0, sr)
                    )
                    val = cute.arch.atomic_add(
                        peer_sig, zero32, sem="acquire", scope="sys"
                    )
                    while val < step:
                        val = cute.arch.atomic_add(
                            peer_sig, zero32, sem="acquire", scope="sys"
                        )
                cute.arch.sync_threads()

            # peer sr's batch-b block sb, in this step's half
            src_off = half_off + (b * blocks_per_rank + sb) * block_bytes
            src_base = shard_win.lsa_pointer(src_off, sr)
            # out: batch b, global block sr*blocks_per_rank + sb
            dst_block = b * blocks_out + sr * blocks_per_rank + sb
            src = cute.make_tensor(
                cute.make_ptr(_WORD, src_base), cute.make_layout(words_per_block)
            )
            out = cute.make_tensor(
                cute.make_ptr(_WORD, out_ptr), cute.make_layout(total_words)
            )
            base = sub * chunk
            for e in cutlass.range(0, chunk, threads):
                out[dst_block * words_per_block + base + e + tidx] = src[
                    base + e + tidx
                ]

    @cute.jit
    def _launch(
        shard_h: cutlass.Int64,
        signal_h: cutlass.Int64,
        sr_ptr: cutlass.Int64,
        sb_ptr: cutlass.Int64,
        out_ptr: cutlass.Int64,
        step: cutlass.Int32,
    ):
        _gather_kernel(shard_h, signal_h, sr_ptr, sb_ptr, out_ptr, step).launch(
            grid=[batch_size * capacity * ctas_per_block, 1, 1],
            block=[threads, 1, 1],
            cooperative=True,
        )

    return cute.compile(_launch, *example_args)


def _check_meta_device(ctx, meta: PlanMetadata) -> None:
    """Reject metadata tensors that do not live on the context's device.

    Their raw pointers go straight to the kernels, and neither
    ``.to(torch.int32)`` nor ``.contiguous()`` moves a tensor between devices,
    so one built on another GPU would be dereferenced there.
    """
    for name, tensor in (
        ("plan.src_rank", meta.plan.src_rank),
        ("plan.src_block", meta.plan.src_block),
        ("dst_slot", meta.dst_slot),
        ("consumers", meta.consumers),
    ):
        if tensor.device != ctx.device:
            raise ValueError(
                f"meta.{name} is on {tensor.device}, not ctx.device {ctx.device}; "
                "the LSA kernels dereference it on the context's device."
            )


def run_lsa_gather(
    ctx,
    meta: PlanMetadata,
    kv_local: torch.Tensor,
    out: torch.Tensor,
    *,
    synchronize: bool = True,
) -> None:
    """Batched (B>=1) reverse-ack + double-buffered forward gather.

    ``kv_local``: this rank's shard, cp-total ``batch_size * blocks_per_rank *
    block_numel``. ``out``: ``batch_size * cp_size * blocks_per_rank *
    block_numel``. ``meta`` carries the validated plan and this rank's consumers.
    """
    check_ctx_meta(ctx, meta, kv_local, out)
    _check_meta_device(ctx, meta)
    if not out.is_contiguous():
        raise ValueError("out must be contiguous: the kernel writes it directly.")
    if ctx.dtype not in _SUPPORTED:
        raise NotImplementedError(f"dtype {ctx.dtype} not supported yet.")
    plan, dst_ranks = meta.plan, meta.consumers

    itemsize = torch.empty((), dtype=ctx.dtype).element_size()
    block_bytes = ctx.block_numel * itemsize
    shard_half_bytes = ctx.shard_numel * itemsize
    if block_bytes % _WORD_BYTES != 0:
        raise ValueError(f"block bytes ({block_bytes}) must be a multiple of 16.")
    words_per_block = block_bytes // _WORD_BYTES
    if words_per_block % _THREADS != 0:
        raise ValueError(
            f"words_per_block ({words_per_block}) must be a multiple of "
            f"{_THREADS} (the CTA thread count): the vectorized copy tiles the "
            "block across threads with no tail mask, so a non-multiple would "
            f"write out of bounds. Require block_bytes % {_WORD_BYTES * _THREADS} "
            f"== 0 (got block_bytes={block_bytes})."
        )
    capacity = plan.capacity
    threads = _THREADS
    cpb = _pick_ctas(ctx.batch_size * capacity, words_per_block, threads)
    step = ctx.next_signal_step()
    half = step % 2

    # Full (B, capacity) plan, flattened row-major to (B*capacity,).
    src_rank = plan.src_rank.to(torch.int32).contiguous()
    src_block = plan.src_block.to(torch.int32).contiguous()

    ncons = int(dst_ranks.numel())
    if ncons > 0:
        dst = dst_ranks.to(torch.int32).contiguous()
        wkey = (ncons, ctx.cp_rank)
        if wkey not in _WAIT_COMPILED:
            _WAIT_COMPILED[wkey] = _compile_wait(
                ncons, (ctx.signal_window.handle, dst.data_ptr(), step)
            )
        _WAIT_COMPILED[wkey](ctx.signal_window.handle, dst.data_ptr(), step)

    ctx.load_shard_half(kv_local, half)

    gkey = (
        capacity,
        ctx.batch_size,
        ctx.blocks_per_rank,
        ctx.cp_size,
        words_per_block,
        ctx.dtype,
        threads,
        cpb,
        ctx.cp_rank,
    )
    if gkey not in _GATHER_COMPILED:
        _GATHER_COMPILED[gkey] = _compile_gather(
            capacity,
            ctx.batch_size,
            ctx.blocks_per_rank,
            ctx.cp_size,
            block_bytes,
            words_per_block,
            shard_half_bytes,
            threads,
            cpb,
            ctx.cp_rank,
            (
                ctx.shard_window.handle,
                ctx.signal_window.handle,
                src_rank.data_ptr(),
                src_block.data_ptr(),
                out.data_ptr(),
                step,
            ),
        )
    _GATHER_COMPILED[gkey](
        ctx.shard_window.handle,
        ctx.signal_window.handle,
        src_rank.data_ptr(),
        src_block.data_ptr(),
        out.data_ptr(),
        step,
    )
    if synchronize:
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Backward: staging-reduce transpose (push into per-consumer slots, FP32 reduce).
# ---------------------------------------------------------------------------
def _compile_push(
    capacity,
    batch_size,
    blocks_per_rank,
    cp_size,
    block_bytes,
    words_per_block,
    threads,
    ctas_per_block,
    my_rank,
    example_args,
):
    """Publish stage_ready, then vectorized-copy grad blocks into producer slots."""
    chunk = words_per_block // ctas_per_block
    blocks_out = cp_size * blocks_per_rank
    total_out = batch_size * blocks_out * words_per_block
    zero32 = cutlass.Int32(0)

    @cute.kernel
    def _push_kernel(signal_h, stage_h, sr_ptr, sb_ptr, slot_ptr, d_out_ptr, step):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        g = bidx // ctas_per_block
        sub = bidx % ctas_per_block
        b = g // capacity
        entry = g % capacity

        signal_win = nccl_cute.Window(signal_h)
        stage_win = nccl_cute.Window(stage_h)

        # Publish my "staging zeroed / ready"; zeroing was a prior torch op.
        if bidx == 0:
            if tidx == 0:
                cute.arch.fence_acq_rel_sys()
                my_ready = cute.make_ptr(
                    cutlass.Int32, signal_win.local_pointer(_READY_OFF)
                )
                cute.arch.atomic_max(my_ready, step, sem="release", scope="sys")

        sr_t = cute.make_tensor(
            cute.make_ptr(cutlass.Int32, sr_ptr),
            cute.make_layout(batch_size * capacity),
        )
        sb_t = cute.make_tensor(
            cute.make_ptr(cutlass.Int32, sb_ptr),
            cute.make_layout(batch_size * capacity),
        )
        slot_t = cute.make_tensor(
            cute.make_ptr(cutlass.Int32, slot_ptr),
            cute.make_layout(batch_size * capacity),
        )
        idx = b * capacity + entry
        sr = sr_t[idx]
        sb = sb_t[idx]
        # Own blocks are handled by the reduce reading d_out directly.
        if sr >= 0:
            if sr != my_rank:
                if tidx == 0:
                    peer_ready = cute.make_ptr(
                        cutlass.Int32, signal_win.lsa_pointer(_READY_OFF, sr)
                    )
                    val = cute.arch.atomic_add(
                        peer_ready, zero32, sem="acquire", scope="sys"
                    )
                    while val < step:
                        val = cute.arch.atomic_add(
                            peer_ready, zero32, sem="acquire", scope="sys"
                        )
                cute.arch.sync_threads()

                slot = slot_t[idx]
                # Producer sr's staging: [max_consumers, B, bpr, block].
                stage_blk = (slot * batch_size + b) * blocks_per_rank + sb
                stg = cute.make_tensor(
                    cute.make_ptr(
                        _WORD, stage_win.lsa_pointer(stage_blk * block_bytes, sr)
                    ),
                    cute.make_layout(words_per_block),
                )
                dst_block = b * blocks_out + sr * blocks_per_rank + sb
                d_out = cute.make_tensor(
                    cute.make_ptr(_WORD, d_out_ptr), cute.make_layout(total_out)
                )
                base = sub * chunk
                for e in cutlass.range(0, chunk, threads):
                    off = base + e + tidx
                    stg[off] = d_out[dst_block * words_per_block + off]

    @cute.jit
    def _launch(
        signal_h: cutlass.Int64,
        stage_h: cutlass.Int64,
        sr_ptr: cutlass.Int64,
        sb_ptr: cutlass.Int64,
        slot_ptr: cutlass.Int64,
        d_out_ptr: cutlass.Int64,
        step: cutlass.Int32,
    ):
        _push_kernel(
            signal_h, stage_h, sr_ptr, sb_ptr, slot_ptr, d_out_ptr, step
        ).launch(
            grid=[batch_size * capacity * ctas_per_block, 1, 1],
            block=[threads, 1, 1],
            cooperative=True,
        )

    return cute.compile(_launch, *example_args)


def _compile_publish_done(example_args):
    @cute.kernel
    def _done_kernel(signal_h, step):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        if bidx == 0:
            if tidx == 0:
                cute.arch.fence_acq_rel_sys()
                win = nccl_cute.Window(signal_h)
                my_done = cute.make_ptr(cutlass.Int32, win.local_pointer(_DONE_OFF))
                cute.arch.atomic_max(my_done, step, sem="release", scope="sys")

    @cute.jit
    def _launch(signal_h: cutlass.Int64, step: cutlass.Int32):
        _done_kernel(signal_h, step).launch(grid=[1, 1, 1], block=[32, 1, 1])

    return cute.compile(_launch, *example_args)


def _compile_wait_done(num_consumers, example_args):
    zero32 = cutlass.Int32(0)

    @cute.kernel
    def _wait_kernel(signal_h, dst_ptr, step):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        if bidx == 0:
            if tidx == 0:
                win = nccl_cute.Window(signal_h)
                dst_t = cute.make_tensor(
                    cute.make_ptr(cutlass.Int32, dst_ptr),
                    cute.make_layout(num_consumers),
                )
                for i in cutlass.range_constexpr(num_consumers):
                    c = dst_t[i]
                    peer = cute.make_ptr(cutlass.Int32, win.lsa_pointer(_DONE_OFF, c))
                    val = cute.arch.atomic_add(peer, zero32, sem="acquire", scope="sys")
                    while val < step:
                        val = cute.arch.atomic_add(
                            peer, zero32, sem="acquire", scope="sys"
                        )

    @cute.jit
    def _launch(signal_h: cutlass.Int64, dst_ptr: cutlass.Int64, step: cutlass.Int32):
        _wait_kernel(signal_h, dst_ptr, step).launch(grid=[1, 1, 1], block=[32, 1, 1])

    return cute.compile(_launch, *example_args)


def _compile_reduce(
    batch_size,
    blocks_per_rank,
    cp_size,
    block_numel,
    max_consumers,
    threads,
    ctas_per_block,
    my_rank,
    cute_dt,
    example_args,
):
    """own d_out + staging slots -> d_kv, FP32 accumulate in registers."""
    chunk = block_numel // ctas_per_block
    blocks_out = cp_size * blocks_per_rank
    total_out = batch_size * blocks_out * block_numel
    total_stage = max_consumers * batch_size * blocks_per_rank * block_numel
    total_kv = batch_size * blocks_per_rank * block_numel

    @cute.kernel
    def _reduce_kernel(d_out_ptr, stage_ptr, dkv_ptr):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        g = bidx // ctas_per_block
        sub = bidx % ctas_per_block
        b = g // blocks_per_rank
        sb = g % blocks_per_rank

        d_out = cute.make_tensor(
            cute.make_ptr(cute_dt, d_out_ptr), cute.make_layout(total_out)
        )
        stage = cute.make_tensor(
            cute.make_ptr(cute_dt, stage_ptr), cute.make_layout(total_stage)
        )
        dkv = cute.make_tensor(
            cute.make_ptr(cute_dt, dkv_ptr), cute.make_layout(total_kv)
        )
        own_block = b * blocks_out + my_rank * blocks_per_rank + sb
        kv_block = b * blocks_per_rank + sb
        base = sub * chunk
        for e in cutlass.range(0, chunk, threads):
            off = base + e + tidx
            acc = cutlass.Float32(d_out[own_block * block_numel + off])
            for s in cutlass.range_constexpr(max_consumers):
                stage_blk = (s * batch_size + b) * blocks_per_rank + sb
                acc += cutlass.Float32(stage[stage_blk * block_numel + off])
            dkv[kv_block * block_numel + off] = cute_dt(acc)

    @cute.jit
    def _launch(
        d_out_ptr: cutlass.Int64, stage_ptr: cutlass.Int64, dkv_ptr: cutlass.Int64
    ):
        _reduce_kernel(d_out_ptr, stage_ptr, dkv_ptr).launch(
            grid=[batch_size * blocks_per_rank * ctas_per_block, 1, 1],
            block=[threads, 1, 1],
        )

    return cute.compile(_launch, *example_args)


def run_lsa_gather_backward(
    ctx,
    meta: PlanMetadata,
    d_out: torch.Tensor,
    d_kv_local: torch.Tensor,
    *,
    synchronize: bool = True,
) -> None:
    """Staging-reduce backward (no atomics).

    Args:
        meta: validated ``PlanMetadata``. Supplies the plan, ``dst_slot`` (the
            slot this rank writes on each entry's producer; 0 for own and
            padding entries), and ``consumers`` (the ranks that read this rank).
        d_out: grad of the gathered buffer, ``cp_size * shard_numel`` (K/V dtype).
        d_kv_local: output shard grad, ``shard_numel`` (K/V dtype).
    """
    check_ctx_meta(ctx, meta, d_kv_local, d_out)
    _check_meta_device(ctx, meta)
    if not d_kv_local.is_contiguous():
        raise ValueError(
            "d_kv_local must be contiguous: the reduce kernel writes it directly."
        )
    if ctx.dtype not in _CUTE_DT:
        raise NotImplementedError(f"dtype {ctx.dtype} not supported yet.")
    plan, dst_slot, dst_ranks = meta.plan, meta.dst_slot, meta.consumers

    B = ctx.batch_size
    bpr = ctx.blocks_per_rank
    bn = ctx.block_numel
    itemsize = torch.empty((), dtype=ctx.dtype).element_size()
    block_bytes = bn * itemsize
    if block_bytes % _WORD_BYTES != 0:
        raise ValueError(f"block bytes ({block_bytes}) must be a multiple of 16.")
    wpb = block_bytes // _WORD_BYTES
    if wpb % _THREADS != 0 or bn % _THREADS != 0:
        raise ValueError(
            f"words_per_block ({wpb}) and block_numel ({bn}) must both be a "
            f"multiple of {_THREADS} (the CTA thread count): the push and reduce "
            "kernels tile across threads with no tail mask, so a non-multiple "
            "would write out of bounds."
        )
    capacity = plan.capacity
    cute_dt = _CUTE_DT[ctx.dtype]
    step = ctx.next_bwd_step()

    src_rank = plan.src_rank.to(torch.int32).contiguous()
    src_block = plan.src_block.to(torch.int32).contiguous()
    slot = dst_slot.to(torch.int32).contiguous()

    # 1. zero my staging so unwritten slots reduce to 0; consumers overwrite.
    ctx.grad_stage.zero_()

    # 2. push: publish my stage_ready + vectorized-copy grads into producers.
    cpb = _pick_ctas(B * capacity, wpb, _THREADS)
    pkey = (capacity, B, bpr, ctx.cp_size, wpb, ctx.dtype, cpb, ctx.cp_rank)
    if pkey not in _PUSH_COMPILED:
        _PUSH_COMPILED[pkey] = _compile_push(
            capacity,
            B,
            bpr,
            ctx.cp_size,
            block_bytes,
            wpb,
            _THREADS,
            cpb,
            ctx.cp_rank,
            (
                ctx.signal_window.handle,
                ctx.grad_stage_window.handle,
                src_rank.data_ptr(),
                src_block.data_ptr(),
                slot.data_ptr(),
                d_out.data_ptr(),
                step,
            ),
        )
    _PUSH_COMPILED[pkey](
        ctx.signal_window.handle,
        ctx.grad_stage_window.handle,
        src_rank.data_ptr(),
        src_block.data_ptr(),
        slot.data_ptr(),
        d_out.data_ptr(),
        step,
    )

    # 3. publish stage_done (boundary: my pushes are visible). The kernel is
    # shape-independent, so it is compiled once and cached module-wide.
    global _DONE_COMPILED
    if _DONE_COMPILED is None:
        _DONE_COMPILED = _compile_publish_done((ctx.signal_window.handle, step))
    _DONE_COMPILED(ctx.signal_window.handle, step)

    # 4. wait for every consumer's stage_done before reducing.
    ncons = int(dst_ranks.numel())
    if ncons > 0:
        dst = dst_ranks.to(torch.int32).contiguous()
        wkey = (ncons, ctx.cp_rank)
        if wkey not in _WAITDONE_COMPILED:
            _WAITDONE_COMPILED[wkey] = _compile_wait_done(
                ncons, (ctx.signal_window.handle, dst.data_ptr(), step)
            )
        _WAITDONE_COMPILED[wkey](ctx.signal_window.handle, dst.data_ptr(), step)

    # 5. fused reduce: own d_out + staging slots -> d_kv (FP32 in registers).
    rcpb = _pick_ctas(B * bpr, bn, _THREADS)
    rkey = (B, bpr, ctx.cp_size, bn, ctx.max_consumers, rcpb, ctx.dtype, ctx.cp_rank)
    if rkey not in _REDUCE_COMPILED:
        _REDUCE_COMPILED[rkey] = _compile_reduce(
            B,
            bpr,
            ctx.cp_size,
            bn,
            ctx.max_consumers,
            _THREADS,
            rcpb,
            ctx.cp_rank,
            cute_dt,
            (d_out.data_ptr(), ctx.grad_stage.data_ptr(), d_kv_local.data_ptr()),
        )
    _REDUCE_COMPILED[rkey](
        d_out.data_ptr(), ctx.grad_stage.data_ptr(), d_kv_local.data_ptr()
    )
    if synchronize:
        torch.cuda.synchronize()
