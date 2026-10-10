# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Native-order cross entropy from BF16 logits, without a dense FP32 save.

T = local tokens; V = the full vocabulary. The V129280 specialization follows
the native ATen log-softmax's 1024 virtual lanes and four-element load order.

The scratch target vector is 16 KiB at T4096. Only the original BF16 logits,
labels, and two FP32 normalization values per row are saved for backward.
"""

import math

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.extra import libdevice

from .ce_exp import exp as packed_exp


@gluon.jit
def _unpack_four(values):
    layout: gl.constexpr = gl.SliceLayout(1, values.type.layout)
    even, odd = gl.split(values.reshape((1024, 2, 2)))
    first, third = gl.split(even)
    second, fourth = gl.split(odd)
    return (
        gl.convert_layout(first, layout),
        gl.convert_layout(second, layout),
        gl.convert_layout(third, layout),
        gl.convert_layout(fourth, layout),
    )


@gluon.jit
def _native_block_sum(values, NUM_WARPS: gl.constexpr):
    # ATen reduces each 32-lane warp, then reduces its 32 warp totals.
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [NUM_WARPS, 1], [1, 0])
    warp_values = gl.convert_layout(values.reshape((32, 32)), layout)
    warp_totals = gl.sum(warp_values, 1)
    warp_totals = gl.convert_layout(
        warp_totals, gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
    )
    return gl.sum(warp_totals, 0)


@gluon.jit
def _row_stats(
    Logits,
    Labels,
    Stats,
    TargetLogProb,
    TOKENS: gl.constexpr,
    VOCAB: gl.constexpr,
    IGNORE_INDEX: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    UNROLL: gl.constexpr,
):
    row = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [NUM_WARPS, 1], [1, 0])
    lanes = gl.arange(0, 1024, layout=gl.SliceLayout(1, layout))
    vector = gl.arange(0, 4, layout=gl.SliceLayout(0, layout))
    base = Logits + row * VOCAB
    full: gl.constexpr = VOCAB // 4096
    tail: gl.constexpr = full * 4096
    maximum = gl.full(
        (1024,), -3.4028234663852886e38, gl.float32, gl.SliceLayout(1, layout)
    )
    for group in range((full + UNROLL - 1) // UNROLL):
        for item in gl.static_range(UNROLL):
            tile = group * UNROLL + item
            values = gl.load(
                base + tile * 4096 + lanes[:, None] * 4 + vector[None, :],
                tile < full,
                other=-float("inf"),
            ).to(gl.float32)
            first, second, third, fourth = _unpack_four(values)
            maximum = gl.maximum(maximum, first)
            maximum = gl.maximum(maximum, second)
            maximum = gl.maximum(maximum, third)
            maximum = gl.maximum(maximum, fourth)
    for offset in gl.static_range(tail, VOCAB, 1024):
        tail_values = gl.load(
            base + offset + lanes, offset + lanes < VOCAB, other=-float("inf")
        ).to(gl.float32)
        maximum = gl.maximum(maximum, tail_values)
    row_maximum = gl.max(maximum, 0)
    total = gl.full((1024,), 0.0, gl.float32, gl.SliceLayout(1, layout))
    for group in range((full + UNROLL - 1) // UNROLL):
        for item in gl.static_range(UNROLL):
            tile = group * UNROLL + item
            values = gl.load(
                base + tile * 4096 + lanes[:, None] * 4 + vector[None, :],
                tile < full,
                other=0,
            ).to(gl.float32)
            first, second, third, fourth = _unpack_four(values)
            next_total = total + libdevice.exp(first - row_maximum)
            next_total += libdevice.exp(second - row_maximum)
            next_total += libdevice.exp(third - row_maximum)
            next_total += libdevice.exp(fourth - row_maximum)
            total = gl.where(tile < full, next_total, total)
    for offset in gl.static_range(tail, VOCAB, 1024):
        valid = offset + lanes < VOCAB
        tail_values = gl.load(base + offset + lanes, valid, other=0).to(gl.float32)
        total = gl.where(valid, total + libdevice.exp(tail_values - row_maximum), total)
    row_sum = _native_block_sum(total, NUM_WARPS)
    log_sum = libdevice.log(row_sum)
    label = gl.load(Labels + row)
    selected = gl.load(base + label, label != IGNORE_INDEX, other=0).to(gl.float32)
    log_probability = (selected - row_maximum) - log_sum
    gl.store(Stats + row, row_maximum)
    gl.store(Stats + TOKENS + row, log_sum)
    gl.store(TargetLogProb + row, gl.where(label != IGNORE_INDEX, log_probability, 0.0))


@gluon.jit
def _loss_sum(TargetLogProb, Loss, TOKENS: gl.constexpr, THREADS: gl.constexpr):
    # ATen NLL sums T/THREADS entries sequentially per virtual thread, then
    # halves the thread array from high offsets to low offsets.
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    groups = gl.arange(0, THREADS // 32, layout=gl.SliceLayout(1, layout))
    lanes = gl.arange(0, 32, layout=gl.SliceLayout(0, layout))
    threads = groups[:, None] * 32 + lanes[None, :]
    accumulators = gl.full((THREADS // 32, 32), 0.0, gl.float32, layout)
    for first in range(0, TOKENS, THREADS):
        values = gl.load(
            TargetLogProb + first + threads, first + threads < TOKENS, other=0.0
        )
        accumulators -= values
    lane_totals = gl.sum(accumulators, 0)
    lane_totals = gl.convert_layout(lane_totals, gl.BlockedLayout([1], [32], [4], [0]))
    gl.store(Loss, gl.sum(lane_totals, 0))


@triton.jit
def _backward(
    Logits,
    Labels,
    Stats,
    GradLoss,
    GradLogits,
    TOKENS: tl.constexpr,
    VOCAB: tl.constexpr,
    IGNORE_INDEX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(1)
    columns = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    offsets = row * VOCAB + columns
    valid = columns < VOCAB
    logits = tl.load(Logits + offsets, valid, other=0).to(tl.float32)
    maximum = tl.load(Stats + row, row < TOKENS, other=0)
    log_sum = tl.load(Stats + TOKENS + row, row < TOKENS, other=0)
    log_probability = (logits - maximum) - log_sum
    label = tl.load(Labels + row, row < TOKENS, other=IGNORE_INDEX)
    label_is_ignored = label == IGNORE_INDEX
    # Valid class IDs fit int32. Keep the ignore comparison at int64, but avoid
    # promoting every vocabulary column to int64 for the target comparison.
    label_column = tl.where(label_is_ignored, -1, label.to(tl.int32))
    grad_loss = tl.load(GradLoss)
    nll_gradient = tl.where(columns == label_column, -grad_loss, 0.0)
    # Native reduction produces +0 for either sign of a zero upstream.
    row_gradient = tl.where(~label_is_ignored & (grad_loss != 0.0), -grad_loss, 0.0)
    probability = packed_exp(log_probability)
    # This FMA and the final BF16 rounding reproduce ATen's separate operators.
    gradient = tl.fma(-probability, row_gradient, nll_gradient)
    tl.store(GradLogits + offsets, gradient, valid)


def forward(
    logits,
    labels,
    ignore_index=-100,
    *,
    num_warps=16,
    unroll=4,
    maxnreg=None,
    metadata=None,
):
    tokens, vocab = logits.shape
    stats = torch.empty((2, tokens), dtype=torch.float32, device=logits.device)
    selected = torch.empty((tokens,), dtype=torch.float32, device=logits.device)
    loss = torch.empty((), dtype=torch.float32, device=logits.device)
    compiled = _row_stats[(tokens,)](
        logits,
        labels,
        stats,
        selected,
        tokens,
        vocab,
        ignore_index,
        num_warps,
        unroll,
        num_warps=num_warps,
        enable_fp_fusion=True,
        maxnreg=maxnreg,
    )
    threads = min(1024, max(32, 2 ** math.floor(math.log2(max(1, tokens // 16)) + 0.5)))
    _loss_sum[(1,)](
        selected, loss, tokens, threads, num_warps=4, enable_fp_fusion=False
    )
    if metadata is not None:
        metadata.update(
            registers=compiled.n_regs,
            spills=compiled.n_spills,
            shared_bytes=compiled.metadata.shared,
        )
    return loss, stats


def backward(
    logits,
    labels,
    stats,
    grad_loss,
    ignore_index=-100,
    *,
    block=2048,
    num_warps=1,
    maxnreg=None,
    output=None,
    metadata=None,
):
    grad_loss = grad_loss.resolve_neg().resolve_conj()
    if output is None:
        output = torch.empty_like(logits)
    tokens, vocab = logits.shape
    compiled = _backward[(triton.cdiv(vocab, block), tokens)](
        logits,
        labels,
        stats,
        grad_loss,
        output,
        tokens,
        vocab,
        ignore_index,
        block,
        num_warps=num_warps,
        enable_fp_fusion=True,
        maxnreg=maxnreg,
    )
    if metadata is not None:
        metadata.update(
            registers=compiled.n_regs,
            spills=compiled.n_spills,
            shared_bytes=compiled.metadata.shared,
        )
    return output
