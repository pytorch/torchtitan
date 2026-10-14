# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""DSv3 MTP cross entropy with compact autograd state and a gated dispatch.

T = local tokens, V = the full vocabulary. MTP weighting, token-count
normalization, label alignment, and TP communication stay in the native loss."""

import logging
import math
from dataclasses import dataclass
from typing import Literal

import spmd_types as spmd
import torch
import triton
import triton.language as tl
from torch._subclasses.fake_tensor import FakeTensor
from torch.autograd.function import once_differentiable
from torchtitan.components.loss import cross_entropy_loss
from torchtitan.config import derive, override
from torchtitan.distributed.spmd_types import spmd_mesh_size
from torchtitan.models.deepseek_v3.mtp import MTPLoss
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.extra import libdevice


EXP_PAIR = tl.constexpr(
    r"""{
    .reg .b64 X, H, C, A, B, R, J, E, P;
    .reg .b32 a0, a1, b0, b1, p0, p1;
    mov.b64 X, {$2, $3};
    mov.b64 H, 0x3F0000003F000000;
    mov.b64 C, 0x3BBB989D3BBB989D;
    fma.rn.ftz.f32x2 R, X, C, H;
    mov.b64 {a0, a1}, R;
    cvt.ftz.sat.f32.f32 a0, a0;
    cvt.ftz.sat.f32.f32 a1, a1;
    fma.rm.ftz.f32 b0, a0, 0f437C0000, 0f4B400001;
    fma.rm.ftz.f32 b1, a1, 0f437C0000, 0f4B400001;
    mov.b64 B, {b0, b1};
    mov.b64 C, 0xCB40007FCB40007F;
    add.f32x2 J, B, C;
    xor.b64 J, J, 0x8000000080000000;
    mov.b64 C, 0x3FB8AA3B3FB8AA3B;
    fma.rn.ftz.f32x2 R, X, C, J;
    mov.b64 C, 0x32A5706032A57060;
    fma.rn.ftz.f32x2 R, X, C, R;
    mov.b64 {a0, a1}, R;
    ex2.approx.ftz.f32 p0, a0;
    ex2.approx.ftz.f32 p1, a1;
    shl.b32 b0, b0, 23;
    shl.b32 b1, b1, 23;
    mov.b64 E, {b0, b1};
    mov.b64 P, {p0, p1};
    mul.f32x2 R, P, E;
    mov.b64 {$0, $1}, R;
}"""
)


@triton.jit
def packed_exp(values):
    return tl.inline_asm_elementwise(
        EXP_PAIR,
        constraints="=f,=f,f,f",
        args=[values],
        dtype=tl.float32,
        is_pure=True,
        pack=2,
    )


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


def _ce_forward(
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


def _ce_backward(
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


# Keep opt-in integrations disabled until all three acceptance gates pass.
ACCEPTED = False
VOCAB_SIZE = 129280
TORCH_VERSION = "2.16.0.dev20261007+cu130"


@torch.compiler.assume_constant_result
def _is_negative_view(tensor: torch.Tensor) -> bool:
    # Dynamo guards the Negative dispatch key as part of TENSOR_MATCH, but
    # cannot represent Tensor.is_neg()'s boolean result in its FX graph.
    return tensor.is_neg()


def supports(logits: torch.Tensor, labels: torch.Tensor) -> bool:
    return (
        type(logits) in (torch.Tensor, FakeTensor)
        and type(labels) in (torch.Tensor, FakeTensor)
        and logits.is_cuda
        and labels.device == logits.device
        and logits.dtype == torch.bfloat16
        and labels.dtype == torch.int64
        and logits.ndim == 2
        and logits.shape == (4096, VOCAB_SIZE)
        and labels.shape == (4096,)
        and logits.is_contiguous()
        and labels.is_contiguous()
        and not _is_negative_view(logits)
        and not _is_negative_view(labels)
        and torch.__version__ == TORCH_VERSION
        and torch.version.cuda == "13.0"
        and triton.__version__ == "3.9.0"
        and (
            isinstance(logits, FakeTensor)
            or torch.cuda.get_device_capability(logits.device) == (10, 3)
        )
    )


def _check(logits, labels):
    if (
        logits.ndim != 2
        or logits.shape[1] != VOCAB_SIZE
        or not 0 < logits.shape[0] <= 4096
        or labels.shape != (logits.shape[0],)
        or logits.dtype != torch.bfloat16
        or labels.dtype != torch.int64
        or not logits.is_cuda
        or labels.device != logits.device
        or not logits.is_contiguous()
        or not labels.is_contiguous()
        or logits.is_neg()
        or labels.is_neg()
        or torch.__version__ != TORCH_VERSION
        or torch.version.cuda != "13.0"
        or triton.__version__ != "3.9.0"
        or not (
            isinstance(logits, FakeTensor)
            or torch.cuda.get_device_capability(logits.device) == (10, 3)
        )
    ):
        raise ValueError(
            "MTP CE requires contiguous CUDA BF16 logits[T,129280] and int64 "
            "labels[T], 1 <= T <= 4096, on the validated Torch/GB300 runtime"
        )


def _check_backward(logits, labels, stats, grad_loss):
    _check(logits, labels)
    if (
        stats.shape != (2, logits.shape[0])
        or stats.dtype != torch.float32
        or stats.device != logits.device
        or not stats.is_contiguous()
        or stats.is_neg()
        or grad_loss.shape != ()
        or grad_loss.dtype != torch.float32
        or grad_loss.device != logits.device
    ):
        raise ValueError(
            "MTP CE backward requires FP32 stats[2,T] and a scalar FP32 gradient "
            "on the logits device"
        )


@torch.library.custom_op(
    "torchtitan::dsv3_mtp_cross_entropy_forward", mutates_args=(), device_types="cuda"
)
def forward_op(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int
) -> tuple[torch.Tensor, torch.Tensor]:

    _check(logits, labels)
    return _ce_forward(logits, labels, ignore_index)


@forward_op.register_fake
def _forward_fake(logits, labels, ignore_index):
    _check(logits, labels)
    return (
        logits.new_empty((), dtype=torch.float32),
        logits.new_empty((2, logits.shape[0]), dtype=torch.float32),
    )


@torch.library.custom_op(
    "torchtitan::dsv3_mtp_cross_entropy_backward", mutates_args=(), device_types="cuda"
)
def backward_op(
    logits: torch.Tensor,
    labels: torch.Tensor,
    stats: torch.Tensor,
    grad_loss: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:

    _check_backward(logits, labels, stats, grad_loss)
    return _ce_backward(logits, labels, stats, grad_loss, ignore_index)


@backward_op.register_fake
def _backward_fake(logits, labels, stats, grad_loss, ignore_index):
    _check_backward(logits, labels, stats, grad_loss)
    return torch.empty_like(logits)


class MTPCrossEntropyFunction(torch.autograd.Function):
    """First-order CE; labels must be valid vocabulary IDs or ignore_index."""

    @staticmethod
    def spmd_typecheck(result, *, logits, labels, ignore_index):
        spmd.rules.einsum("tv,t->", logits, labels, out=result)

    @staticmethod
    def forward(ctx, logits, labels, ignore_index):  # pyrefly: ignore[bad-override]
        loss, stats = forward_op(logits, labels, ignore_index)
        ctx.save_for_backward(logits, labels, stats)
        ctx.ignore_index = ignore_index
        ctx.set_materialize_grads(False)
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss):  # pyrefly: ignore[bad-override]
        if grad_loss is None:
            return None, None, None
        logits, labels, stats = ctx.saved_tensors
        gradient = backward_op(logits, labels, stats, grad_loss, ctx.ignore_index)
        return gradient, None, None


def cross_entropy_sum(logits, labels, *, ignore_index=-100):
    """Experimental local-vocabulary CE for the validated GB300 runtime.

    This direct entry point runs the candidate for measurement. Model dispatch
    must also check ``ACCEPTED`` and ``supports`` before enabling it.
    """
    return MTPCrossEntropyFunction.apply(logits, labels, ignore_index)


logger = logging.getLogger(__name__)


def _cross_entropy_loss(
    pred_TV: torch.Tensor,
    labels_T: torch.Tensor,
    *,
    global_vocab_size: int | None = None,
    reduction: Literal["sum", "none"] = "sum",
) -> torch.Tensor:
    if (
        ACCEPTED
        and reduction == "sum"
        and global_vocab_size in (None, VOCAB_SIZE)
        and spmd_mesh_size("tp") == 1
        and supports(pred_TV, labels_T)
    ):
        return cross_entropy_sum(pred_TV, labels_T)
    return cross_entropy_loss(
        pred_TV, labels_T, global_vocab_size=global_vocab_size, reduction=reduction
    )


class FusedDSv3MTPLoss(MTPLoss):
    """Reuse native MTP composition and specialize only its raw CE callback."""

    @dataclass(kw_only=True, slots=True)
    class Config(MTPLoss.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.fn = _cross_entropy_loss
        if not ACCEPTED:
            logger.warning(
                "MTP cross-entropy fusion has not met all acceptance gates; "
                "model dispatch will use native cross entropy."
            )


@override(
    target=MTPLoss.Config,
    exact=True,
    description="Use the acceptance-gated DSv3 MTP cross-entropy autograd override.",
)
def fused_dsv3_mtp_loss(cfg: MTPLoss.Config) -> MTPLoss.Config:
    if cfg.global_vocab_size not in (None, VOCAB_SIZE):
        logger.warning(
            "MTP cross-entropy fusion requires the full 129280 vocabulary; "
            "keeping the native loss configuration."
        )
        return cfg
    return derive(cfg, FusedDSv3MTPLoss.Config)
