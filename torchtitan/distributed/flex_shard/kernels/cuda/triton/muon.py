# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA Muon local updates; PyTorch retains the NS matrix multiplications."""

import math

import torch
import triton
import triton.language as tl

_BLOCK_SIZE = 1024


@triton.jit
def _offset(index, shape: tl.constexpr, strides: tl.constexpr):
    offset = tl.full(index.shape, 0, tl.int64)
    for axis in tl.static_range(len(shape) - 1, -1, -1):
        offset += (index % shape[axis]) * strides[axis]
        index = index // shape[axis]
    return offset


@triton.jit
def _lerp(start, end, weight):
    delta = end - start
    return tl.where(
        tl.abs(weight) < 0.5,
        tl.fma(weight, delta, start),
        tl.fma(-(1.0 - weight), delta, end),
    )


@triton.jit
def _prepare(
    grad,
    buf,
    out,
    weight,
    momentum,
    numel,
    SHAPE: tl.constexpr,
    GS: tl.constexpr,
    BS: tl.constexpr,
    OS: tl.constexpr,
    NESTEROV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = index < numel
    gradient = tl.load(grad + _offset(index, SHAPE, GS), mask, other=0).to(tl.float32)
    old = tl.load(buf + _offset(index, SHAPE, BS), mask, other=0).to(tl.float32)
    updated = _lerp(old, gradient, weight).to(buf.dtype.element_ty)
    tl.store(buf + _offset(index, SHAPE, BS), updated, mask)
    # Eager stores the updated momentum before the Nesterov lerp.
    value = _lerp(gradient, updated.to(tl.float32), momentum) if NESTEROV else updated
    tl.store(out + _offset(index, SHAPE, OS), value, mask)


@triton.jit
def _update(
    param,
    direction,
    decay,
    alpha,
    numel,
    SHAPE: tl.constexpr,
    PS: tl.constexpr,
    DS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = index < numel
    offset = _offset(index, SHAPE, PS)
    value = tl.load(param + offset, mask, other=0).to(tl.float32)
    update = tl.load(direction + _offset(index, SHAPE, DS), mask, other=0).to(
        tl.float32
    )
    # Preserve the dtype rounding between eager mul_ and add_.
    decayed = (value * decay).to(param.dtype.element_ty).to(tl.float32)
    tl.store(param + offset, tl.fma(update, alpha, decayed), mask)


@triton.jit
def _normalize_with_norm(
    Y,
    Norm,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    YS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Flatten batch and chunks to avoid CUDA's grid axis 1 limit.
    program = tl.program_id(0).to(tl.int64)
    chunks: tl.constexpr = triton.cdiv(N, BLOCK)
    chunk = program % chunks
    batch = program // chunks
    norm = tl.load(Norm + batch).to(tl.float32)
    epsilon = tl.full((), EPS, tl.float32).to(Norm.dtype.element_ty).to(tl.float32)
    norm = tl.maximum(norm, epsilon, propagate_nan=tl.PropagateNan.ALL)
    index = chunk * BLOCK + tl.arange(0, BLOCK)
    pointer = Y + _offset(batch * N + index, SHAPE, YS)
    value = tl.load(pointer, index < N, 0).to(tl.float32)
    tl.store(pointer, tl.div_rn(value, norm), index < N)


def normalize_(result, norm, eps):
    """Clamp the PyTorch norm and normalize in place without changing its reduction."""
    if not result.numel():
        return
    # Traverse the contiguous matrix axis, including transposed NS inputs.
    if result.stride(-2) < result.stride(-1):
        result = result.transpose(-2, -1)
    shape = tuple(result.shape)
    numel = math.prod(shape[-2:])
    batch = math.prod(shape[:-2])
    block = 8192
    chunks = triton.cdiv(numel, block)
    _normalize_with_norm[(chunks * batch,)](
        result,
        norm,
        numel,
        shape,
        result.stride(),
        eps,
        block,
        num_warps=8,
        enable_fp_fusion=False,
    )


def momentum_(gradient, momentum_buffer, out, *, weight, momentum, nesterov):
    """Fuse momentum and Nesterov without materializing intermediate tensors."""
    count = gradient.numel()
    if count:
        _prepare[(triton.cdiv(count, _BLOCK_SIZE),)](
            gradient,
            momentum_buffer,
            out,
            weight,
            momentum,
            count,
            tuple(gradient.shape),
            gradient.stride(),
            momentum_buffer.stride(),
            out.stride(),
            nesterov,
            _BLOCK_SIZE,
            enable_fp_fusion=False,
        )


def check_supported(params):
    """Reject storage and backends outside the CUDA kernel contract."""
    if torch.version.hip is not None or any(
        p.device.type != "cuda" or p.dtype not in (torch.float32, torch.bfloat16)
        for p in params
    ):
        raise ValueError("Triton Muon updates require CUDA FP32 or BF16 storage")


def decay_and_update_(parameter, direction, *, decay, alpha):
    """Fuse parameter decay and update with storage rounding between them."""
    count = parameter.numel()
    if count:
        _update[(triton.cdiv(count, _BLOCK_SIZE),)](
            parameter,
            direction,
            decay,
            alpha,
            count,
            tuple(parameter.shape),
            parameter.stride(),
            direction.stride(),
            _BLOCK_SIZE,
            enable_fp_fusion=False,
        )
