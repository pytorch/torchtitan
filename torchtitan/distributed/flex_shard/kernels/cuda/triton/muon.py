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
from triton.language.extra.cuda import libdevice

_BLOCK_SIZE = 1024
_LAUNCHERS = {}


def _launch(kernel, grid, *arguments, num_warps=4):
    # Cache code and specialization metadata, never input/output tensors.
    key = (
        kernel,
        grid,
        num_warps,
        tuple(
            (value.device, value.dtype, value.data_ptr() % 16)
            if isinstance(value, torch.Tensor)
            else value
            for value in arguments
        ),
    )
    launcher = _LAUNCHERS.get(key)
    if launcher is None:
        compiled = kernel[grid](*arguments, num_warps=num_warps, enable_fp_fusion=False)
        _LAUNCHERS[key] = compiled[(*grid, *(1 for _ in range(3 - len(grid))))]
    else:
        launcher(*arguments)


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
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
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
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
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
def _cast_partial(
    X,
    Y,
    P,
    N: tl.constexpr,
    C: tl.constexpr,
    SHAPE: tl.constexpr,
    XS: tl.constexpr,
    YS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    chunk, batch = tl.program_id(0), tl.program_id(1)
    index = chunk * BLOCK + tl.arange(0, BLOCK)
    logical = batch * N + index
    value = tl.load(X + _offset(logical, SHAPE, XS), index < N, 0)
    rounded = value.to(tl.bfloat16)
    tl.store(Y + _offset(logical, SHAPE, YS), rounded, index < N)
    value = rounded.to(tl.float32)
    tl.store(P + batch * C + chunk, tl.sum(value * value, 0))


@triton.jit
def _normalize_partial(
    Y,
    P,
    N: tl.constexpr,
    C: tl.constexpr,
    SHAPE: tl.constexpr,
    YS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    REDUCE: tl.constexpr,
):
    chunk, batch = tl.program_id(0), tl.program_id(1)
    reduction = tl.arange(0, REDUCE)
    partial = tl.load(P + batch * C + reduction, reduction < C, 0)
    # Match eager rounding after BF16 norm, before clamp and division.
    norm = libdevice.sqrt_rn(tl.sum(partial, 0)).to(tl.bfloat16).to(tl.float32)
    epsilon = tl.full((), EPS, tl.float32).to(tl.bfloat16).to(tl.float32)
    norm = tl.maximum(norm, epsilon)
    index = chunk * BLOCK + tl.arange(0, BLOCK)
    pointer = Y + _offset(batch * N + index, SHAPE, YS)
    value = tl.load(pointer, index < N, 0).to(tl.float32)
    tl.store(pointer, tl.div_rn(value, norm), index < N)


@triton.jit
def _copy_scaled(
    X,
    Y,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    XS: tl.constexpr,
    YS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + _offset(index, SHAPE, XS), index < N, 0)
    rounded = value.to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + _offset(index, SHAPE, YS), rounded * SCALE, index < N)


def _normalize(update, eps):
    shape = tuple(update.shape)
    numel = math.prod(shape[-2:])
    batch = math.prod(shape[:-2])
    block = 8192
    chunks = triton.cdiv(numel, block)
    result = torch.empty_like(update, dtype=torch.bfloat16)
    partial = torch.empty((batch, chunks), device=update.device, dtype=torch.float32)
    _launch(
        _cast_partial,
        (chunks, batch),
        update,
        result,
        partial,
        numel,
        chunks,
        shape,
        update.stride(),
        result.stride(),
        block,
        num_warps=8,
    )
    _launch(
        _normalize_partial,
        (chunks, batch),
        result,
        partial,
        numel,
        chunks,
        shape,
        result.stride(),
        eps,
        block,
        triton.next_power_of_2(chunks),
        num_warps=8,
    )
    return result


def _zeropower(update, coefficients, eps):
    result = _normalize(update, eps)
    transposed = result.shape[-2] > result.shape[-1]
    if transposed:
        result = result.transpose(-2, -1)
    shape = result.shape
    matrices = result if result.ndim == 2 else result.reshape(-1, *shape[-2:])
    addmm = torch.addmm if result.ndim == 2 else torch.baddbmm
    del result
    for a, b, c in coefficients:
        gram = matrices @ matrices.transpose(-2, -1)
        gram_update = addmm(gram, gram, gram, beta=b, alpha=c)
        del gram
        matrices = addmm(matrices, gram_update, matrices, beta=a)
        del gram_update
    result = matrices.reshape(shape)
    return result.transpose(-2, -1) if transposed else result


def prepare_muon_input(gradient, momentum_buffer, *, momentum, nesterov, out):
    """Fuse momentum and Nesterov without materializing intermediate tensors."""
    count = gradient.numel()
    if count:
        _prepare[(triton.cdiv(count, _BLOCK_SIZE),)](
            gradient,
            momentum_buffer,
            out,
            1 - momentum,
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
    return out


def local_updates(param_groups, *, adjust_learning_rate, coefficients_by_step):
    """Bind optimizer metadata outside the Tensor-only kernel entry points."""
    if any(
        p.device.type != "cuda" or p.dtype not in (torch.float32, torch.bfloat16)
        for group in param_groups
        for p in group["params"]
    ):
        raise ValueError("Triton Muon updates require CUDA FP32 or BF16 storage")

    def compute_direction(
        prepared,
        *,
        matrix_views,
        lr_reference_shape,
        adjust_lr_fn,
        ns_coefficients,
        ns_steps,
        eps,
    ):
        reference = (
            None
            if lr_reference_shape is None
            else adjust_learning_rate(1.0, adjust_lr_fn, lr_reference_shape)
        )
        coefficients = coefficients_by_step(ns_coefficients, ns_steps)
        for view in matrix_views:
            matrices = view.view_as_matrix_batch(prepared)
            result = _zeropower(matrices, coefficients, eps)
            scale = (
                1.0
                if reference is None
                else adjust_learning_rate(1.0, adjust_lr_fn, matrices.shape[-2:])
                / reference
            )
            _launch(
                _copy_scaled,
                (triton.cdiv(matrices.numel(), _BLOCK_SIZE),),
                result,
                matrices,
                matrices.numel(),
                tuple(matrices.shape),
                result.stride(),
                matrices.stride(),
                scale,
                _BLOCK_SIZE,
            )
        return prepared

    def apply_update(
        parameter,
        direction,
        *,
        lr,
        weight_decay,
        adjust_lr_fn,
        compute_matrix_shape,
    ):
        adjusted_lr = adjust_learning_rate(lr, adjust_lr_fn, compute_matrix_shape)
        count = parameter.numel()
        if count:
            _update[(triton.cdiv(count, _BLOCK_SIZE),)](
                parameter,
                direction,
                1 - lr * weight_decay,
                -adjusted_lr,
                count,
                tuple(parameter.shape),
                parameter.stride(),
                direction.stride(),
                _BLOCK_SIZE,
                enable_fp_fusion=False,
            )
        return parameter

    return prepare_muon_input, compute_direction, apply_update
