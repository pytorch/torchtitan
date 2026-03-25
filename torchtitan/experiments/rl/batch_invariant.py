# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Batch-invariant mode for reproducible RL training.

When enabled, replaces ``mm``, ``addmm``, ``log_softmax``, and ``mean.dim``
with Triton kernels that use a fixed tile iteration order, producing
bit-identical results for the same input regardless of batch composition.

Also disables reduced-precision reductions and TF32 to prevent
batch-size-dependent rounding.

The kernels are registered via ``torch.library.Library("aten", "IMPL")``
so they are transparent to the model code — no changes needed in the model
definition.

Based on https://github.com/thinking-machines-lab/batch_invariant_ops.

Usage:
    from torchtitan.experiments.rl.batch_invariant import enable_batch_invariant_mode
    enable_batch_invariant_mode()
"""

import contextlib
import logging
from collections.abc import Generator
from typing import Any

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Triton kernel: log_softmax with fixed per-row reduction
# ---------------------------------------------------------------------------


@triton.jit
def _log_softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """One block per row, multi-pass for large vocab sizes."""
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Pass 1: find max for numerical stability
    max_val = -float("inf")
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float("inf"))
        max_val = tl.max(tl.maximum(vals, max_val))

    # Pass 2: sum of exp(x - max)
    sum_exp = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0))

    log_sum_exp = tl.log(sum_exp)

    # Pass 3: compute and store log_softmax
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask)
        output = vals - max_val - log_sum_exp
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)


def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Batch-invariant log_softmax along the last dimension."""
    if dim != -1 and dim != x.ndim - 1:
        raise ValueError(
            "This implementation only supports log_softmax along the last dimension"
        )

    original_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    n_rows, n_cols = x_2d.shape

    output = torch.empty_like(x_2d)
    BLOCK_SIZE = 1024
    grid = (n_rows,)

    _log_softmax_kernel[grid](
        x_2d,
        output,
        x_2d.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output.reshape(original_shape)


# ---------------------------------------------------------------------------
# Triton kernel: mean reduction along a single dim
# ---------------------------------------------------------------------------


@triton.jit
def _mean_kernel(
    input_ptr,
    output_ptr,
    input_stride0,
    input_stride1,
    input_stride2,
    output_stride0,
    output_stride1,
    M,
    N,
    K,
    BLOCK_SIZE: tl.constexpr,
):
    """Input viewed as (M, N, K) where N is the reduction dim."""
    pid = tl.program_id(0)
    m_idx = pid // K
    k_idx = pid % K

    if m_idx >= M or k_idx >= K:
        return

    acc = 0.0
    for n_start in range(0, N, BLOCK_SIZE):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N
        input_idx = (
            m_idx * input_stride0 + n_offsets * input_stride1 + k_idx * input_stride2
        )
        vals = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        acc += tl.sum(vals)

    mean_val = acc / N
    output_idx = m_idx * output_stride0 + k_idx * output_stride1
    tl.store(output_ptr + output_idx, mean_val)


def mean_dim(
    x: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Batch-invariant mean along a single dimension."""
    assert x.is_cuda, "Input must be a CUDA tensor"
    assert (
        -x.ndim <= dim < x.ndim
    ), f"Invalid dimension {dim} for tensor with {x.ndim} dimensions"

    if dim < 0:
        dim = dim + x.ndim

    if dtype is None:
        if x.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            dtype = torch.float32
        else:
            dtype = x.dtype

    if x.dtype != dtype:
        x = x.to(dtype)

    shape = list(x.shape)

    # View as (M, N, K) where N is the reduction dim
    M = 1
    for i in range(dim):
        M *= shape[i]
    N = shape[dim]
    K = 1
    for i in range(dim + 1, len(shape)):
        K *= shape[i]

    input_3d = x.reshape(M, N, K)

    if keepdim:
        output_shape = shape.copy()
        output_shape[dim] = 1
    else:
        output_shape = shape[:dim] + shape[dim + 1 :]

    output = torch.empty(output_shape, dtype=dtype, device=x.device)

    if keepdim:
        output_2d = output.reshape(M, 1, K).squeeze(1)
    else:
        output_2d = output.reshape(M, K)

    grid = (M * K,)
    BLOCK_SIZE = 1024

    _mean_kernel[grid](
        input_3d,
        output_2d,
        input_3d.stride(0),
        input_3d.stride(1),
        input_3d.stride(2),
        output_2d.stride(0),
        output_2d.stride(1) if output_2d.ndim > 1 else 0,
        M,
        N,
        K,
        BLOCK_SIZE,
    )
    return output


# ---------------------------------------------------------------------------
# Triton kernel: matrix multiplication (mm / addmm)
# ---------------------------------------------------------------------------


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fixed-tile-order matmul: C[M, N] = A[M, K] @ B[K, N] in fp32."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        mask_a = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        mask_b = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask_c)


def triton_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batch-invariant matrix multiplication using Triton with fixed tile order."""
    assert a.ndim == 2 and b.ndim == 2, "inputs must be 2D"
    assert a.shape[1] == b.shape[0], f"shape mismatch: {a.shape} @ {b.shape}"

    M, K = a.shape
    _, N = b.shape

    # Ensure contiguous for predictable strides
    a = a.contiguous()
    b = b.contiguous()

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return c


# ---------------------------------------------------------------------------
# ATen override wrappers
# ---------------------------------------------------------------------------


def _mm_batch_invariant(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batch-invariant wrapper for aten::mm."""
    return triton_mm(a, b)


def _addmm_batch_invariant(
    bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Batch-invariant wrapper for aten::addmm."""
    return triton_mm(a, b) + bias


def _log_softmax_batch_invariant(
    x: torch.Tensor, dim: int, _half_to_float: bool
) -> torch.Tensor:
    assert not _half_to_float, "half_to_float not implemented"
    return log_softmax(x, dim=dim)


def _mean_batch_invariant(
    x: torch.Tensor,
    dim: list[int],
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    assert dtype is None or dtype == torch.float32, f"unsupported dtype: {dtype}"
    if len(dim) == 1:
        return mean_dim(x, dim[0], keepdim=keepdim)
    else:
        assert x.dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ), "only float types supported"
        if len(dim) == 0:
            dim = list(range(x.ndim))
        # Reduce dimensions one at a time from largest to smallest
        # to handle shifting indices during iterative reduction.
        result = x.to(torch.float32)
        sorted_dims = sorted([d % x.ndim for d in dim], reverse=True)
        for d in sorted_dims:
            result = mean_dim(result, dim=d, keepdim=True)
        if not keepdim:
            for d in sorted_dims:
                result = result.squeeze(d)
        return result


# ---------------------------------------------------------------------------
# Enable / disable / context manager
# ---------------------------------------------------------------------------

_ENABLED = False
_LIB: torch.library.Library | None = None
_SAVED_STATE: dict[str, Any] | None = None


def is_batch_invariant_mode_enabled() -> bool:
    return _ENABLED


def enable_batch_invariant_mode() -> None:
    """Enable batch-invariant mode for reproducible RL training.

    Replaces ``mm``, ``addmm``, ``_log_softmax``, and ``mean.dim`` with
    Triton kernels that use a fixed tile iteration order, producing
    bit-identical results for the same input regardless of batch composition.

    Also disables reduced-precision reductions and TF32 to prevent
    batch-size-dependent rounding.

    Safe to call multiple times (idempotent).
    """
    global _ENABLED, _LIB, _SAVED_STATE
    if _ENABLED:
        return

    dispatch_key = getattr(
        torch.accelerator.current_accelerator(), "type", "cpu"
    ).upper()

    _LIB = torch.library.Library("aten", "IMPL")
    _LIB.impl("aten::mm", _mm_batch_invariant, dispatch_key)
    _LIB.impl("aten::addmm", _addmm_batch_invariant, dispatch_key)
    _LIB.impl("aten::_log_softmax", _log_softmax_batch_invariant, dispatch_key)
    _LIB.impl("aten::mean.dim", _mean_batch_invariant, dispatch_key)

    # Save current state for restoration
    _SAVED_STATE = {
        "bf16": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "fp16": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
    }

    # Disable reduced-precision reductions: these allow cuBLAS to use
    # lower-precision accumulation that can round differently depending
    # on batch size / tile decomposition.
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Disable TF32 for exact fp32 accumulation
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Enable deterministic algorithms so cuBLAS uses batch-invariant
    # matmul implementations
    torch.use_deterministic_algorithms(True)

    _ENABLED = True
    logger.info(
        "Batch-invariant mode enabled: mm, addmm, _log_softmax, mean.dim "
        "overridden with Triton kernels; reduced-precision reductions "
        "and TF32 disabled"
    )


def disable_batch_invariant_mode() -> None:
    """Unregister batch-invariant ATen overrides and restore settings."""
    global _ENABLED, _LIB, _SAVED_STATE
    if _LIB is not None:
        _LIB._destroy()
    # Restore saved settings
    if _SAVED_STATE is not None:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            _SAVED_STATE["bf16"]
        )
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            _SAVED_STATE["fp16"]
        )
        torch.backends.cuda.matmul.allow_tf32 = _SAVED_STATE["tf32_matmul"]
        torch.backends.cudnn.allow_tf32 = _SAVED_STATE["tf32_cudnn"]
        torch.use_deterministic_algorithms(_SAVED_STATE["deterministic"])
    _ENABLED = False
    _LIB = None
    _SAVED_STATE = None


@contextlib.contextmanager
def set_batch_invariant_mode(
    enabled: bool = True,
) -> Generator[None, None, None]:
    """Context manager to temporarily enable/disable batch-invariant mode."""
    global _ENABLED, _LIB, _SAVED_STATE
    old_data = (_ENABLED, _LIB, _SAVED_STATE)
    if enabled:
        enable_batch_invariant_mode()
    else:
        disable_batch_invariant_mode()
    yield
    disable_batch_invariant_mode()
    _ENABLED, _LIB, _SAVED_STATE = old_data
