# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Batch-invariant mode for reproducible RL training.

When enabled, replaces ``log_softmax`` and ``mean.dim`` with Triton kernels
that use a fixed tile iteration order, and configures cuBLAS for deterministic
matmul via ``torch.use_deterministic_algorithms`` plus disabled reduced-precision
reductions.

We intentionally do NOT override ``mm``/``addmm``/``bmm`` with Triton kernels.
During backward, autograd computes ``grad_weight = activation^T @ grad_output``
where the K dimension is ``batch * seq``.  Different batch sizes produce
different K values, which changes the number of K-tiles in a Triton kernel and
therefore the fp32 accumulation boundaries — breaking batch invariance.  cuBLAS
deterministic algorithms handle this correctly.

The kernels are registered via ``torch.library.Library("aten", "IMPL")``
so they are transparent to the model code — no changes needed in the model
definition.

Based on https://github.com/thinking-machines-lab/batch_invariant_ops

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
# ATen override wrappers
# ---------------------------------------------------------------------------


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

    Replaces ``_log_softmax`` and ``mean.dim`` with Triton kernels that
    produce bit-identical results for the same input regardless of batch
    composition.

    Matmul operations (``mm``, ``addmm``, ``bmm``) are left to cuBLAS
    with deterministic algorithms enabled via
    ``torch.use_deterministic_algorithms(True)``.  Triton matmul kernels
    cannot be batch-invariant because during backward, autograd computes
    ``grad_weight = activation^T @ grad_output`` where K = batch * seq —
    different batch sizes change the K-tile decomposition and therefore
    the fp32 accumulation boundaries.

    Also disables reduced-precision reductions and TF32 to prevent
    batch-size-dependent rounding in cuBLAS.

    Safe to call multiple times (idempotent).
    """
    global _ENABLED, _LIB, _SAVED_STATE
    if _ENABLED:
        return

    dispatch_key = getattr(
        torch.accelerator.current_accelerator(), "type", "cpu"
    ).upper()

    _LIB = torch.library.Library("aten", "IMPL")
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
        "Batch-invariant mode enabled: _log_softmax, mean.dim overridden "
        "with Triton kernels; deterministic algorithms enabled; "
        "reduced-precision reductions and TF32 disabled"
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
