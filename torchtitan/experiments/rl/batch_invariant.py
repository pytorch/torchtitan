# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Batch-invariant Triton kernels registered as ATen overrides.

When enabled, these replace cuBLAS matmul, log_softmax, and mean with Triton
kernels that use a fixed tile iteration order. This guarantees that the
same input sequence produces bit-identical results regardless of what other
sequences are in the batch.

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
from collections.abc import Callable, Generator
from typing import Any

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Triton kernel: persistent matmul with deterministic tile ordering
# ---------------------------------------------------------------------------


def _matmul_launch_metadata(
    grid: Callable[..., Any], kernel: Any, args: dict[str, Any]
) -> dict[str, Any]:
    ret: dict[str, Any] = {}
    m, n, k = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={m}, N={n}, K={k}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * m * n * k
    ret["bytes"] = bytes_per_elem * (m * k + n * k + m * n)
    return ret


@triton.jit
def _compute_pid(
    tile_id,
    num_pid_in_group,
    num_pid_m,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit(launch_metadata=_matmul_launch_metadata)
def _matmul_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        if A_LARGE:
            offs_am = offs_am.to(tl.int64)
        if B_LARGE:
            offs_bn = offs_bn.to(tl.int64)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            if A_LARGE or B_LARGE:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
            else:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (
                offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
            )
            b_ptrs = b_ptr + (
                offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
            )

            a = tl.load(
                a_ptrs,
                mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K,
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K,
                other=0.0,
            )
            accumulator = tl.dot(a, b, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if HAS_BIAS:
            bias_ptrs = bias_ptr + offs_cn
            bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0).to(tl.float32)
            accumulator += bias
        c = accumulator.to(c_ptr.dtype.element_ty)
        tl.store(c_ptrs, c, mask=c_mask)


def _get_num_sms() -> int:
    """Return the number of SMs on the current accelerator."""
    device_type = getattr(torch.accelerator.current_accelerator(), "type", "cpu")
    if device_type == "cuda":
        return torch.cuda.get_device_properties(0).multi_processor_count
    return torch.get_num_threads()


_MATMUL_CONFIGS = {
    torch.bfloat16: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 3,
        "num_warps": 8,
    },
    torch.float16: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 3,
        "num_warps": 8,
    },
    torch.float32: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "num_stages": 3,
        "num_warps": 8,
    },
}


def matmul_persistent(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Persistent matmul with deterministic tile ordering.

    Args:
        a: (M, K) input matrix
        b: (K, N) input matrix
        bias: optional (N,) bias vector

    Returns:
        (M, N) result of a @ b [+ bias]
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert bias is None or bias.dim() == 1, "Bias must be 1D"

    NUM_SMS = _get_num_sms()
    M, K = a.shape
    K, N = b.shape  # noqa: F841
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(META: dict[str, Any]) -> tuple[int]:
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            ),
        )

    _matmul_kernel_persistent[grid](
        a,
        b,
        c,
        bias,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        NUM_SMS=NUM_SMS,
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        HAS_BIAS=bias is not None,
        **_MATMUL_CONFIGS[a.dtype],
    )
    return c


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


def _mm_batch_invariant(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return matmul_persistent(a, b)


def _addmm_batch_invariant(
    bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    return matmul_persistent(a, b, bias=bias)


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
        n_elems = 1
        for d in dim:
            n_elems *= x.shape[d]
        return (
            torch.sum(x, dim=dim, keepdim=keepdim, dtype=torch.float32).to(
                dtype or x.dtype
            )
            / n_elems
        )


# ---------------------------------------------------------------------------
# Enable / disable / context manager
# ---------------------------------------------------------------------------

_ENABLED = False
_LIB: torch.library.Library | None = None


def is_batch_invariant_mode_enabled() -> bool:
    return _ENABLED


def enable_batch_invariant_mode() -> None:
    """Register batch-invariant Triton kernels as ATen overrides.

    Replaces ``mm``, ``addmm``, ``_log_softmax``, and ``mean.dim`` with
    Triton kernels that produce bit-identical results for the same input
    regardless of batch composition.

    Safe to call multiple times (idempotent).
    """
    global _ENABLED, _LIB
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

    _ENABLED = True
    logger.info(
        "Batch-invariant mode enabled: mm, addmm, _log_softmax, "
        "mean.dim overridden with Triton kernels"
    )


def disable_batch_invariant_mode() -> None:
    """Unregister batch-invariant ATen overrides."""
    global _ENABLED, _LIB
    if _LIB is not None:
        _LIB._destroy()
    _ENABLED = False
    _LIB = None


@contextlib.contextmanager
def set_batch_invariant_mode(
    enabled: bool = True,
) -> Generator[None, None, None]:
    """Context manager to temporarily enable/disable batch-invariant mode."""
    global _ENABLED, _LIB
    old_data = (_ENABLED, _LIB)
    if enabled:
        enable_batch_invariant_mode()
    else:
        disable_batch_invariant_mode()
    yield
    if _LIB is not None:
        _LIB._destroy()
    _ENABLED, _LIB = old_data
