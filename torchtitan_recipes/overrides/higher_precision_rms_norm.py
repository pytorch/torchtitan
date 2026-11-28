# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Higher-precision RMSNorm override with Triton kernels.

``higher_precision_rms_norm`` replaces every exact ``RMSNorm.Config`` selected by the
override framework with ``HigherPrecisionRMSNorm``, which keeps the ``nn.RMSNorm``
parameter layout (so checkpoints are interchangeable) but computes the norm
with Triton kernels:

- Forward: the sum of squares is accumulated beyond FP32 precision, in one of two
  modes selected by ``use_kahan_summation`` (default True):

  - True: compensated FP32 summation, with no FP64 element math. Each square is
    split into an exact (product, rounding error) pair with an FMA, and the
    pairs are added as double-float values with TwoSum, both within a column
    block (a ``tl.reduce`` tree) and across blocks. This is a Kahan-style
    compensation that, unlike Fast2Sum, does not require ``|a| >= |b|``.
  - False: squares and sums in FP64. This can be compute-bound on GPUs with
    reduced FP64 throughput; prefer True there.

  ``rstd`` comes from per-row scalar FP64 math in both modes. ``tl.rsqrt`` on
  FP64 is accurate to about 1 ulp of FP64 rather than correctly rounded, so the
  stored FP32 copy is almost always the correctly rounded value. In FP64 mode
  ``rstd`` stays FP64 and ``x * rstd * weight`` is evaluated in FP64, then
  rounded once to the output dtype. In Kahan mode ``rstd`` is rounded to FP32 and
  ``(x * rstd) * weight`` is evaluated in FP32, so the output carries two FP32
  roundings before the cast to the output dtype.
- Backward: FP32 math. ``dx`` is a per-row two-pass kernel; ``dw`` is a
  deterministic two-stage reduction over rows (no atomics).
- FP64 input is supported with ``use_kahan_summation=False``. The forward and
  backward then run entirely in FP64, and ``rstd`` is saved in FP64.
- Dtypes: ``input_dtype`` and ``out_dtype`` are separate settings.
  ``input_dtype`` rounds the input to that dtype inside the kernel. This equals
  casting the input to ``input_dtype`` before the norm, without materializing the
  cast copy; the norm math then runs on the rounded values. Backward re-rounds
  from the saved original input. ``out_dtype`` is the dtype of the returned
  tensor. If unset, it is ``input_dtype`` when that is set, and otherwise the
  input's dtype. Setting only ``out_dtype`` normalizes the unrounded input and
  rounds once, at the output. ``dx`` always has the original input's dtype.
"""

from dataclasses import dataclass, replace
from typing import Literal

import torch
import triton
import triton.language as tl

from torchtitan.config import derive, override, TORCH_DTYPE_MAP
from torchtitan.models.common.nn_modules import RMSNorm

__all__ = [
    "HigherPrecisionRMSNorm",
    "rms_norm_backward_kernel",
    "rms_norm_forward_kernel",
    "higher_precision_rms_norm",
    "higher_precision_rms_norm_backward_op",
    "higher_precision_rms_norm_op",
]

_TRITON_DTYPES = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}


# Elements of x held per program in the row kernels (BLOCK_T * BLOCK_D).
_ROW_BLOCK_ELEMENTS = 8192
_MAX_BLOCK_D = 8192
_ROW_KERNEL_NUM_WARPS = 8
# dw kernel tile: rows per inner step and columns per program.
_DW_BLOCK_T = 32
_DW_BLOCK_D = 128
# Row-chunks per SM for the dw partial sums.
_DW_CHUNKS_PER_SM = 4


@triton.jit
def _two_sum(a, b):
    """Return ``(s, e)`` with ``a + b == s + e`` exactly, for any ``|a|``, ``|b|``."""
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


@triton.jit
def _two_sum_combine(hi_a, lo_a, hi_b, lo_b):
    """Combine two ``(hi, lo)`` double-float partial sums (``tl.reduce`` combiner)."""
    hi, err = _two_sum(hi_a, hi_b)
    return hi, lo_a + lo_b + err


@triton.jit
def _round_input(
    x,
    ROUND_INPUT: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    """Round ``x`` through ``INPUT_DTYPE`` when ``ROUND_INPUT``."""
    if ROUND_INPUT:
        x = x.to(INPUT_DTYPE).to(COMPUTE_DTYPE)
    return x


@triton.jit
def _rms_norm_forward_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    rstd_ptr,
    NUM_ROWS,
    eps,
    x_row_stride,
    D: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    USE_KAHAN: tl.constexpr,
    ROUND_INPUT: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """``y = x * rsqrt(mean(x^2) + eps) * w`` for rows of ``x [NUM_ROWS, D]``."""
    rows = (tl.program_id(0).to(tl.int64) * BLOCK_T) + tl.arange(0, BLOCK_T)
    row_mask = rows < NUM_ROWS

    # Sum of squares as an FP32 (hi, lo) pair, or in FP64.
    sum_hi = tl.zeros([BLOCK_T], dtype=tl.float32)
    sum_lo = tl.zeros([BLOCK_T], dtype=tl.float32)
    sum_sq_fp64 = tl.zeros([BLOCK_T], dtype=tl.float64)
    for offset in tl.range(0, D, BLOCK_D):
        cols = offset + tl.arange(0, BLOCK_D)
        x = tl.load(
            x_ptr + rows[:, None] * x_row_stride + cols[None, :],
            mask=row_mask[:, None] & (cols[None, :] < D),
            other=0.0,
            eviction_policy="evict_last",
        ).to(COMPUTE_DTYPE)
        x = _round_input(x, ROUND_INPUT, INPUT_DTYPE, COMPUTE_DTYPE)
        if USE_KAHAN:
            # x * x as an exact (square, rounding error) pair, reduced with
            # compensated (TwoSum) adds in the row and across blocks. This is
            # only exact if the compiler never contracts a mul and a following
            # add into an FMA, so the launch sets enable_fp_fusion=False.
            square = x * x
            square_err = tl.fma(x, x, -square)
            block_hi, block_lo = tl.reduce(
                (square, square_err), axis=1, combine_fn=_two_sum_combine
            )
            sum_hi, err = _two_sum(sum_hi, block_hi)
            sum_lo += err + block_lo
        else:
            sum_sq_fp64 += tl.sum(x * x, axis=1)

    if USE_KAHAN:
        sum_sq = sum_hi.to(tl.float64) + sum_lo.to(tl.float64)
    else:
        sum_sq = sum_sq_fp64
    # Per-row scalar math is FP64 in both modes. In FP64 mode rstd stays FP64 for
    # the multiplies below; in Kahan mode it is rounded to FP32 first. The stored
    # copy (consumed by backward) is always FP32.
    rstd = tl.rsqrt(sum_sq / D + eps)
    tl.store(rstd_ptr + rows, rstd.to(rstd_ptr.dtype.element_ty), mask=row_mask)
    if USE_KAHAN:
        rstd = rstd.to(tl.float32)

    for offset in tl.range(0, D, BLOCK_D):
        cols = offset + tl.arange(0, BLOCK_D)
        mask = row_mask[:, None] & (cols[None, :] < D)
        x = tl.load(
            x_ptr + rows[:, None] * x_row_stride + cols[None, :],
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(COMPUTE_DTYPE)
        x = _round_input(x, ROUND_INPUT, INPUT_DTYPE, COMPUTE_DTYPE)
        y = x * rstd[:, None]
        if HAS_WEIGHT:
            w = tl.load(w_ptr + cols, mask=cols < D, other=1.0).to(COMPUTE_DTYPE)
            y = y * w[None, :]
        # y is contiguous [NUM_ROWS, D].
        tl.store(y_ptr + rows[:, None] * D + cols[None, :], y, mask=mask)


@triton.jit
def _rms_norm_backward_dx_kernel(
    dy_ptr,
    x_ptr,
    w_ptr,
    rstd_ptr,
    dx_ptr,
    NUM_ROWS,
    dy_row_stride,
    x_row_stride,
    D: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    ROUND_INPUT: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """``dx = rstd * (dy * w - x_hat * mean(dy * w * x_hat))``, ``x_hat = x * rstd``."""
    rows = (tl.program_id(0).to(tl.int64) * BLOCK_T) + tl.arange(0, BLOCK_T)
    row_mask = rows < NUM_ROWS
    rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0)

    correction = tl.zeros([BLOCK_T], dtype=COMPUTE_DTYPE)
    for offset in tl.range(0, D, BLOCK_D):
        cols = offset + tl.arange(0, BLOCK_D)
        mask = row_mask[:, None] & (cols[None, :] < D)
        dy = tl.load(
            dy_ptr + rows[:, None] * dy_row_stride + cols[None, :],
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(COMPUTE_DTYPE)
        x = tl.load(
            x_ptr + rows[:, None] * x_row_stride + cols[None, :],
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(COMPUTE_DTYPE)
        x = _round_input(x, ROUND_INPUT, INPUT_DTYPE, COMPUTE_DTYPE)
        if HAS_WEIGHT:
            w = tl.load(w_ptr + cols, mask=cols < D, other=1.0).to(COMPUTE_DTYPE)
            dy = dy * w[None, :]
        correction += tl.sum(dy * x * rstd[:, None], axis=1)

    for offset in tl.range(0, D, BLOCK_D):
        cols = offset + tl.arange(0, BLOCK_D)
        mask = row_mask[:, None] & (cols[None, :] < D)
        dy = tl.load(
            dy_ptr + rows[:, None] * dy_row_stride + cols[None, :],
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(COMPUTE_DTYPE)
        x = tl.load(
            x_ptr + rows[:, None] * x_row_stride + cols[None, :],
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(COMPUTE_DTYPE)
        x = _round_input(x, ROUND_INPUT, INPUT_DTYPE, COMPUTE_DTYPE)
        if HAS_WEIGHT:
            w = tl.load(w_ptr + cols, mask=cols < D, other=1.0).to(COMPUTE_DTYPE)
            dy = dy * w[None, :]
        x_hat = x * rstd[:, None]
        dx = rstd[:, None] * (dy - x_hat * (correction[:, None] * (1.0 / D)))
        # dx is contiguous [NUM_ROWS, D].
        tl.store(dx_ptr + rows[:, None] * D + cols[None, :], dx, mask=mask)


@triton.jit
def _rms_norm_backward_dw_kernel(
    dy_ptr,
    x_ptr,
    rstd_ptr,
    partial_dw_ptr,
    NUM_ROWS,
    ROWS_PER_CHUNK,
    dy_row_stride,
    x_row_stride,
    D: tl.constexpr,
    ROUND_INPUT: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """Write ``partial_dw[chunk] = sum_{t in chunk} dy[t] * x[t] * rstd[t]``."""
    chunk = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    col_mask = cols < D

    row_begin = chunk * ROWS_PER_CHUNK
    row_end = tl.minimum(row_begin + ROWS_PER_CHUNK, NUM_ROWS)
    acc = tl.zeros([BLOCK_D], dtype=COMPUTE_DTYPE)
    for start in tl.range(row_begin, row_end, BLOCK_T):
        rows = start + tl.arange(0, BLOCK_T)
        row_mask = rows < row_end
        mask = row_mask[:, None] & col_mask[None, :]
        dy = tl.load(
            dy_ptr + rows[:, None] * dy_row_stride + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(COMPUTE_DTYPE)
        x = tl.load(
            x_ptr + rows[:, None] * x_row_stride + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(COMPUTE_DTYPE)
        x = _round_input(x, ROUND_INPUT, INPUT_DTYPE, COMPUTE_DTYPE)
        rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0)
        acc += tl.sum(dy * x * rstd[:, None], axis=0)
    tl.store(partial_dw_ptr + chunk * D + cols, acc, mask=col_mask)


def _resolve_dtypes(
    x: torch.Tensor,
    input_dtype: torch.dtype | None,
    out_dtype: torch.dtype | None,
) -> tuple[bool, tl.dtype, torch.dtype]:
    """Return ``(round_input, triton_input_dtype, y_dtype)``."""
    for dtype in (x.dtype, input_dtype, out_dtype):
        if dtype is not None and dtype not in _TRITON_DTYPES:
            raise ValueError(f"Unsupported norm dtype {dtype}")
    round_input = input_dtype is not None and input_dtype != x.dtype
    return (
        round_input,
        _TRITON_DTYPES[input_dtype if round_input else x.dtype],
        out_dtype or input_dtype or x.dtype,
    )


def _stats_dtype(x: torch.Tensor) -> torch.dtype:
    """Dtype of ``rstd`` and of the backward math: FP64 for FP64 input, else FP32."""
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def _as_rows(tensor: torch.Tensor) -> torch.Tensor:
    """View ``[..., D]`` as ``[T, D]`` with unit stride along ``D``."""
    rows = tensor.reshape(-1, tensor.shape[-1])
    return rows if rows.stride(-1) == 1 else rows.contiguous()


def _row_block_shape(dim: int) -> tuple[int, int]:
    block_d = min(triton.next_power_of_2(dim), _MAX_BLOCK_D)
    block_t = max(1, _ROW_BLOCK_ELEMENTS // block_d)
    return block_t, block_d


def rms_norm_forward_kernel(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    use_kahan_summation: bool = True,
    input_dtype: torch.dtype | None = None,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(y, rstd)`` for ``x [..., D]``; ``rstd`` has shape ``x.shape[:-1]``.

    ``y`` has dtype ``out_dtype``, else ``input_dtype``, else ``x.dtype``. ``rstd``
    is FP64 for FP64 ``x`` and FP32 otherwise. FP64 ``x`` requires
    ``use_kahan_summation=False``.
    """
    dim = x.shape[-1]
    if weight is not None and weight.shape != (dim,):
        raise ValueError(f"weight must have shape ({dim},), got {tuple(weight.shape)}")
    if x.dtype == torch.float64 and use_kahan_summation:
        raise ValueError("FP64 input requires use_kahan_summation=False")
    round_input, triton_input_dtype, y_dtype = _resolve_dtypes(
        x, input_dtype, out_dtype
    )
    x_rows = _as_rows(x)
    num_rows = x_rows.shape[0]
    y = torch.empty(x.shape, dtype=y_dtype, device=x.device)
    rstd = torch.empty(x.shape[:-1], dtype=_stats_dtype(x), device=x.device)

    block_t, block_d = _row_block_shape(dim)
    _rms_norm_forward_kernel[(triton.cdiv(num_rows, block_t),)](
        x_rows,
        weight if weight is not None else x_rows,
        y,
        rstd,
        num_rows,
        eps,
        x_rows.stride(0),
        D=dim,
        HAS_WEIGHT=weight is not None,
        USE_KAHAN=use_kahan_summation,
        ROUND_INPUT=round_input,
        INPUT_DTYPE=triton_input_dtype,
        COMPUTE_DTYPE=tl.float32 if use_kahan_summation else tl.float64,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=_ROW_KERNEL_NUM_WARPS,
        # Keep the TwoSum steps exact: no mul+add contraction into FMA.
        enable_fp_fusion=False,
    )
    return y, rstd


def rms_norm_backward_kernel(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor | None,
    rstd: torch.Tensor,
    input_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(dx, dw)``; ``dw`` is an empty tensor when ``weight`` is None.

    ``x`` is the original forward input; ``dx`` has ``x.dtype``.
    """
    dim = x.shape[-1]
    round_input, triton_input_dtype, _ = _resolve_dtypes(x, input_dtype, None)
    x_rows = _as_rows(x)
    dy_rows = _as_rows(dy)
    num_rows = x_rows.shape[0]
    dx = torch.empty(x.shape, dtype=x.dtype, device=x.device)

    block_t, block_d = _row_block_shape(dim)
    _rms_norm_backward_dx_kernel[(triton.cdiv(num_rows, block_t),)](
        dy_rows,
        x_rows,
        weight if weight is not None else x_rows,
        rstd,
        dx,
        num_rows,
        dy_rows.stride(0),
        x_rows.stride(0),
        D=dim,
        HAS_WEIGHT=weight is not None,
        ROUND_INPUT=round_input,
        INPUT_DTYPE=triton_input_dtype,
        COMPUTE_DTYPE=_TRITON_DTYPES[_stats_dtype(x)],
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=_ROW_KERNEL_NUM_WARPS,
    )
    if weight is None:
        return dx, x.new_empty(0)

    # Two-stage dw: each program reduces a fixed chunk of rows in order, then the
    # chunks are summed by a fixed-order reduction, so the result is deterministic.
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    rows_per_chunk = max(
        _DW_BLOCK_T, triton.cdiv(num_rows, _DW_CHUNKS_PER_SM * num_sms)
    )
    num_chunks = triton.cdiv(num_rows, rows_per_chunk)
    partial_dw = torch.empty((num_chunks, dim), dtype=_stats_dtype(x), device=x.device)
    _rms_norm_backward_dw_kernel[(num_chunks, triton.cdiv(dim, _DW_BLOCK_D))](
        dy_rows,
        x_rows,
        rstd,
        partial_dw,
        num_rows,
        rows_per_chunk,
        dy_rows.stride(0),
        x_rows.stride(0),
        D=dim,
        ROUND_INPUT=round_input,
        INPUT_DTYPE=triton_input_dtype,
        COMPUTE_DTYPE=_TRITON_DTYPES[_stats_dtype(x)],
        BLOCK_T=_DW_BLOCK_T,
        BLOCK_D=_DW_BLOCK_D,
        num_warps=4,
    )
    return dx, partial_dw.sum(0).to(weight.dtype)


@torch.library.custom_op(
    "torchtitan::higher_precision_rms_norm",
    mutates_args=(),
    device_types="cuda",
)
def higher_precision_rms_norm_op(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    use_kahan_summation: bool = True,
    input_dtype: torch.dtype | None = None,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm over the last dim; returns ``(y, rstd)``."""
    return rms_norm_forward_kernel(
        x, weight, eps, use_kahan_summation, input_dtype, out_dtype
    )


@higher_precision_rms_norm_op.register_fake
def _higher_precision_rms_norm_op_fake(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    use_kahan_summation: bool = True,
    input_dtype: torch.dtype | None = None,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, y_dtype = _resolve_dtypes(x, input_dtype, out_dtype)
    return (
        torch.empty(x.shape, dtype=y_dtype, device=x.device),
        torch.empty(x.shape[:-1], dtype=_stats_dtype(x), device=x.device),
    )


@torch.library.custom_op(
    "torchtitan::higher_precision_rms_norm_backward",
    mutates_args=(),
    device_types="cuda",
)
def higher_precision_rms_norm_backward_op(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor | None,
    rstd: torch.Tensor,
    input_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradients ``(dx, dw)`` of ``torchtitan::higher_precision_rms_norm``."""
    return rms_norm_backward_kernel(dy, x, weight, rstd, input_dtype)


@higher_precision_rms_norm_backward_op.register_fake
def _higher_precision_rms_norm_backward_op_fake(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor | None,
    rstd: torch.Tensor,
    input_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    dw = x.new_empty(0) if weight is None else torch.empty_like(weight)
    return torch.empty(x.shape, dtype=x.dtype, device=x.device), dw


def _higher_precision_rms_norm_setup_context(ctx, inputs, output) -> None:
    x, weight, _, _, input_dtype, _ = inputs
    _, rstd = output
    ctx.input_dtype = input_dtype
    ctx.save_for_backward(x, weight, rstd)
    ctx.mark_non_differentiable(rstd)


def _higher_precision_rms_norm_autograd_backward(ctx, grad_y, grad_rstd):
    x, weight, rstd = ctx.saved_tensors
    dx, dw = higher_precision_rms_norm_backward_op(
        grad_y, x, weight, rstd, ctx.input_dtype
    )
    return dx, (None if weight is None else dw), None, None, None, None


higher_precision_rms_norm_op.register_autograd(
    _higher_precision_rms_norm_autograd_backward,
    setup_context=_higher_precision_rms_norm_setup_context,
)


class HigherPrecisionRMSNorm(RMSNorm):
    """RMSNorm with FP64 / Kahan-FP32 sum-of-squares accumulation (Triton)."""

    @dataclass(kw_only=True, slots=True)
    class Config(RMSNorm.Config):
        # True: compensated FP32 accumulation (no FP64 element math). False: FP64,
        # which can be compute-bound on GPUs with reduced FP64 throughput.
        use_kahan_summation: bool = True
        # Round the input to this dtype in-kernel before normalizing, like casting
        # it first without the copy. None keeps the input unrounded.
        input_dtype: torch.dtype | None = None
        # Output dtype (e.g. FP32 for the LM head or residual add). None means
        # ``input_dtype`` if set, else the input's dtype.
        out_dtype: torch.dtype | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.use_kahan_summation = config.use_kahan_summation
        self.input_dtype = config.input_dtype
        self.out_dtype = config.out_dtype

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        y, _ = higher_precision_rms_norm_op(
            input,
            self.weight,
            self.eps,
            self.use_kahan_summation,
            self.input_dtype,
            self.out_dtype,
        )
        return y


@override(
    target=RMSNorm.Config,
    exact=True,
    description="RMSNorm with FP64 / Kahan-FP32 sum-of-squares accumulation in Triton.",
)
def higher_precision_rms_norm(
    cfg: RMSNorm.Config,
    *,
    use_kahan_summation: bool = True,
    input_dtype: Literal["float16", "bfloat16", "float32"] | None = None,
    out_dtype: Literal["float16", "bfloat16", "float32"] | None = None,
) -> HigherPrecisionRMSNorm.Config:
    # Dtypes are names so they can be passed as JSON on the --override CLI.
    sharding_config = cfg.sharding_config
    if sharding_config is not None:
        # The Triton op is opaque to SPMD type propagation, so run it as a local
        # region typed by the norm's declared input and output layouts.
        sharding_config = replace(sharding_config, local_spmd=True)
    return derive(
        cfg,
        HigherPrecisionRMSNorm.Config,
        use_kahan_summation=use_kahan_summation,
        input_dtype=None if input_dtype is None else TORCH_DTYPE_MAP[input_dtype],
        out_dtype=None if out_dtype is None else TORCH_DTYPE_MAP[out_dtype],
        sharding_config=sharding_config,
    )
