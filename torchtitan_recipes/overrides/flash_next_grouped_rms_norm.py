# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Triton kernels for Qwen3.8-Flash-Next's per-stream offset RMSNorm.

``GroupedOffsetRMSNorm`` normalizes each ``group_size`` slice of the
``[T, S * D]`` residual streams and scales by ``1 + weight``. Compiled, its
backward fuses the input gradient into the column reduction over tokens for
the weight gradient, which reads the stream with a ``S * D`` stride. These
kernels keep the two reductions apart:

```text
forward   one program per (token chunk, group): y = x * rstd * (1 + w), keep rstd
backward  one program per (token chunk, group): dx row by row (one pass), and a
          per-chunk dw partial, summed after the kernel (deterministic)
```

Used by the gated residual override
(``torchtitan_recipes.overrides.flash_next_gated_residual``).
"""

import torch
import triton
import triton.language as tl


__all__ = ["grouped_offset_rms_norm"]

# Rows per program, from a sweep at T=16384, C=4x2560 on GB300 (4 warps both).
# The backward keeps one dw accumulator per program, so its chunk also bounds
# the partial buffer at ceil(T / 16) x C fp32.
_FORWARD_ROWS_PER_PROGRAM = 8
_BACKWARD_ROWS_PER_PROGRAM = 16


@triton.jit
def _forward_kernel(
    x_ptr,
    weight_ptr,
    y_ptr,
    rstd_ptr,
    num_rows,
    eps,
    CHANNELS: tl.constexpr,
    GROUP: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < GROUP
    cols = group * GROUP + offsets
    scale = 1.0 + tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    row_begin = pid * ROWS_PER_PROGRAM
    for row in range(row_begin, tl.minimum(row_begin + ROWS_PER_PROGRAM, num_rows)):
        # int64: row * CHANNELS overflows int32 past ~210K rows at 125B widths.
        row64 = row.to(tl.int64)
        x = tl.load(x_ptr + row64 * CHANNELS + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        rstd = tl.rsqrt(tl.sum(x * x, 0) / GROUP + eps)
        tl.store(rstd_ptr + row64 * NUM_GROUPS + group, rstd)
        tl.store(
            y_ptr + row64 * CHANNELS + cols,
            (x * rstd * scale).to(y_ptr.dtype.element_ty),
            mask=mask,
        )


@triton.jit
def _backward_kernel(
    x_ptr,
    weight_ptr,
    rstd_ptr,
    grad_y_ptr,
    grad_x_ptr,
    weight_partial_ptr,
    num_rows,
    CHANNELS: tl.constexpr,
    GROUP: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < GROUP
    cols = group * GROUP + offsets
    scale = 1.0 + tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    grad_weight = tl.zeros([BLOCK], tl.float32)
    row_begin = pid * ROWS_PER_PROGRAM
    for row in range(row_begin, tl.minimum(row_begin + ROWS_PER_PROGRAM, num_rows)):
        row64 = row.to(tl.int64)
        x = tl.load(x_ptr + row64 * CHANNELS + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        grad_y = tl.load(grad_y_ptr + row64 * CHANNELS + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        rstd = tl.load(rstd_ptr + row64 * NUM_GROUPS + group)
        hat = x * rstd
        grad_hat = grad_y * scale
        # d/dx of x * rsqrt(mean(x^2) + eps): rstd * (g - hat * mean(g * hat)).
        mean = tl.sum(grad_hat * hat, 0) / GROUP
        tl.store(
            grad_x_ptr + row64 * CHANNELS + cols,
            (rstd * (grad_hat - hat * mean)).to(grad_x_ptr.dtype.element_ty),
            mask=mask,
        )
        grad_weight += grad_y * hat
    tl.store(
        weight_partial_ptr + pid.to(tl.int64) * CHANNELS + cols, grad_weight, mask=mask
    )


def _launch_shape(num_rows: int, group_size: int, channels: int, rows_per_program: int):
    num_groups = channels // group_size
    grid = (triton.cdiv(num_rows, rows_per_program), num_groups)
    return num_groups, grid, triton.next_power_of_2(group_size)


@torch.library.custom_op(
    "torchtitan_flash_next::grouped_offset_rms_norm", mutates_args=()
)
def _forward_op(
    x: torch.Tensor, weight: torch.Tensor, eps: float, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    num_rows, channels = x.shape
    num_groups, grid, block = _launch_shape(
        num_rows, group_size, channels, _FORWARD_ROWS_PER_PROGRAM
    )
    y = torch.empty_like(x)
    rstd = torch.empty(num_rows, num_groups, device=x.device, dtype=torch.float32)
    _forward_kernel[grid](
        x,
        weight,
        y,
        rstd,
        num_rows,
        eps,
        CHANNELS=channels,
        GROUP=group_size,
        NUM_GROUPS=num_groups,
        ROWS_PER_PROGRAM=_FORWARD_ROWS_PER_PROGRAM,
        BLOCK=block,
        num_warps=4,
    )
    return y, rstd


@_forward_op.register_fake
def _(x, weight, eps, group_size):
    return torch.empty_like(x), x.new_empty(
        x.shape[0], x.shape[1] // group_size, dtype=torch.float32
    )


@torch.library.custom_op(
    "torchtitan_flash_next::grouped_offset_rms_norm_backward", mutates_args=()
)
def _backward_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    grad_y: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_rows, channels = x.shape
    num_groups, grid, block = _launch_shape(
        num_rows, group_size, channels, _BACKWARD_ROWS_PER_PROGRAM
    )
    grad_y = grad_y.contiguous()
    grad_x = torch.empty_like(x)
    weight_partial = torch.empty(
        grid[0], channels, device=x.device, dtype=torch.float32
    )
    _backward_kernel[grid](
        x,
        weight,
        rstd,
        grad_y,
        grad_x,
        weight_partial,
        num_rows,
        CHANNELS=channels,
        GROUP=group_size,
        NUM_GROUPS=num_groups,
        ROWS_PER_PROGRAM=_BACKWARD_ROWS_PER_PROGRAM,
        BLOCK=block,
        num_warps=4,
    )
    return grad_x, weight_partial.sum(0).to(weight.dtype)


@_backward_op.register_fake
def _(x, weight, rstd, grad_y, group_size):
    return torch.empty_like(x), torch.empty_like(weight)


def _setup_context(ctx, inputs, output):
    x, weight, _eps, group_size = inputs
    _y, rstd = output
    ctx.save_for_backward(x, weight, rstd)
    ctx.group_size = group_size


def _autograd_backward(ctx, grad_y, _grad_rstd):
    x, weight, rstd = ctx.saved_tensors
    grad_x, grad_weight = _backward_op(x, weight, rstd, grad_y, ctx.group_size)
    return grad_x, grad_weight, None, None


_forward_op.register_autograd(_autograd_backward, setup_context=_setup_context)


def grouped_offset_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, *, eps: float, group_size: int
) -> torch.Tensor:
    """``(1 + weight) * rmsnorm`` over each ``group_size`` slice of the last dim."""
    channels = x.shape[-1]
    assert channels % group_size == 0 and weight.shape == (channels,)
    y, _rstd = _forward_op(
        x.reshape(-1, channels).contiguous(), weight, eps, group_size
    )
    return y.view(x.shape)
