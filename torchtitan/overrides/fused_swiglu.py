# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Fused SwiGLU activation override.

``fused_swiglu`` replaces every ``SwiGLU`` activation selected by the override
framework, including dense feed-forwards, dist-GEMM feed-forwards, and grouped
experts. Projection implementations remain unchanged.
"""

from dataclasses import dataclass
from typing import Any

import spmd_types as spmd
import torch
import triton
import triton.language as tl

from torchtitan.config import derive, override
from torchtitan.models.common.activation import BinaryActivationFn, SwiGLU

__all__ = [
    "FusedSwiGLU",
    "fused_swiglu",
    "silu_and_mul_backward_kernel",
    "silu_and_mul_forward_kernel",
    "silu_and_mul_op",
]


_MAX_BLOCK_N = 2048
_SILU_AND_MUL_BLOCK_M = 4


@triton.jit
def _silu_and_mul_forward_kernel(
    gate,
    up,
    out,
    offsets,
    NUM_ROWS: tl.constexpr,
    NUM_COLS: tl.constexpr,
    NUM_OFFSETS: tl.constexpr,
    HAS_OFFSETS: tl.constexpr,
    GATE_ROW_STRIDE: tl.constexpr,
    GATE_COL_STRIDE: tl.constexpr,
    UP_ROW_STRIDE: tl.constexpr,
    UP_COL_STRIDE: tl.constexpr,
    OUT_ROW_STRIDE: tl.constexpr,
    OUT_COL_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    """Compute ``silu(gate) * up`` for optionally offset-limited rows."""
    row_start = tl.program_id(0).to(tl.int64) * BLOCK_M
    row_limit = NUM_ROWS
    if HAS_OFFSETS:
        row_limit = tl.load(offsets + NUM_OFFSETS - 1)
        if row_start >= row_limit:
            return

    rows = row_start + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < row_limit) & (cols[None, :] < NUM_COLS)

    gate_values = tl.load(
        gate + rows[:, None] * GATE_ROW_STRIDE + cols[None, :] * GATE_COL_STRIDE,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up_values = tl.load(
        up + rows[:, None] * UP_ROW_STRIDE + cols[None, :] * UP_COL_STRIDE,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    silu = gate_values * tl.sigmoid(gate_values)
    tl.store(
        out + rows[:, None] * OUT_ROW_STRIDE + cols[None, :] * OUT_COL_STRIDE,
        silu * up_values,
        mask=mask,
    )


@triton.jit
def _silu_and_mul_backward_kernel(
    grad_out,
    gate,
    up,
    grad_gate,
    grad_up,
    offsets,
    NUM_ROWS: tl.constexpr,
    NUM_COLS: tl.constexpr,
    NUM_OFFSETS: tl.constexpr,
    HAS_OFFSETS: tl.constexpr,
    GRAD_OUT_ROW_STRIDE: tl.constexpr,
    GRAD_OUT_COL_STRIDE: tl.constexpr,
    GATE_ROW_STRIDE: tl.constexpr,
    GATE_COL_STRIDE: tl.constexpr,
    UP_ROW_STRIDE: tl.constexpr,
    UP_COL_STRIDE: tl.constexpr,
    GRAD_GATE_ROW_STRIDE: tl.constexpr,
    GRAD_GATE_COL_STRIDE: tl.constexpr,
    GRAD_UP_ROW_STRIDE: tl.constexpr,
    GRAD_UP_COL_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    """Backward for ``_silu_and_mul_forward_kernel`` over defined rows."""
    row_start = tl.program_id(0).to(tl.int64) * BLOCK_M
    row_limit = NUM_ROWS
    if HAS_OFFSETS:
        row_limit = tl.load(offsets + NUM_OFFSETS - 1)
        if row_start >= row_limit:
            return

    rows = row_start + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < row_limit) & (cols[None, :] < NUM_COLS)

    grad_values = tl.load(
        grad_out
        + rows[:, None] * GRAD_OUT_ROW_STRIDE
        + cols[None, :] * GRAD_OUT_COL_STRIDE,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate_values = tl.load(
        gate + rows[:, None] * GATE_ROW_STRIDE + cols[None, :] * GATE_COL_STRIDE,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up_values = tl.load(
        up + rows[:, None] * UP_ROW_STRIDE + cols[None, :] * UP_COL_STRIDE,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    sigmoid = tl.sigmoid(gate_values)
    silu = gate_values * sigmoid
    silu_grad = sigmoid * (1.0 + gate_values * (1.0 - sigmoid))

    tl.store(
        grad_gate
        + rows[:, None] * GRAD_GATE_ROW_STRIDE
        + cols[None, :] * GRAD_GATE_COL_STRIDE,
        grad_values * up_values * silu_grad,
        mask=mask,
    )
    tl.store(
        grad_up
        + rows[:, None] * GRAD_UP_ROW_STRIDE
        + cols[None, :] * GRAD_UP_COL_STRIDE,
        grad_values * silu,
        mask=mask,
    )


def silu_and_mul_forward_kernel(
    gate: torch.Tensor,
    up: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``silu(gate) * up`` with optional grouped_mm row offsets."""
    if offsets is not None and offsets.numel() == 0:
        raise ValueError("offsets must be non-empty when provided.")
    out = torch.empty_like(gate, memory_format=torch.contiguous_format)

    block_m = _SILU_AND_MUL_BLOCK_M
    block_n = min(_MAX_BLOCK_N, triton.next_power_of_2(gate.shape[1]))
    grid = (triton.cdiv(gate.shape[0], block_m), triton.cdiv(gate.shape[1], block_n))
    _silu_and_mul_forward_kernel[grid](
        gate,
        up,
        out,
        offsets if offsets is not None else gate,
        NUM_ROWS=gate.shape[0],
        NUM_COLS=gate.shape[1],
        NUM_OFFSETS=offsets.numel() if offsets is not None else 0,
        HAS_OFFSETS=offsets is not None,
        GATE_ROW_STRIDE=gate.stride(0),
        GATE_COL_STRIDE=gate.stride(1),
        UP_ROW_STRIDE=up.stride(0),
        UP_COL_STRIDE=up.stride(1),
        OUT_ROW_STRIDE=out.stride(0),
        OUT_COL_STRIDE=out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
    )
    return out


def silu_and_mul_backward_kernel(
    grad_out: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if offsets is not None and offsets.numel() == 0:
        raise ValueError("offsets must be non-empty when provided.")
    grad_gate = torch.empty_like(gate, memory_format=torch.contiguous_format)
    grad_up = torch.empty_like(up, memory_format=torch.contiguous_format)

    block_m = _SILU_AND_MUL_BLOCK_M
    block_n = min(_MAX_BLOCK_N, triton.next_power_of_2(gate.shape[1]))
    grid = (triton.cdiv(gate.shape[0], block_m), triton.cdiv(gate.shape[1], block_n))
    _silu_and_mul_backward_kernel[grid](
        grad_out,
        gate,
        up,
        grad_gate,
        grad_up,
        offsets if offsets is not None else gate,
        NUM_ROWS=gate.shape[0],
        NUM_COLS=gate.shape[1],
        NUM_OFFSETS=offsets.numel() if offsets is not None else 0,
        HAS_OFFSETS=offsets is not None,
        GRAD_OUT_ROW_STRIDE=grad_out.stride(0),
        GRAD_OUT_COL_STRIDE=grad_out.stride(1),
        GATE_ROW_STRIDE=gate.stride(0),
        GATE_COL_STRIDE=gate.stride(1),
        UP_ROW_STRIDE=up.stride(0),
        UP_COL_STRIDE=up.stride(1),
        GRAD_GATE_ROW_STRIDE=grad_gate.stride(0),
        GRAD_GATE_COL_STRIDE=grad_gate.stride(1),
        GRAD_UP_ROW_STRIDE=grad_up.stride(0),
        GRAD_UP_COL_STRIDE=grad_up.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
    )
    return grad_gate, grad_up


@torch.library.custom_op(
    "torchtitan::silu_and_mul",
    mutates_args=(),
    device_types="cuda",
)
def silu_and_mul_op(
    gate: torch.Tensor,
    up: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``silu(gate) * up`` over optionally offset-limited rows."""
    return silu_and_mul_forward_kernel(gate, up, offsets)


@silu_and_mul_op.register_fake
def silu_and_mul_op_fake(
    gate: torch.Tensor,
    up: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(gate, memory_format=torch.contiguous_format)


@torch.library.custom_op(
    "torchtitan::silu_and_mul_backward",
    mutates_args=(),
    device_types="cuda",
)
def silu_and_mul_backward_op(
    grad_out: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gradients for ``torchtitan::silu_and_mul``."""
    return silu_and_mul_backward_kernel(grad_out, gate, up, offsets)


@silu_and_mul_backward_op.register_fake
def silu_and_mul_backward_op_fake(
    grad_out: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(gate, memory_format=torch.contiguous_format),
        torch.empty_like(up, memory_format=torch.contiguous_format),
    )


def silu_and_mul_autograd_backward(ctx, grad_out):
    if ctx.has_offsets:
        gate, up, offsets = ctx.saved_tensors
    else:
        gate, up = ctx.saved_tensors
        offsets = None
    grad_gate, grad_up = silu_and_mul_backward_op(
        grad_out,
        gate,
        up,
        offsets,
    )
    return grad_gate, grad_up, None


def silu_and_mul_setup_context(ctx, inputs, output):
    gate, up = inputs[:2]
    offsets = inputs[2] if len(inputs) > 2 else None
    ctx.has_offsets = offsets is not None
    if offsets is None:
        ctx.save_for_backward(gate, up)
    else:
        ctx.save_for_backward(gate, up, offsets)


silu_and_mul_op.register_autograd(
    silu_and_mul_autograd_backward, setup_context=silu_and_mul_setup_context
)


class FusedSwiGLU(BinaryActivationFn):
    """SwiGLU activation implemented by the fused Triton operation."""

    @dataclass(kw_only=True, slots=True)
    class Config(BinaryActivationFn.Config):
        pass

    def __init__(self, config: Config) -> None:
        pass

    def __call__(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        offsets = kwargs.get("offsets")
        if offsets is not None:
            return silu_and_mul_op(gate, up, offsets)
        return _silu_and_mul_2d(gate, up)


def _silu_and_mul_2d(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    # TODO(pianpwk): Migrate this local_map workaround to a custom op SPMD
    # propagation rule registration system.
    activation_type = (
        {"dp": spmd.V, "cp": spmd.V, "tp": spmd.V},
        spmd.PartitionSpec(("dp", "cp"), "tp"),
    )
    return spmd.local_map(
        in_types=(activation_type, activation_type),
        out_types=activation_type,
    )(
        lambda gate, up: silu_and_mul_op(
            gate.reshape(-1, gate.shape[-1]),
            up.reshape(-1, up.shape[-1]),
        ).reshape(gate.shape)
    )(
        gate, up
    )


@override(
    target=SwiGLU.Config,
    exact=True,
    description="Fuse the SwiGLU SiLU and multiply operations with Triton.",
)
def fused_swiglu(cfg: SwiGLU.Config) -> FusedSwiGLU.Config:
    return derive(cfg, FusedSwiGLU.Config)
