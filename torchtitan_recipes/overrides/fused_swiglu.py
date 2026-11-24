# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Bundled fused SwiGLU activation override.

``fused_swiglu`` replaces every ``SwiGLU`` activation selected by the override
framework, including dense feed-forwards, dist-GEMM feed-forwards, and grouped
experts. Projection implementations remain unchanged.

TODO(acisseJZhong): Revisit deprecating this override once DistMoE provides its
own fused SwiGLU and DeepEP/HybridEP are retired. Local compile can then cover
dense feed-forwards and the reference expert-parallel dispatcher.
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
import triton
import triton.language as tl

from torchtitan.config import derive, override
from torchtitan.models.common.activation import BinaryActivationFn, SwiGLU
from torchtitan.tools.utils import device_type

__all__ = [
    "FusedSwiGLU",
    "fused_swiglu",
    "silu_and_mul_backward_kernel_",
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
    NUM_ROWS,  # Not constexpr: routed row counts change every step.
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
    offsets,
    NUM_ROWS,  # Not constexpr: routed row counts change every step.
    NUM_COLS: tl.constexpr,
    NUM_OFFSETS: tl.constexpr,
    HAS_OFFSETS: tl.constexpr,
    GRAD_OUT_ROW_STRIDE: tl.constexpr,
    GRAD_OUT_COL_STRIDE: tl.constexpr,
    GATE_ROW_STRIDE: tl.constexpr,
    GATE_COL_STRIDE: tl.constexpr,
    UP_ROW_STRIDE: tl.constexpr,
    UP_COL_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    """Backward for ``_silu_and_mul_forward_kernel`` over defined rows.

    Stores the gradients over ``gate`` and ``up``: each program loads its tiles
    before storing, and tiles do not overlap.
    """
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
        gate + rows[:, None] * GATE_ROW_STRIDE + cols[None, :] * GATE_COL_STRIDE,
        grad_values * up_values * silu_grad,
        mask=mask,
    )
    tl.store(
        up + rows[:, None] * UP_ROW_STRIDE + cols[None, :] * UP_COL_STRIDE,
        grad_values * silu,
        mask=mask,
    )


def silu_and_mul_forward_kernel(
    gate_up: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``silu(gate) * up`` from ``gate_up [R, 2, F]``, with optional row offsets."""
    if offsets is not None and offsets.numel() == 0:
        raise ValueError("offsets must be non-empty when provided.")
    # Strided views: the kernel reads both halves of gate_up in place.
    gate, up = gate_up.unbind(-2)
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


def silu_and_mul_backward_kernel_(
    grad_out: torch.Tensor,
    gate_up: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> None:
    """Overwrite ``gate_up [R, 2, F]`` with its gradient."""
    if offsets is not None and offsets.numel() == 0:
        raise ValueError("offsets must be non-empty when provided.")
    gate, up = gate_up.unbind(-2)

    block_m = _SILU_AND_MUL_BLOCK_M
    block_n = min(_MAX_BLOCK_N, triton.next_power_of_2(gate.shape[1]))
    grid = (triton.cdiv(gate.shape[0], block_m), triton.cdiv(gate.shape[1], block_n))
    _silu_and_mul_backward_kernel[grid](
        grad_out,
        gate,
        up,
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
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
    )


@torch.library.custom_op(
    "torchtitan::silu_and_mul",
    mutates_args=(),
    device_types=device_type,
)
def silu_and_mul_op(
    gate_up: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``silu(gate) * up`` from ``gate_up [R, 2, F]``, with optional row offsets."""
    return silu_and_mul_forward_kernel(gate_up, offsets)


@silu_and_mul_op.register_fake
def silu_and_mul_op_fake(
    gate_up: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    return gate_up.new_empty(gate_up.shape[0], gate_up.shape[2])


@torch.library.custom_op(
    "torchtitan::silu_and_mul_backward_",
    mutates_args=("gate_up",),
    device_types=device_type,
)
def silu_and_mul_backward_op_(
    grad_out: torch.Tensor,
    gate_up: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> None:
    """Overwrite ``gate_up`` with its gradient for ``torchtitan::silu_and_mul``."""
    silu_and_mul_backward_kernel_(grad_out, gate_up, offsets)


@silu_and_mul_backward_op_.register_fake
def silu_and_mul_backward_op_fake(
    grad_out: torch.Tensor,
    gate_up: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> None:
    return None


def silu_and_mul_autograd_backward(ctx, grad_out):
    if ctx.has_offsets:
        gate_up, offsets = ctx.saved_tensors
    else:
        (gate_up,) = ctx.saved_tensors
        offsets = None
    # Saves a [R, 2, F] buffer. Nothing reads gate_up after this: its producer,
    # w13, needs only its input and weight for backward.
    silu_and_mul_backward_op_(grad_out, gate_up, offsets)
    return gate_up, None


def silu_and_mul_setup_context(ctx, inputs, output):
    gate_up = inputs[0]
    if gate_up.is_leaf and gate_up.requires_grad:
        raise ValueError(
            "torchtitan::silu_and_mul overwrites gate_up with its gradient in "
            "backward, so gate_up must be an autograd intermediate such as "
            "w13's output, not a leaf tensor."
        )
    offsets = inputs[1] if len(inputs) > 1 else None
    ctx.has_offsets = offsets is not None
    if offsets is None:
        ctx.save_for_backward(gate_up)
    else:
        ctx.save_for_backward(gate_up, offsets)


silu_and_mul_op.register_autograd(
    silu_and_mul_autograd_backward, setup_context=silu_and_mul_setup_context
)


class FusedSwiGLU(BinaryActivationFn):
    """SwiGLU activation implemented by the fused Triton operation.

    Backward overwrites ``gate_up`` with its gradient instead of allocating a
    new ``[T, 2, F]`` tensor. So ``gate_up`` must be ``w13``'s output, whose
    backward needs only its input and weight, and no other op may save it:
    saved-tensor hooks (remat, activation offload) skip autograd's version
    check, so a second saver would silently read the gradient. For the same
    reason ``gate_up`` can't be backpropagated through twice
    (``retain_graph=True``).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BinaryActivationFn.Config):
        pass

    def __init__(self, config: Config) -> None:
        pass

    def __call__(
        self, gate_up: torch.Tensor, *, offsets: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Eager: the compiler can't fuse into the Triton op, so the base's compiled
        # region would only add graphs.
        if offsets is not None:
            return silu_and_mul_op(gate_up, offsets)
        return _silu_and_mul_dense(gate_up)

    def _activation_fn(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        # Only satisfies the abstract base: the __call__ above never calls it.
        return torch.nn.functional.silu(gate) * up


def _silu_and_mul_dense(gate_up: torch.Tensor) -> torch.Tensor:
    # TODO(pianpwk): Migrate this local_map workaround to a custom op SPMD
    # propagation rule registration system.
    gate_up_type = (
        {"dp": spmd.V, "cp": spmd.V, "tp": spmd.V},
        spmd.PartitionSpec(("dp", "cp"), None, "tp"),
    )
    activation_type = (
        {"dp": spmd.V, "cp": spmd.V, "tp": spmd.V},
        spmd.PartitionSpec(("dp", "cp"), "tp"),
    )
    return spmd.local_map(in_types=(gate_up_type,), out_types=activation_type)(
        lambda gate_up: silu_and_mul_op(
            gate_up.reshape(-1, *gate_up.shape[-2:])
        ).reshape(*gate_up.shape[:-2], gate_up.shape[-1])
    )(gate_up)


@override(
    target=SwiGLU.Config,
    exact=True,
    description="Fuse the SwiGLU SiLU and multiply operations with Triton.",
)
def fused_swiglu(cfg: SwiGLU.Config) -> FusedSwiGLU.Config:
    return derive(cfg, FusedSwiGLU.Config)
