# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SwiGLU and MXFP8 fusion candidates for the single-layer graph experiment."""

import torch
import triton

from torchao.prototype.mx_formats.kernels import _triton_calculate_scale_rceil
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

# Shape suffix legend:
#   T = token rows, F = shared-expert hidden features, D = quantized features

_FORWARD_ROW_LAYOUT = gl.BlockedLayout([1, 1, 16], [16, 1, 2], [2, 2, 1], [2, 0, 1])
_FORWARD_COL_LAYOUT = gl.BlockedLayout([16, 1, 1], [2, 1, 16], [1, 2, 2], [0, 2, 1])
_BACKWARD_ROW_LAYOUT = gl.BlockedLayout([1, 1, 8], [8, 1, 4], [4, 1, 1], [2, 0, 1])
_BACKWARD_COL_LAYOUT = gl.BlockedLayout([8, 1, 1], [4, 1, 8], [1, 1, 4], [0, 2, 1])


@gluon.jit
def _store_mxfp8_tile(
    values,
    row_q,
    col_q,
    row_scale,
    col_scale,
    row_start,
    col_start,
    NUM_ROWS: gl.constexpr,
    NUM_COLS: gl.constexpr,
    COL_GROUPS: gl.constexpr,
    ROW_LAYOUT: gl.constexpr,
    COL_LAYOUT: gl.constexpr,
):
    row = row_start + gl.arange(
        0, 32, layout=gl.SliceLayout(1, gl.SliceLayout(2, ROW_LAYOUT))
    )
    group = gl.arange(
        0, COL_GROUPS, layout=gl.SliceLayout(0, gl.SliceLayout(2, ROW_LAYOUT))
    )
    within = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(1, ROW_LAYOUT)))
    columns = col_start + group[None, :, None] * 32 + within[None, None, :]
    row_rcp, row_exponent = _triton_calculate_scale_rceil(
        gl.abs(values.to(gl.float32)), axis=2, USE_PTX=True
    )
    normalized_row = values.to(gl.float32) * row_rcp[:, :, None]
    gl.store(
        row_q + row[:, None, None] * NUM_COLS + columns,
        normalized_row.to(gl.float8e4nv),
    )
    scale_rows = row[:, None]
    scale_cols = col_start // 32 + group[None, :]
    row_scale_stride: gl.constexpr = triton.cdiv(NUM_COLS, 128) * 4
    scale_offsets = (
        (scale_rows // 128) * row_scale_stride * 128
        + (scale_cols // 4) * 512
        + (scale_rows % 32) * 16
        + ((scale_rows % 128) // 32) * 4
        + scale_cols % 4
    )
    gl.store(row_scale + scale_offsets, row_exponent)

    # Make each 32-element column group contiguous within a warp before reducing.
    col_values = gl.convert_layout(values, COL_LAYOUT)
    row = row_start + gl.arange(
        0, 32, layout=gl.SliceLayout(1, gl.SliceLayout(2, COL_LAYOUT))
    )
    group = gl.arange(
        0, COL_GROUPS, layout=gl.SliceLayout(0, gl.SliceLayout(2, COL_LAYOUT))
    )
    within = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(1, COL_LAYOUT)))
    columns = col_start + group[None, :, None] * 32 + within[None, None, :]
    col_rcp, col_exponent = _triton_calculate_scale_rceil(
        gl.abs(col_values.to(gl.float32)), axis=0, USE_PTX=True
    )
    normalized_col = col_values.to(gl.float32) * col_rcp[None, :, :]
    gl.store(
        col_q + columns * NUM_ROWS + row[:, None, None],
        normalized_col.to(gl.float8e4nv),
    )
    scale_groups = gl.arange(
        0, COL_GROUPS, layout=gl.SliceLayout(1, gl.SliceLayout(0, COL_LAYOUT))
    )
    scale_within = gl.arange(
        0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(0, COL_LAYOUT))
    )
    scale_rows = col_start + scale_groups[:, None] * 32 + scale_within[None, :]
    scale_cols = row_start // 32
    col_scale_stride: gl.constexpr = triton.cdiv(NUM_ROWS, 128) * 4
    scale_offsets = (
        (scale_rows // 128) * col_scale_stride * 128
        + (scale_cols // 4) * 512
        + (scale_rows % 32) * 16
        + ((scale_rows % 128) // 32) * 4
        + scale_cols % 4
    )
    gl.store(col_scale + scale_offsets, col_exponent)


@gluon.jit
def _swiglu_mxfp8_kernel(
    gate_up,
    grad_out,
    row_q,
    col_q,
    row_scale,
    col_scale,
    NUM_ROWS: gl.constexpr,
    NUM_FEATURES: gl.constexpr,
    BACKWARD: gl.constexpr,
    COL_GROUPS: gl.constexpr,
    ROW_LAYOUT: gl.constexpr,
    COL_LAYOUT: gl.constexpr,
):
    row_start = gl.program_id(0) * 32
    col_start = gl.program_id(1) * (32 * COL_GROUPS)
    row = row_start + gl.arange(
        0, 32, layout=gl.SliceLayout(1, gl.SliceLayout(2, ROW_LAYOUT))
    )
    group = gl.arange(
        0, COL_GROUPS, layout=gl.SliceLayout(0, gl.SliceLayout(2, ROW_LAYOUT))
    )
    within = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(1, ROW_LAYOUT)))
    columns = col_start + group[None, :, None] * 32 + within[None, None, :]
    offsets = row[:, None, None] * (2 * NUM_FEATURES) + columns
    gate = gl.load(gate_up + offsets).to(gl.float32)
    up = gl.load(gate_up + offsets + NUM_FEATURES).to(gl.float32)
    sigmoid = 1.0 / (1.0 + gl.exp(-gate))
    silu = gate * sigmoid
    if BACKWARD:
        grad = gl.load(grad_out + row[:, None, None] * NUM_FEATURES + columns).to(
            gl.float32
        )
        silu_grad = sigmoid * (1.0 + gate * (1.0 - sigmoid))
        # Retain the BF16 rounding performed by the production activation op.
        grad_gate = (grad * up * silu_grad).to(gl.bfloat16)
        grad_up = (grad * silu).to(gl.bfloat16)
        _store_mxfp8_tile(
            grad_gate,
            row_q,
            col_q,
            row_scale,
            col_scale,
            row_start,
            col_start,
            NUM_ROWS,
            2 * NUM_FEATURES,
            COL_GROUPS,
            ROW_LAYOUT,
            COL_LAYOUT,
        )
        _store_mxfp8_tile(
            grad_up,
            row_q,
            col_q,
            row_scale,
            col_scale,
            row_start,
            col_start + NUM_FEATURES,
            NUM_ROWS,
            2 * NUM_FEATURES,
            COL_GROUPS,
            ROW_LAYOUT,
            COL_LAYOUT,
        )
    else:
        activation = (silu * up).to(gl.bfloat16)
        _store_mxfp8_tile(
            activation,
            row_q,
            col_q,
            row_scale,
            col_scale,
            row_start,
            col_start,
            NUM_ROWS,
            NUM_FEATURES,
            COL_GROUPS,
            ROW_LAYOUT,
            COL_LAYOUT,
        )


def _allocate_mxfp8_operands(
    gate_up_T2F: torch.Tensor, *, backward: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_rows, _, num_features = gate_up_T2F.shape
    num_cols = 2 * num_features if backward else num_features
    row_q_TD = gate_up_T2F.new_empty(num_rows, num_cols, dtype=torch.float8_e4m3fn)
    col_q_TD = torch.empty_strided(
        (num_rows, num_cols),
        (1, num_rows),
        device=gate_up_T2F.device,
        dtype=torch.float8_e4m3fn,
    )
    row_scale = gate_up_T2F.new_empty(
        triton.cdiv(num_rows, 128) * 128,
        triton.cdiv(num_cols, 128) * 4,
        dtype=torch.float8_e8m0fnu,
    )
    col_scale = gate_up_T2F.new_empty(
        triton.cdiv(num_cols, 128) * 128,
        triton.cdiv(num_rows, 128) * 4,
        dtype=torch.float8_e8m0fnu,
    )
    return row_q_TD, col_q_TD, row_scale, col_scale


def _validate_gate_up(gate_up_T2F: torch.Tensor) -> None:
    if gate_up_T2F.ndim != 3 or gate_up_T2F.shape[1] != 2:
        raise ValueError("SwiGLU MXFP8 fusion requires a [T, 2, F] input")
    if gate_up_T2F.dtype != torch.bfloat16 or not gate_up_T2F.is_contiguous():
        raise ValueError("SwiGLU MXFP8 fusion requires a contiguous BF16 input")
    if any(
        size <= 0 or size % 128 for size in (gate_up_T2F.shape[0], gate_up_T2F.shape[2])
    ):
        raise ValueError("SwiGLU MXFP8 fusion requires T and F divisible by 128")


@torch.library.custom_op(
    "torchtitan_auto_fuser::swiglu_mxfp8_quantize", mutates_args=(), device_types="cuda"
)
def swiglu_mxfp8_quantize(
    gate_up_T2F: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize the production SwiGLU output in both MXFP8 orientations."""
    _validate_gate_up(gate_up_T2F)
    operands = _allocate_mxfp8_operands(gate_up_T2F, backward=False)
    num_rows, _, num_features = gate_up_T2F.shape
    _swiglu_mxfp8_kernel[(num_rows // 32, num_features // 128)](
        gate_up_T2F,
        gate_up_T2F,
        operands[0],
        operands[1],
        operands[2].view(torch.uint8),
        operands[3].view(torch.uint8),
        NUM_ROWS=num_rows,
        NUM_FEATURES=num_features,
        BACKWARD=False,
        COL_GROUPS=4,
        ROW_LAYOUT=_FORWARD_ROW_LAYOUT,
        COL_LAYOUT=_FORWARD_COL_LAYOUT,
        num_warps=4,
    )
    return operands


@swiglu_mxfp8_quantize.register_fake
def _swiglu_mxfp8_quantize_fake(
    gate_up_T2F: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _allocate_mxfp8_operands(gate_up_T2F, backward=False)


@torch.library.custom_op(
    "torchtitan_auto_fuser::swiglu_backward_mxfp8_quantize",
    mutates_args=(),
    device_types="cuda",
)
def swiglu_backward_mxfp8_quantize(
    grad_out_TF: torch.Tensor, gate_up_T2F: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize the production SwiGLU input gradient in both orientations."""
    _validate_gate_up(gate_up_T2F)
    num_rows, _, num_features = gate_up_T2F.shape
    if (
        grad_out_TF.shape != (num_rows, num_features)
        or grad_out_TF.dtype != torch.bfloat16
        or not grad_out_TF.is_contiguous()
    ):
        raise ValueError(
            "SwiGLU MXFP8 fusion requires a contiguous BF16 [T, F] gradient"
        )
    operands = _allocate_mxfp8_operands(gate_up_T2F, backward=True)
    _swiglu_mxfp8_kernel[(num_rows // 32, num_features // 64)](
        gate_up_T2F,
        grad_out_TF,
        operands[0],
        operands[1],
        operands[2].view(torch.uint8),
        operands[3].view(torch.uint8),
        NUM_ROWS=num_rows,
        NUM_FEATURES=num_features,
        BACKWARD=True,
        COL_GROUPS=2,
        ROW_LAYOUT=_BACKWARD_ROW_LAYOUT,
        COL_LAYOUT=_BACKWARD_COL_LAYOUT,
        num_warps=4,
    )
    return operands


@swiglu_backward_mxfp8_quantize.register_fake
def _swiglu_backward_mxfp8_quantize_fake(
    grad_out_TF: torch.Tensor, gate_up_T2F: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _allocate_mxfp8_operands(gate_up_T2F, backward=True)
