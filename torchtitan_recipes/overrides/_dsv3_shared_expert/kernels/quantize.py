# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""TorchAO RCEIL helpers for the fused SwiGLU boundaries.

E8M0 byte zero uses reciprocal 2**127. A nonfinite block maximum produces
scale byte 255 and NaN payloads, matching the public TorchAO CUDA quantizer.
"""

import torch
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
import triton.language as tl
from triton.experimental.gluon.language.nvidia.blackwell import fence_async_shared, tma

from .descriptors import _matrix


@gluon.jit
def _scale(amax):
    # torchao RCEIL: E8M0 = round-up(amax / 448), reciprocal = 2**(127 - E).
    biased = gl.inline_asm_elementwise(
        "cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;",
        "=h,r",
        [amax * (1.0 / 448.0)],
        dtype=gl.uint16,
        is_pure=True,
        pack=1,
    ).to(gl.uint8)
    biased = gl.where(amax < float("inf"), biased, 255).to(gl.uint8)
    exponent = biased.to(gl.int32)
    reciprocal = gl.where(
        exponent == 254,
        2.0**-127,
        ((254 - exponent) << 23).to(gl.float32, bitcast=True),
    )
    reciprocal = gl.where(exponent == 255, float("nan"), reciprocal)
    return reciprocal, biased


@gluon.jit
def _blocked(row, block, BLOCKS: gl.constexpr):
    # cuBLAS 128x4 scale tiles stored as 32x16 for a [rows, BLOCKS] scale matrix.
    return (
        ((row // 128) * ((BLOCKS + 3) // 4) + block // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + block % 4
    )


@gluon.jit
def _absmax_combine(a, b):
    # Public TorchAO propagates a NaN in either orientation's 32-value block.
    return gl.inline_asm_elementwise(
        "max.NaN.xorsign.abs.bf16x2 $0, $1, $2;",
        "=r,r,r",
        [a, b],
        dtype=gl.int32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _low(words):
    return ((words & 0x7FFF) << 16).to(gl.float32, bitcast=True)


@gluon.jit
def _high(words):
    return (words & 0x7FFF0000).to(gl.float32, bitcast=True)


@gluon.jit
def _bf16_bits(value):
    # Power-of-two reciprocals, including 2**-127, are exact in BF16.
    return value.to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) & 0xFFFF


@gluon.jit
def _cast_row_words(even, odd, reciprocals):
    # Two BF16-pair words (four consecutive columns) -> one word of four E4M3.
    # Power-of-two scaling is exact in BF16 except below 2**-126, where BF16
    # and FP32 products both convert to a signed E4M3 zero.
    even, reciprocals = gl.broadcast(even, reciprocals)
    odd, reciprocals = gl.broadcast(odd, reciprocals)
    return gl.inline_asm_elementwise(
        "{.reg .b32 a, b; .reg .b16 x, y; "
        "mul.rn.bf16x2 a, $1, $3; mul.rn.bf16x2 b, $2, $3; "
        "cvt.rn.satfinite.e4m3x2.bf16x2 x, a; cvt.rn.satfinite.e4m3x2.bf16x2 y, b; "
        "mov.b32 $0, {x, y};}",
        "=r,r,r,r",
        [even, odd, reciprocals],
        dtype=gl.int32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _cast_col_words(w0, w1, w2, w3, reciprocals):
    # Four rows of one column-pair word -> four E4M3 rows of each column.
    w0, reciprocals = gl.broadcast(w0, reciprocals)
    return gl.inline_asm_elementwise(
        "{.reg .b32 a, b, c, d, ab, cd; .reg .b16 x, y, z, w; "
        "mul.rn.bf16x2 a, $2, $6; mul.rn.bf16x2 b, $3, $6; "
        "mul.rn.bf16x2 c, $4, $6; mul.rn.bf16x2 d, $5, $6; "
        "cvt.rn.satfinite.e4m3x2.bf16x2 x, a; cvt.rn.satfinite.e4m3x2.bf16x2 y, b; "
        "cvt.rn.satfinite.e4m3x2.bf16x2 z, c; cvt.rn.satfinite.e4m3x2.bf16x2 w, d; "
        "mov.b32 ab, {x, y}; mov.b32 cd, {z, w}; "
        "prmt.b32 $0, ab, cd, 0x6420; prmt.b32 $1, ab, cd, 0x7531;}",
        "=r,=r,r,r,r,r,r",
        [w0, w1, w2, w3, reciprocals],
        dtype=(gl.int32, gl.int32),
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _quantize(
    tile_smem,
    row_stage,
    col_stage,
    row_desc,
    col_desc,
    row_scale,
    col_scale,
    row_start,
    column_start,
    M: gl.constexpr,
    N: gl.constexpr,
    BM: gl.constexpr,
    ROWWISE: gl.constexpr,
    COLWISE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    ROW_LAYOUT: gl.constexpr = None,
    COL_LAYOUT: gl.constexpr = None,
    ROW_SCALE_LAYOUT: gl.constexpr = None,
    COL_SCALE_LAYOUT: gl.constexpr = None,
):
    WR: gl.constexpr = BM // (32 * gl.num_ctas())
    WC: gl.constexpr = NUM_WARPS // WR
    # Word layouts over the [BM, 32] int32 view of the [BM, 64] BF16 tile:
    # each thread owns whole 32-element blocks in its orientation.
    ROW: gl.constexpr = (
        ROW_LAYOUT
        if ROW_LAYOUT is not None
        else gl.BlockedLayout([1, 32 // WC], [32, 1], [WR, WC], [1, 0])
    )
    COL: gl.constexpr = (
        COL_LAYOUT
        if COL_LAYOUT is not None
        else gl.BlockedLayout([32 // WC, 1], [WC, 32 // WC], [WR, WC], [0, 1])
    )
    ROW_SCALE: gl.constexpr = (
        ROW_SCALE_LAYOUT
        if ROW_SCALE_LAYOUT is not None
        else gl.BlockedLayout([1, 2], [32, 1], [WR, WC], [1, 0])
    )
    COL_SCALE: gl.constexpr = (
        COL_SCALE_LAYOUT
        if COL_SCALE_LAYOUT is not None
        else gl.BlockedLayout([1, 2], [1, 32], [WR, WC], [1, 0])
    )
    words_smem = tile_smem.reinterpret(
        gl.int32,
        [BM, 32],
        gl.NVMMASharedLayout(
            swizzle_byte_width=128,
            element_bitwidth=32,
            rank=2,
            cga_layout=tile_smem.layout.cga_layout,
        ),
    )
    if ROWWISE:
        words = gl.reshape(words_smem.load(ROW), (BM, 2, 16))
        pairs = gl.reduce(words, 2, _absmax_combine)
        low, high = _low(pairs), _high(pairs)
        reciprocal, biased = _scale(
            gl.maximum(low, high, propagate_nan=tl.PropagateNan.ALL)
        )
        bits = _bf16_bits(reciprocal)
        even, odd = gl.split(gl.reshape(words, (BM, 2, 8, 2)))
        packed = gl.expand_dims(
            gl.convert_layout(bits | (bits << 16), gl.SliceLayout(2, even.type.layout)),
            2,
        )
        row_stage.reinterpret(
            gl.int32,
            [BM, 16],
            gl.NVMMASharedLayout(
                swizzle_byte_width=64,
                element_bitwidth=32,
                rank=2,
                cga_layout=row_stage.layout.cga_layout,
            ),
        ).store(gl.reshape(_cast_row_words(even, odd, packed), (BM, 16)))
        rows = row_start + gl.arange(0, BM, layout=gl.SliceLayout(1, ROW_SCALE))
        groups = column_start // 32 + gl.arange(
            0, 2, layout=gl.SliceLayout(0, ROW_SCALE)
        )
        gl.store(
            row_scale
            + _blocked(gl.expand_dims(rows, 1), gl.expand_dims(groups, 0), N // 32),
            gl.convert_layout(biased, ROW_SCALE),
            gl.expand_dims(rows, 1) < M,
        )
    if COLWISE:
        words = gl.reshape(words_smem.load(COL), (BM // 32, 32, 32))
        pairs = gl.reduce(words, 1, _absmax_combine)
        low, high = _low(pairs), _high(pairs)
        low_reciprocal, low_biased = _scale(low)
        high_reciprocal, high_biased = _scale(high)
        packed = _bf16_bits(low_reciprocal) | (_bf16_bits(high_reciprocal) << 16)
        # [block, 8 groups, 4 rows, 32 words] -> four row operands.
        grouped = gl.permute(
            gl.reshape(words, (BM // 32, 8, 2, 2, 32)), (0, 1, 4, 2, 3)
        )
        first, second = gl.split(grouped)
        w0, w2 = gl.split(first)
        w1, w3 = gl.split(second)
        packed = gl.convert_layout(packed, gl.SliceLayout(1, w0.type.layout))
        even_column, odd_column = _cast_col_words(
            w0, w1, w2, w3, gl.expand_dims(packed, 1)
        )
        # [block, group, word j, column parity] -> [64 columns, BM // 4 words].
        tile = gl.permute(gl.join(even_column, odd_column), (2, 3, 0, 1))
        col_stage.reinterpret(
            gl.int32,
            [64, BM // 4],
            gl.NVMMASharedLayout(
                swizzle_byte_width=128,
                element_bitwidth=32,
                rank=2,
                cga_layout=col_stage.layout.cga_layout,
            ),
        ).store(gl.reshape(tile, (64, BM // 4)))
        biased = gl.reshape(gl.join(low_biased, high_biased), (BM // 32, 64))
        groups = row_start // 32 + gl.arange(
            0, BM // 32, layout=gl.SliceLayout(1, COL_SCALE)
        )
        columns = column_start + gl.arange(0, 64, layout=gl.SliceLayout(0, COL_SCALE))
        gl.store(
            col_scale
            + _blocked(gl.expand_dims(columns, 0), gl.expand_dims(groups, 1), M // 32),
            gl.convert_layout(biased, COL_SCALE),
            gl.expand_dims(groups, 1) < M // 32,
        )
    fence_async_shared()
    gl.barrier()
    if ROWWISE:
        tma.async_store(row_desc, [row_start, column_start], row_stage)
    if COLWISE:
        tma.async_store(col_desc, [column_start, row_start], col_stage)


@gluon.jit
def _quantize_values_kernel(
    input, row, column, row_scale, column_scale, M: gl.constexpr, N: gl.constexpr
):
    layout: gl.constexpr = gl.BlockedLayout([1, 2], [4, 8], [4, 1], [1, 0])
    rows = gl.program_id(0) * 128 + gl.arange(0, 128, layout=gl.SliceLayout(1, layout))
    columns = gl.program_id(1) * 64 + gl.arange(0, 64, layout=gl.SliceLayout(0, layout))
    values = gl.load(input + rows[:, None] * N + columns[None, :])
    tile = gl.allocate_shared_memory(
        gl.bfloat16,
        (128, 64),
        gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2),
    )
    row_stage = gl.allocate_shared_memory(gl.float8e4nv, (128, 64), row.layout)
    column_stage = gl.allocate_shared_memory(gl.float8e4nv, (64, 128), column.layout)
    tile.store(values)
    _quantize(
        tile,
        row_stage,
        column_stage,
        row,
        column,
        row_scale,
        column_scale,
        gl.program_id(0) * 128,
        gl.program_id(1) * 64,
        M,
        N,
        128,
        True,
        True,
        4,
    )
    tma.store_wait(0)


def quantize_values(values):
    """Exercise the exact epilogue quantizer over arbitrary BF16 raw patterns."""
    rows, columns = values.shape
    row = values.new_empty(values.shape, dtype=torch.float8_e4m3fn)
    column = values.new_empty((columns, rows), dtype=torch.float8_e4m3fn)
    row_scale = values.new_empty((rows * columns // 32,), dtype=torch.float8_e8m0fnu)
    column_scale = torch.empty_like(row_scale)
    _quantize_values_kernel[(rows // 128, columns // 64)](
        values,
        _matrix(row, rows, columns, 128, 64),
        _matrix(column, columns, rows, 64, 128),
        row_scale.view(torch.uint8),
        column_scale.view(torch.uint8),
        rows,
        columns,
        num_warps=4,
    )
    return row, column.t(), row_scale, column_scale
