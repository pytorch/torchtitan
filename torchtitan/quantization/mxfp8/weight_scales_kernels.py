# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Triton kernels loaded by the MXFP8 ops only when they execute."""

import triton
import triton.language as tl


@triton.jit
def transpose_mxfp8_weight_scales_kernel(
    input_ptr,
    output_ptr,
    NUM_WEIGHT_ROWS: tl.constexpr,
    NUM_WEIGHT_COLS: tl.constexpr,
    NUM_BYTES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tile = offsets // 512
    within_tile = offsets % 512
    # For an [N, K] weight, FPROP scales are [N, K/32] and DGRAD scales
    # are [K, N/32]. Each layout pads rows to 128 and columns to 4, so
    # their padded shapes can differ even though their byte counts match.
    # Decode output offsets using DGRAD's column-tile count; source offsets
    # below use FPROP's own column-tile count.
    num_output_col_tiles = tl.cdiv(NUM_WEIGHT_ROWS, 128)
    # SWIZZLE_32_4_4 packs each 128x4 tile as [32, 4, 4]: the
    # middle axis selects the group of 32 rows, the last the scale column.
    row = (
        (tile // num_output_col_tiles) * 128
        + ((within_tile % 16) // 4) * 32
        + within_tile // 16
    )
    col = (tile % num_output_col_tiles) * 4 + within_tile % 4

    # A transposed square tile copies one source scale to 32 output rows.
    source_row = col * 32
    source_col = row // 32
    source_tile = (source_row // 128) * tl.cdiv(NUM_WEIGHT_COLS, 128) + (
        source_col // 4
    )
    source_offset = (
        source_tile * 512
        + (source_row % 32) * 16
        + ((source_row % 128) // 32) * 4
        + source_col % 4
    )
    # Valid DGRAD coordinates map inside the logical FPROP scale matrix.
    # Masked loads return zero for DGRAD padding, which we still store:
    # destination padding is rebuilt, never copied from source padding.
    valid = (
        (offsets < NUM_BYTES) & (row < NUM_WEIGHT_COLS) & (col < NUM_WEIGHT_ROWS // 32)
    )
    value = tl.load(input_ptr + source_offset, mask=valid, other=0)
    tl.store(output_ptr + offsets, value, mask=offsets < NUM_BYTES)
