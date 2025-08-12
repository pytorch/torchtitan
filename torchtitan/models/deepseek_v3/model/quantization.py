# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchtitan.tools.logging import logger

# Fixed block size of 128x128 as specified in the algorithm
BLOCK_SIZE = 128


def calculate_scale_shape(
    weight: torch.Tensor, BLOCK_SIZE: int = BLOCK_SIZE
) -> torch.Size:
    # Calculate the scale tensor shape
    orig_shape = weight.shape

    # Calculate number of blocks needed
    block_rows = (orig_shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_cols = (orig_shape[1] + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Verify scale_inv shape matches expected block dimensions
    expected_scale_shape = torch.Size((block_rows, block_cols))

    return expected_scale_shape


def dequantize_from_fp8(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    dtype=torch.bfloat16,
    BLOCK_SIZE: int = BLOCK_SIZE,
) -> torch.Tensor:
    # Convert to float32 for computation
    float_weight = weight.to(torch.float32)
    # Get original dimensions
    orig_shape = weight.shape

    # Verify scale_inv shape matches expected block dimensions
    expected_scale_shape = calculate_scale_shape(weight, BLOCK_SIZE)
    block_rows, block_cols = expected_scale_shape
    if scale_inv.shape != expected_scale_shape:
        logger.warning(
            f"scale_inv shape {scale_inv.shape} doesn't match expected shape {expected_scale_shape}"
        )

    # NOTE: When processing large models on-the-fly, misalignment between block boundaries
    # and DTensor local shape partitioning can lead to silent numerical inaccuracies.
    dequantized = float_weight.detach().clone().to(dtype=dtype)

    # Apply scaling factors to each block
    for i in range(block_rows):
        row_start = i * BLOCK_SIZE
        row_end = min(row_start + BLOCK_SIZE, orig_shape[0])

        for j in range(block_cols):
            col_start = j * BLOCK_SIZE
            col_end = min(col_start + BLOCK_SIZE, orig_shape[1])

            # Get the block
            block = float_weight[row_start:row_end, col_start:col_end]

            scale = scale_inv[i, j]
            block = block * scale

            # Explicitly convert block to dtype
            block_converted = block.to(dtype=torch.float32)
            # Store the dequantized block
            dequantized[row_start:row_end, col_start:col_end] = block_converted

    return dequantized
