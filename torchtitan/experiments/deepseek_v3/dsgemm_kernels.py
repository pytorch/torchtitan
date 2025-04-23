# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# This module contains code to support working with DeepGEMM:
# DeepGEMM: clean and efficient FP8 GEMM kernels with fine-grained scaling
# https://github.com/deepseek-ai/DeepGEMM


# Groupwise dynamic_quant kernel from DeepSeek-AI, with modifications:
# https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py

from typing import Tuple

import torch
import triton
import triton.language as tl

__all__ = ["groupwise_activation_quant"]


@triton.jit
def grid_stride_quant_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    n_rows,
    n_cols,
    stride_xr,
    stride_xc,
    stride_yr,
    stride_yc,
    stride_s,
    BLOCK_SIZE: tl.constexpr,
    CLAMP_VALUE: tl.constexpr = 448.0,
):
    """
    Grid stride quantization kernel for ds 1x128 format, that processes entire rows.

    Use a grid stride loop pattern where:
    - Each program instance processes one row
    - Within each row, a grid stride loop processes all blocks in that row

    Args:
        x_ptr : Pointer to the input tensor.
        y_ptr : Pointer to the output tensor where quantized values will be stored.
        s_ptr : Pointer to the output tensor where scaling factors will be stored.
        n_rows (int): Number of rows in the tensor.
        n_cols (int): Number of columns in the tensor.
        stride_xr, stride_xc: Row and column strides for input tensor.
        stride_yr, stride_yc: Row and column strides for output tensor.

        stride_s: Stride for scaling factors tensor.

        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.
        CLAMP_VALUE (tl.constexpr): The maximum representable value in the target format.

    Returns:
        None
    """
    # Get program ID for row
    row_idx = tl.program_id(axis=0)

    # Skip if our row_idx is out of bounds
    if row_idx < n_rows:

        # Calculate number of blocks in each row
        n_blocks_per_row = tl.cdiv(n_cols, BLOCK_SIZE)

        # Calculate row pointers
        x_row_ptr = x_ptr + row_idx * stride_xr
        y_row_ptr = y_ptr + row_idx * stride_yr
        s_row_ptr = s_ptr + row_idx * (n_cols // BLOCK_SIZE) * stride_s

        # Grid stride loop to process all blocks in this row
        # Each pid will handle multiple blocks in this row
        pid_block = tl.program_id(axis=1)
        n_total_blocks = n_cols // BLOCK_SIZE
        grid_size_block = tl.num_programs(axis=1)

        # Loop through blocks
        for block_idx in range(pid_block, n_total_blocks, grid_size_block):
            # Compute starting column for this block
            col_start = block_idx * BLOCK_SIZE

            col_offs = col_start + tl.arange(0, BLOCK_SIZE)

            x_ptrs = x_row_ptr + col_offs * stride_xc

            x = tl.load(x_ptrs).to(tl.float32)

            # Find maximum value in this block
            x_abs = tl.abs(x)
            block_max = tl.max(x_abs)

            # scaling factor
            s = block_max / CLAMP_VALUE

            # Add small epsilon to avoid division by zero
            s = tl.maximum(s, 1e-9)

            # Quantize values
            y = x / s

            # Convert to target dtype
            y = y.to(y_ptr.dtype.element_ty)

            # Store quantized values
            y_ptrs = y_row_ptr + col_offs * stride_yc
            tl.store(y_ptrs, y)

            # Store scaling factor for this block
            s_ptr_idx = s_row_ptr + block_idx * stride_s
            tl.store(s_ptr_idx, s)


def groupwise_activation_quant(
    x: torch.Tensor,
    block_size: int = 128,
    switching_size=2048,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Autodispatch wrapper for ds input format (1x128)
    This wrapper will auto select the optimal kernel for fastest groupwise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Last dimension size must be divisible by block_size.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
        switching size (int, optional): The row size at which point we toggle between simple rowwise and gride stride loop kernel.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    # Make input contiguous if not already
    if not x.is_contiguous():
        x = x.contiguous()

    # Validate input
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"

    # Get shape information
    shape = x.shape
    n_rows = x.numel() // shape[-1]  # Total rows considering all batch dimensions
    n_cols = shape[-1]  # Width of each row
    n_blocks_per_row = n_cols // block_size

    # Create output tensors
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s_shape = list(shape[:-1]) + [n_blocks_per_row]
    s = x.new_empty(s_shape, dtype=torch.float32)

    # toggle between dynamic and grid stride loop kernels
    if n_rows < switching_size:
        grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
        dynamic_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
        return y, s

    # otherwise use grid stride loop kernel

    # Calculate strides
    # For 2D view of the tensor (rows × columns)
    stride_xr = shape[-1]  # Stride between rows in x
    stride_xc = 1  # Stride between columns in x
    stride_yr = shape[-1]  # Stride between rows in y
    stride_yc = 1  # Stride between columns in y
    stride_s = 1  # Stride for scaling factors

    # n_threads_per_row is determined as a fraction of the total blocks needed per row
    threads_per_sm = 512  # 1024
    n_threads_per_row = min(n_blocks_per_row, max(1, threads_per_sm // n_rows))

    # Launch kernel with 2D grid: (rows, threads_per_row)
    grid = (n_rows, n_threads_per_row)

    grid_stride_quant_kernel[grid](
        x.reshape(-1, shape[-1]),
        y.reshape(-1, shape[-1]),
        s.reshape(-1, n_blocks_per_row),
        n_rows,
        n_cols,
        stride_xr,
        stride_xc,
        stride_yr,
        stride_yc,
        stride_s,
        BLOCK_SIZE=block_size,
        CLAMP_VALUE=448.0,
    )

    return y, s


# --------  DeepSeek original kernel ---------------------------------
@triton.jit
def dynamic_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.

    Args:
        x_ptr (triton.Pointer): Pointer to the input tensor.
        y_ptr (triton.Pointer): Pointer to the output tensor where quantized values will be stored.
        s_ptr (triton.Pointer): Pointer to the output tensor where scaling factors will be stored.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

    Returns:
        None
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)
