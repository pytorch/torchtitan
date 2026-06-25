# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_segments_fp32_to_bf16_kernel(
    input_ptrs,
    tensor_indices,
    src_offsets,
    numels,
    dst_offsets,
    output,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    chunk_id = tl.program_id(0)
    segment_id = tl.program_id(1)
    offsets = chunk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    tensor_index = tl.load(tensor_indices + segment_id)
    src_offset = tl.load(src_offsets + segment_id)
    numel = tl.load(numels + segment_id)
    dst_offset = tl.load(dst_offsets + segment_id)
    mask = offsets < numel

    src_base_i64 = tl.load(input_ptrs + tensor_index)
    src_base = src_base_i64.to(tl.pointer_type(tl.float32))
    values = tl.load(src_base + src_offset + offsets, mask=mask, other=0.0)
    tl.store(output + dst_offset + offsets, values, mask=mask)


@triton.jit
def _pack_segments_bf16_to_fp32_kernel(
    input_ptrs,
    tensor_indices,
    src_offsets,
    numels,
    dst_offsets,
    output,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    chunk_id = tl.program_id(0)
    segment_id = tl.program_id(1)
    offsets = chunk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    tensor_index = tl.load(tensor_indices + segment_id)
    src_offset = tl.load(src_offsets + segment_id)
    numel = tl.load(numels + segment_id)
    dst_offset = tl.load(dst_offsets + segment_id)
    mask = offsets < numel

    src_base_i64 = tl.load(input_ptrs + tensor_index)
    src_base = src_base_i64.to(tl.pointer_type(tl.bfloat16))
    values = tl.load(src_base + src_offset + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    tl.store(output + dst_offset + offsets, values, mask=mask)


@triton.jit
def _pack_segments_fp32_to_fp32_kernel(
    input_ptrs,
    tensor_indices,
    src_offsets,
    numels,
    dst_offsets,
    output,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    chunk_id = tl.program_id(0)
    segment_id = tl.program_id(1)
    offsets = chunk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    tensor_index = tl.load(tensor_indices + segment_id)
    src_offset = tl.load(src_offsets + segment_id)
    numel = tl.load(numels + segment_id)
    dst_offset = tl.load(dst_offsets + segment_id)
    mask = offsets < numel

    src_base_i64 = tl.load(input_ptrs + tensor_index)
    src_base = src_base_i64.to(tl.pointer_type(tl.float32))
    values = tl.load(src_base + src_offset + offsets, mask=mask, other=0.0)
    tl.store(output + dst_offset + offsets, values, mask=mask)


@triton.jit
def _pack_segments_bf16_to_bf16_kernel(
    input_ptrs,
    tensor_indices,
    src_offsets,
    numels,
    dst_offsets,
    output,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    chunk_id = tl.program_id(0)
    segment_id = tl.program_id(1)
    offsets = chunk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    tensor_index = tl.load(tensor_indices + segment_id)
    src_offset = tl.load(src_offsets + segment_id)
    numel = tl.load(numels + segment_id)
    dst_offset = tl.load(dst_offsets + segment_id)
    mask = offsets < numel

    src_base_i64 = tl.load(input_ptrs + tensor_index)
    src_base = src_base_i64.to(tl.pointer_type(tl.bfloat16))
    values = tl.load(src_base + src_offset + offsets, mask=mask, other=0.0)
    tl.store(output + dst_offset + offsets, values, mask=mask)


def pack_segments_into_flat_buffer_triton(
    inputs: list[torch.Tensor],
    tensor_indices: Sequence[int],
    src_offsets: Sequence[int],
    numels: Sequence[int],
    dst_offsets: Sequence[int],
    output: torch.Tensor,
    *,
    block_size: int = 1024,
    num_warps: int = 4,
) -> list[torch.Tensor] | None:
    """Pack flat source segments into ``output`` with one Triton kernel.

    ``inputs`` are flat contiguous views. Segment metadata is in elements:
    input tensor index, source offset, length, and destination offset.
    """
    if torch.compiler.is_compiling() or block_size <= 0:
        return None
    if output.dim() != 1 or output.device.type != "cuda" or not output.is_contiguous():
        return None
    if not (
        len(tensor_indices)
        == len(src_offsets)
        == len(numels)
        == len(dst_offsets)
    ):
        raise ValueError("Segment descriptor lists must have the same length.")
    if not inputs:
        if len(tensor_indices) == 0:
            return []
        return None

    device = output.device
    input_dtype = inputs[0].dtype
    if input_dtype not in (torch.float32, torch.bfloat16):
        return None
    if output.dtype not in (torch.float32, torch.bfloat16):
        return None
    for tensor in inputs:
        if tensor.device != device or tensor.dtype != input_dtype:
            return None
        if tensor.dim() != 1 or not tensor.is_contiguous():
            return None

    nsegments = len(tensor_indices)
    max_segment_numel = max(numels, default=0)
    input_ptrs = torch.tensor(
        [tensor.data_ptr() for tensor in inputs],
        dtype=torch.int64,
        device=device,
    )
    tensor_indices_t = torch.tensor(tensor_indices, dtype=torch.int64, device=device)
    src_offsets_t = torch.tensor(src_offsets, dtype=torch.int64, device=device)
    numels_t = torch.tensor(numels, dtype=torch.int64, device=device)
    dst_offsets_t = torch.tensor(dst_offsets, dtype=torch.int64, device=device)
    scratch = [
        input_ptrs,
        tensor_indices_t,
        src_offsets_t,
        numels_t,
        dst_offsets_t,
        *inputs,
    ]

    if output.numel() == 0 or nsegments == 0 or max_segment_numel == 0:
        return scratch

    grid = (triton.cdiv(max_segment_numel, block_size), nsegments)
    if input_dtype == torch.float32 and output.dtype == torch.bfloat16:
        _pack_segments_fp32_to_bf16_kernel[grid](
            input_ptrs,
            tensor_indices_t,
            src_offsets_t,
            numels_t,
            dst_offsets_t,
            output,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    elif input_dtype == torch.bfloat16 and output.dtype == torch.float32:
        _pack_segments_bf16_to_fp32_kernel[grid](
            input_ptrs,
            tensor_indices_t,
            src_offsets_t,
            numels_t,
            dst_offsets_t,
            output,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    elif input_dtype == torch.float32 and output.dtype == torch.float32:
        _pack_segments_fp32_to_fp32_kernel[grid](
            input_ptrs,
            tensor_indices_t,
            src_offsets_t,
            numels_t,
            dst_offsets_t,
            output,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    elif input_dtype == torch.bfloat16 and output.dtype == torch.bfloat16:
        _pack_segments_bf16_to_bf16_kernel[grid](
            input_ptrs,
            tensor_indices_t,
            src_offsets_t,
            numels_t,
            dst_offsets_t,
            output,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    else:
        return None
    return scratch


__all__ = [
    "pack_segments_into_flat_buffer_triton",
]
