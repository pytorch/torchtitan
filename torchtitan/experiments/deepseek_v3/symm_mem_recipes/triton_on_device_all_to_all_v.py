# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from .triton_barrier import blockwise_barrier
from .triton_utils import sync_threads


@triton.jit
def _exchange_row_offsets(
    split_sizes_ptrs,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCKS_PER_REMOTE_RANK: tl.constexpr,
):
    remote_rank = tl.program_id(0) // BLOCKS_PER_REMOTE_RANK

    # split_sizes_ptr for all ranks
    # All these vector stacks into split_sizes_matrix
    split_sizes_ptrs = split_sizes_ptrs.to(tl.pointer_type(tl.uint64))

    # split_sizes_matrix[remote_rank, :]
    input_split_sizes_ptr = tl.load(split_sizes_ptrs + remote_rank).to(
        tl.pointer_type(tl.int64)
    )

    offsets_ = tl.arange(0, world_size)
    input_split_sizes = tl.load(
        input_split_sizes_ptr + offsets_, mask=offsets_ <= rank, other=0
    )

    num_rows = tl.load(input_split_sizes_ptr + rank)
    input_row_offset = tl.sum(input_split_sizes) - num_rows

    # split_sizes_matrix[:, rank]
    output_split_sizes_ptrs = (
        tl.load(split_sizes_ptrs + offsets_).to(tl.pointer_type(tl.int64)) + rank
    )
    output_split_sizes = tl.load(
        output_split_sizes_ptrs, mask=offsets_ <= remote_rank, other=0
    )
    output_row_offset = tl.sum(output_split_sizes) - num_rows

    return input_row_offset, output_row_offset, num_rows


@triton.jit
def on_device_all_to_all_v_kernel(
    output_ptr,
    output_splits_ptr,
    input_ptrs,
    input_splits_ptr,
    signal_pad_ptrs,
    dim: tl.constexpr,  # Separate dim for easier vectorization
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCKS_PER_REMOTE_RANK: tl.constexpr,
    UNROLL_FACTOR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size, sem="relaxed")
    sync_threads()

    remote_rank = tl.program_id(0) // BLOCKS_PER_REMOTE_RANK
    block_offset = tl.program_id(0) % BLOCKS_PER_REMOTE_RANK

    input_row_offset, output_row_offset, num_rows = _exchange_row_offsets(
        input_splits_ptr, rank, world_size, BLOCKS_PER_REMOTE_RANK
    )

    output_splits_ptr = output_splits_ptr.to(tl.pointer_type(tl.uint64))
    if block_offset == 0:
        # Update output_splits
        tl.store(output_splits_ptr + remote_rank, num_rows)

    input_ptr = (
        tl.load(input_ptrs.to(tl.pointer_type(tl.uint64)) + remote_rank).to(
            tl.pointer_type(tl.bfloat16)
        )
        + input_row_offset * dim
    )
    output_ptr = output_ptr + output_row_offset * dim

    outer_loop_step = BLOCK_SIZE * UNROLL_FACTOR
    outer_loop_iters_per_rank = tl.cdiv(
        tl.cdiv(num_rows * dim, outer_loop_step), BLOCKS_PER_REMOTE_RANK
    )
    numel_per_rank = outer_loop_step * outer_loop_iters_per_rank
    offset = numel_per_rank * block_offset
    end = tl.minimum(numel_per_rank * (block_offset + 1), num_rows * dim)

    unroll_region_size = (end - offset) // outer_loop_step * outer_loop_step
    for i in tl.range(offset, offset + unroll_region_size, outer_loop_step):
        datas = []
        for j in tl.range(
            i,
            i + outer_loop_step,
            BLOCK_SIZE,
            loop_unroll_factor=UNROLL_FACTOR,
        ):
            offsets = j + tl.arange(0, BLOCK_SIZE)
            data = tl.load(input_ptr + offsets)
            tl.store(output_ptr + offsets, data)

    offset += unroll_region_size
    while offset < end:
        offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_rows * dim
        data = tl.load(input_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, data, mask=mask)
        offset += BLOCK_SIZE

    sync_threads()
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size, sem="relaxed")
    return


def _on_device_all_to_all_v(
    output: torch.Tensor,
    output_splits: torch.Tensor,
    input: torch.Tensor,
    input_splits: torch.Tensor,
    group: dist.ProcessGroup = dist.group.WORLD,
    BLOCKS_PER_REMOTE_RANK=8,
    UNROLL_FACTOR: int = 8,
    BLOCK_SIZE: int = 16384,
):
    assert output.dim() == 2, f"{output.shape}"
    assert input.dim() == 2, f"{input.shape}"
    assert output.shape[1] == input.shape[1]

    dim = output.shape[1]
    input_hdl = symm_mem.rendezvous(input, group=group)
    input_splits_hdl = symm_mem.rendezvous(input_splits, group=group)

    num_blocks = input_hdl.world_size * BLOCKS_PER_REMOTE_RANK
    kernel = on_device_all_to_all_v_kernel[(num_blocks, 1, 1)](
        output,
        output_splits,
        input_hdl.buffer_ptrs_dev,
        input_splits_hdl.buffer_ptrs_dev,
        input_hdl.signal_pad_ptrs_dev,
        dim=dim,
        rank=input_hdl.rank,
        world_size=input_hdl.world_size,
        BLOCKS_PER_REMOTE_RANK=BLOCKS_PER_REMOTE_RANK,
        UNROLL_FACTOR=UNROLL_FACTOR,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16,
    )
    # log_triton_kernel(kernel)
    return output


class OnDeviceAllToAllV(torch.autograd.Function):
    # A symmetric memory holding the grad_output during backward
    grad_output_buf = None
    # A symmetric memory for exchanges split sizes during both forward and backward
    splits_buf = None
    # Maximum output length (need to be set before use of OnDeviceAllToAllV)
    max_output_len = None

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        input_splits: torch.Tensor,
        group: dist.ProcessGroup = dist.group.WORLD,
    ):
        """
        Args:
            input: input tensor with data for all ranks concatenated.
            input_splits: input splits of shape (group.world_size,)
            group: process group to scope the collective.
        """
        # Initialize input splits buffer (one time only)
        if OnDeviceAllToAllV.splits_buf is None:
            OnDeviceAllToAllV.splits_buf = symm_mem.empty(
                *input_splits.shape,
                dtype=input_splits.dtype,
                device=input_splits.device,
            )

        if OnDeviceAllToAllV.max_output_len is None:
            raise RuntimeError(
                "Please set max output length via `OnDeviceAllToAllV.max_output_len = ...`"
            )

        # Allocate output buffer
        output = input.new_empty(OnDeviceAllToAllV.max_output_len, *input.shape[1:])
        # Allocate output splits tensor
        output_splits = torch.empty_like(input_splits)
        # Copy input splits to the buffer
        OnDeviceAllToAllV.splits_buf.copy_(input_splits)

        # Shuffle input to output
        _on_device_all_to_all_v(
            output, output_splits, input, OnDeviceAllToAllV.splits_buf, group=group
        )

        # Output splits in forward is the input splits in backward
        ctx.save_for_backward(output_splits)
        ctx.group = group
        ctx.input_shape = input.shape
        return output, output_splits

    @staticmethod
    def backward(ctx, grad_output, grad_splits):
        """
        Backward is implemented as a shuffle of the output's gradients to the input.
        Args:
            `grad_output`: output's gradients passed from the downstream.
            `grad_splits`: unused.
        """

        # Initialize grad_output buffer (one time only)
        if OnDeviceAllToAllV.grad_output_buf is None:
            assert (
                OnDeviceAllToAllV.max_output_len is not None
            ), "`max_output_len` not set"
            OnDeviceAllToAllV.grad_output_buf = symm_mem.empty(
                OnDeviceAllToAllV.max_output_len,
                *grad_output.shape[1:],
                dtype=grad_output.dtype,
                device=grad_output.device,
            )

        # TODO: is there a way to tell autograd to feed grad_output directly to
        # our symm_mem buffer?
        OnDeviceAllToAllV.grad_output_buf.narrow(0, 0, grad_output.shape[0]).copy_(
            grad_output
        )

        # Size info
        (grad_output_splits,) = ctx.saved_tensors
        OnDeviceAllToAllV.splits_buf.copy_(grad_output_splits)
        grad_input_splits = torch.empty_like(grad_output_splits)  # unused
        grad_input = grad_output.new_empty(*ctx.input_shape)

        # Shuffle gradients back to the input
        _on_device_all_to_all_v(
            grad_input,
            grad_input_splits,
            OnDeviceAllToAllV.grad_output_buf,
            OnDeviceAllToAllV.splits_buf,
            group=ctx.group,
        )
        return grad_input, None, None


# Alias
on_device_all_to_all_v = OnDeviceAllToAllV.apply
