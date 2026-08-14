# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl


_MAX_PROGRAMS = 1024
_BLOCK_SIZE = 4096
_MAX_INT32_INDEXED_ELEMENTS = 2**31 - _MAX_PROGRAMS * _BLOCK_SIZE


@triton.jit
def _accumulate_tensor_statistics_triton(
    value_ptr,
    counts_ptr,
    sums_ptr,
    maximum_ptr,
    enabled_ptr,
    value_count,
    BLOCK_SIZE: tl.constexpr,
    NEEDS_LOOP: tl.constexpr,
    USE_INT64_INDEX: tl.constexpr,
):
    if tl.load(enabled_ptr) == 0:
        return

    nonfinite_count = tl.zeros((), dtype=tl.int32)
    zero_count = tl.zeros((), dtype=tl.int32)
    absolute_sum = tl.zeros((), dtype=tl.float32)
    square_sum = tl.zeros((), dtype=tl.float32)
    fourth_moment_sum = tl.zeros((), dtype=tl.float32)
    absolute_maximum = tl.full((), -float("inf"), dtype=tl.float32)

    program_id = tl.program_id(0)
    indexed_value_count = value_count
    if USE_INT64_INDEX:
        program_id = program_id.to(tl.int64)
        indexed_value_count = value_count.to(tl.int64)
    program_start = program_id * BLOCK_SIZE
    if NEEDS_LOOP:
        # The bounded grid loops over additional blocks for large tensors.
        program_count = tl.num_programs(0)
        if USE_INT64_INDEX:
            program_count = program_count.to(tl.int64)
        program_stride = program_count * BLOCK_SIZE
        for block_start in tl.range(
            program_start,
            indexed_value_count,
            program_stride,
            num_stages=3,
        ):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            present = offsets < indexed_value_count
            value = tl.load(value_ptr + offsets, mask=present, other=0.0).to(tl.float32)
            finite = (
                present
                & (value == value)
                & (value != float("inf"))
                & (value != -float("inf"))
            )
            finite_value = tl.where(finite, value, 0.0)
            absolute = tl.abs(finite_value)
            square = finite_value * finite_value
            nonfinite = present & ~finite  # pyrefly: ignore [deprecated]
            nonfinite_i32 = nonfinite.to(  # pyrefly: ignore [missing-attribute]
                tl.int32
            )
            zero_i32 = (  # pyrefly: ignore [missing-attribute]
                finite & (value == 0.0)
            ).to(tl.int32)
            nonfinite_count += tl.sum(nonfinite_i32)
            zero_count += tl.sum(zero_i32)
            absolute_sum += tl.sum(absolute)
            square_sum += tl.sum(square)
            fourth_moment_sum += tl.sum(square * square)
            absolute_maximum = tl.maximum(
                absolute_maximum,
                tl.max(tl.where(finite, absolute, -float("inf"))),
            )
    else:
        offsets = program_start + tl.arange(0, BLOCK_SIZE)
        present = offsets < indexed_value_count
        value = tl.load(value_ptr + offsets, mask=present, other=0.0).to(tl.float32)
        finite = (
            present
            & (value == value)
            & (value != float("inf"))
            & (value != -float("inf"))
        )
        finite_value = tl.where(finite, value, 0.0)
        absolute = tl.abs(finite_value)
        square = finite_value * finite_value
        nonfinite = present & ~finite  # pyrefly: ignore [deprecated]
        nonfinite_i32 = nonfinite.to(tl.int32)  # pyrefly: ignore [missing-attribute]
        zero_i32 = (finite & (value == 0.0)).to(  # pyrefly: ignore [missing-attribute]
            tl.int32
        )
        nonfinite_count = tl.sum(nonfinite_i32)
        zero_count = tl.sum(zero_i32)
        absolute_sum = tl.sum(absolute)
        square_sum = tl.sum(square)
        fourth_moment_sum = tl.sum(square * square)
        absolute_maximum = tl.max(tl.where(finite, absolute, -float("inf")))

    # Programs summarize disjoint slices; atomics merge the three output groups.
    tl.atomic_add(counts_ptr + 1, tl.cast(nonfinite_count, tl.int64))
    tl.atomic_add(counts_ptr + 2, tl.cast(zero_count, tl.int64))
    if tl.program_id(0) == 0:
        tl.atomic_add(counts_ptr, tl.cast(value_count, tl.int64))
        tl.atomic_add(counts_ptr + 3, 1)
    tl.atomic_add(sums_ptr, absolute_sum)
    tl.atomic_add(sums_ptr + 1, square_sum)
    tl.atomic_add(sums_ptr + 2, fourth_moment_sum)
    tl.atomic_max(maximum_ptr, absolute_maximum)


def accumulate_contiguous_tensor_statistics(
    value: torch.Tensor,
    counts: torch.Tensor,
    sums: torch.Tensor,
    maximum: torch.Tensor,
    enabled: torch.Tensor,
) -> None:
    """Launch the bounded-grid accumulator for one contiguous tensor."""

    ideal_program_count = max(
        1,
        (value.numel() + _BLOCK_SIZE - 1) // _BLOCK_SIZE,
    )
    program_count = min(_MAX_PROGRAMS, ideal_program_count)
    needs_loop = ideal_program_count > program_count
    _accumulate_tensor_statistics_triton[(program_count,)](
        value,
        counts,
        sums,
        maximum,
        enabled,
        value.numel(),
        BLOCK_SIZE=_BLOCK_SIZE,  # pyrefly: ignore[bad-argument-type]
        NEEDS_LOOP=needs_loop,  # pyrefly: ignore[bad-argument-type]
        USE_INT64_INDEX=value.numel() > _MAX_INT32_INDEXED_ELEMENTS,
    )
