# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl

from . import statistics


# Triton kernels may read module globals only when they are constexpr objects.
NUMEL_INDEX = tl.constexpr(statistics.NUMEL_INDEX)
NONFINITE_COUNT_INDEX = tl.constexpr(statistics.NONFINITE_COUNT_INDEX)
ZERO_COUNT_INDEX = tl.constexpr(statistics.ZERO_COUNT_INDEX)
OBSERVATION_COUNT_INDEX = tl.constexpr(statistics.OBSERVATION_COUNT_INDEX)
ABS_SUM_INDEX = tl.constexpr(statistics.ABS_SUM_INDEX)
SQUARE_SUM_INDEX = tl.constexpr(statistics.SQUARE_SUM_INDEX)
FOURTH_MOMENT_SUM_INDEX = tl.constexpr(statistics.FOURTH_MOMENT_SUM_INDEX)


# Launch at most 1024 programs of 4096 elements; larger tensors loop. Not tuned
# per GPU (13 us for 1M bf16 elements on GB300).
_MAX_PROGRAMS = 1024
_BLOCK_SIZE = 4096


@triton.jit
def _accumulate_tensor_statistics_triton(
    value_ptr,
    sum_statistics_ptr,
    maximum_ptr,
    enabled_ptr,
    value_count,
    BLOCK_SIZE: tl.constexpr,
):
    if tl.load(enabled_ptr) == 0:
        return

    nonfinite_count = tl.zeros((), dtype=tl.int32)
    zero_count = tl.zeros((), dtype=tl.int32)
    absolute_sum = tl.zeros((), dtype=tl.float32)
    square_sum = tl.zeros((), dtype=tl.float32)
    fourth_moment_sum = tl.zeros((), dtype=tl.float32)
    absolute_maximum = tl.full((), -float("inf"), dtype=tl.float32)

    # Each program reads every program_count-th block. int64 offsets, because a
    # logged tensor can have more than 2**31 elements.
    program_start = tl.program_id(0).to(tl.int64) * BLOCK_SIZE
    program_stride = tl.num_programs(0).to(tl.int64) * BLOCK_SIZE
    for block_start in tl.range(
        program_start, value_count, program_stride, num_stages=3
    ):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        present = offsets < value_count
        value = tl.load(value_ptr + offsets, mask=present, other=0.0).to(tl.float32)
        # nan and +-inf fail this test.
        finite = present & (tl.abs(value) < float("inf"))
        finite_value = tl.where(finite, value, 0.0)
        absolute = tl.abs(finite_value)
        square = finite_value * finite_value
        nonfinite_count += tl.sum((present & ~finite).to(tl.int32))
        zero_count += tl.sum((finite & (value == 0.0)).to(tl.int32))
        absolute_sum += tl.sum(absolute)
        square_sum += tl.sum(square)
        fourth_moment_sum += tl.sum(square * square)
        absolute_maximum = tl.maximum(
            absolute_maximum, tl.max(tl.where(finite, absolute, -float("inf")))
        )

    # Each program reduced its own blocks; atomics add the results into the row.
    tl.atomic_add(
        sum_statistics_ptr + NONFINITE_COUNT_INDEX,
        tl.cast(nonfinite_count, tl.float32),
    )
    tl.atomic_add(
        sum_statistics_ptr + ZERO_COUNT_INDEX,
        tl.cast(zero_count, tl.float32),
    )
    if tl.program_id(0) == 0:
        tl.atomic_add(
            sum_statistics_ptr + NUMEL_INDEX,
            tl.cast(value_count, tl.float32),
        )
        tl.atomic_add(sum_statistics_ptr + OBSERVATION_COUNT_INDEX, 1.0)
    tl.atomic_add(sum_statistics_ptr + ABS_SUM_INDEX, absolute_sum)
    tl.atomic_add(sum_statistics_ptr + SQUARE_SUM_INDEX, square_sum)
    tl.atomic_add(
        sum_statistics_ptr + FOURTH_MOMENT_SUM_INDEX,
        fourth_moment_sum,
    )
    tl.atomic_max(maximum_ptr, absolute_maximum)


def accumulate_contiguous_tensor_statistics(
    value: torch.Tensor,
    sum_statistics: torch.Tensor,
    maximum: torch.Tensor,
    enabled: torch.Tensor,
) -> None:
    """Launch the accumulator for one contiguous tensor."""
    program_count = min(_MAX_PROGRAMS, max(1, triton.cdiv(value.numel(), _BLOCK_SIZE)))
    _accumulate_tensor_statistics_triton[(program_count,)](
        value,
        sum_statistics,
        maximum,
        enabled,
        value.numel(),
        BLOCK_SIZE=_BLOCK_SIZE,  # pyrefly: ignore[bad-argument-type]
    )
