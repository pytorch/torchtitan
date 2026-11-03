# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

# Columns of `sum_statistics`, combined with SUM over observations and ranks.
NUMEL_INDEX = 0
NONFINITE_COUNT_INDEX = 1
ZERO_COUNT_INDEX = 2
OBSERVATION_COUNT_INDEX = 3

ABS_SUM_INDEX = 4
SQUARE_SUM_INDEX = 5
FOURTH_MOMENT_SUM_INDEX = 6
SUM_STATISTIC_FIELD_COUNT = 7  # Count of indices above, not an index.


@torch.library.custom_op(
    "torchtitan::accumulate_tensor_statistics",
    mutates_args={"sum_statistics", "maxima"},
)
def accumulate_tensor_statistics(
    value: torch.Tensor,
    sum_statistics: torch.Tensor,
    maxima: torch.Tensor,
    enabled: torch.Tensor,
    row: int,
) -> None:
    """Add one observation of ``value`` to ``row``; no-op when ``enabled == 0``.

    ``sum_statistics[row]`` holds the 7 SUM columns above, ``maxima[row]`` the
    finite absolute maximum. Nonfinite values count toward ``numel`` and
    ``nonfinite_count`` only.

    Example:
        value = torch.tensor([0.0, 1.0, -2.0, 3.0])
        accumulate_tensor_statistics(value, sum_statistics, maxima, enabled, row=0)
        # sum_statistics[0] += [4, 0, 1, 1, 6, 14, 98]; maxima[0] = 3
    """
    # Lazy: model code imports this module, and CPU-only installs have no Triton.
    from .statistics_triton import accumulate_contiguous_tensor_statistics

    # Statistics ignore element order, so a permuted dense tensor (for example a
    # transpose) is read in storage order without a copy. Only layouts with gaps
    # are copied.
    if not value.is_contiguous():
        value = value.permute(value.dim_order())
        if not value.is_contiguous():
            value = value.contiguous()
    accumulate_contiguous_tensor_statistics(
        value, sum_statistics[row], maxima[row], enabled
    )


@accumulate_tensor_statistics.register_kernel("cpu")
def _(
    value: torch.Tensor,
    sum_statistics: torch.Tensor,
    maxima: torch.Tensor,
    enabled: torch.Tensor,
    row: int,
) -> None:
    """Reference implementation in plain PyTorch, used on CPU.

    Accelerators use the Triton kernel; the GPU tests compare the two.
    """
    if not bool(enabled):
        return
    sum_statistics_row = sum_statistics[row]
    maximum_row = maxima[row]
    value = value.detach()

    # One log call is one observation, including an empty tensor.
    sum_statistics_row[OBSERVATION_COUNT_INDEX].add_(1)
    if value.numel() == 0:
        return

    finite = torch.isfinite(value)
    value_fp32 = value.to(torch.float32)
    finite_value = torch.where(finite, value_fp32, 0.0)
    square = finite_value.square()

    sum_statistics_row[NUMEL_INDEX].add_(value.numel())
    sum_statistics_row[NONFINITE_COUNT_INDEX].add_(torch.count_nonzero(~finite))
    sum_statistics_row[ZERO_COUNT_INDEX].add_(
        torch.count_nonzero(finite & (value == 0))
    )
    sum_statistics_row[ABS_SUM_INDEX].add_(finite_value.abs().sum())
    sum_statistics_row[SQUARE_SUM_INDEX].add_(square.sum())
    sum_statistics_row[FOURTH_MOMENT_SUM_INDEX].add_(square.square().sum())

    finite_absolute = torch.where(finite, value_fp32.abs(), -torch.inf)
    maximum_row.copy_(torch.maximum(maximum_row, finite_absolute.amax()))
