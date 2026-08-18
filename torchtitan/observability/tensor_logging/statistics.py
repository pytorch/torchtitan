# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn


NUMEL_INDEX = 0
NONFINITE_COUNT_INDEX = 1
ZERO_COUNT_INDEX = 2
OBSERVATION_COUNT_INDEX = 3

ABS_SUM_INDEX = 4
SQUARE_SUM_INDEX = 5
FOURTH_MOMENT_SUM_INDEX = 6
SUM_STATISTIC_FIELD_COUNT = 7  # Count of indices above, not an index.


class StatisticBuffers(nn.Module):
    """Raw tensor statistics with one row per registered metric.

    `sum_statistics` stores the seven fields combined with SUM. `maxima` stores
    the field combined with MAX. `enabled` is 1 when this training step should
    update the rows and 0 when logging operations should leave them unchanged.

    Example:

        sum_statistics[metric] = [
            numel,
            nonfinite_count,
            zero_count,
            observation_count,
            abs_sum,
            square_sum,
            fourth_moment_sum,
        ]
        maxima[metric] = abs_max
    """

    sum_statistics: torch.Tensor
    maxima: torch.Tensor
    enabled: torch.Tensor

    def __init__(
        self,
        metric_count: int,
        *,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "sum_statistics",
            torch.zeros(
                (metric_count, SUM_STATISTIC_FIELD_COUNT),
                dtype=torch.float32,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "maxima",
            torch.full(
                (metric_count,),
                -torch.inf,
                dtype=torch.float32,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "enabled",
            torch.zeros((), dtype=torch.int32, device=device),
            persistent=False,
        )

    def clear(self) -> None:
        self.sum_statistics.zero_()
        self.maxima.fill_(-torch.inf)


def _view_tensor_in_storage_order(value: torch.Tensor) -> torch.Tensor:
    """Try to create a contiguous view without copying the tensor.

    Statistics do not depend on element order. Reading common transposes and
    permutations in storage order avoids a full tensor copy. A slice with gaps
    can remain noncontiguous; the caller copies that fallback.

    Example:

        value = torch.empty(2, 3, 4).transpose(0, 1)
        # shape=(3, 2, 4), stride=(4, 12, 1)
        scan_value = _view_tensor_in_storage_order(value)
        # `scan_value` is contiguous and shares storage with `value`.
    """

    if value.ndim <= 1:
        return value

    # Follow decreasing strides so adjacent storage regions become neighbors.
    dimension_order = sorted(
        range(value.ndim),
        key=lambda dimension: value.stride()[dimension],
        reverse=True,
    )
    value = value.permute(dimension_order)
    if value.is_contiguous() or value.ndim <= 2:
        return value

    # Size-one dimensions do not affect which storage elements are visited.
    shape_stride = [
        (size, stride)
        for size, stride in zip(value.shape, value.stride(), strict=True)
        if size != 1
    ]
    if len(shape_stride) <= 1:
        return value.reshape(value.numel())

    # Coalesce only dimensions whose strides prove that they are contiguous.
    collapsed_shape = [shape_stride[0][0]]
    collapsed_stride = [shape_stride[0][1]]
    for size, stride in shape_stride[1:]:
        if collapsed_stride[-1] == stride * size:
            collapsed_shape[-1] *= size
            collapsed_stride[-1] = stride
        else:
            collapsed_shape.append(size)
            collapsed_stride.append(stride)

    if len(collapsed_shape) < len(shape_stride):
        value = value.as_strided(
            collapsed_shape,
            collapsed_stride,
            value.storage_offset(),
        )
    return value


@torch.library.custom_op(
    "torchtitan::accumulate_tensor_statistics",
    mutates_args={"sum_statistics", "maximum"},
)
def accumulate_tensor_statistics(
    value: torch.Tensor,
    sum_statistics: torch.Tensor,
    maximum: torch.Tensor,
    enabled: torch.Tensor,
    slot_index: torch.Tensor,
) -> None:
    """Add one tensor observation to one preallocated buffer row.

    PyTorch represents this function as one node in the compiled graph. The
    custom-op declaration tells PyTorch that the node changes `sum_statistics`
    and `maximum` in place.

    While PyTorch builds a graph, FakeTensors contain shapes and dtypes but no
    real values. The fake implementation therefore returns `None` without
    scanning a tensor or changing a buffer.

    During training, if `enabled` is 0, it returns without changing the row.
    Because `enabled` is a tensor, changing cadence does not require a different compiled graph.

    Example:

        value = torch.tensor([0.0, 1.0, -2.0, 3.0])
        sum_statistics = torch.zeros(1, 7)
        maximum = torch.full((1,), -torch.inf)
        enabled = torch.ones((), dtype=torch.int32)
        slot_index = torch.tensor(0)

        accumulate_tensor_statistics(
            value,
            sum_statistics,
            maximum,
            enabled,
            slot_index,
        )

        assert sum_statistics[0].tolist() == [4, 0, 1, 1, 6, 14, 98]
        assert maximum[0].item() == 3
    """

    row = int(slot_index.item())
    sum_statistics_row = sum_statistics[row]
    maximum_row = maximum[row]

    if value.is_cuda:
        from .statistics_triton import accumulate_contiguous_tensor_statistics

        # The Triton kernel scans contiguous storage. Avoid a copy for common
        # transposes/permutations; copy only layouts that still contain gaps.
        value = _view_tensor_in_storage_order(value)
        if not value.is_contiguous():
            value = value.contiguous()
        accumulate_contiguous_tensor_statistics(
            value,
            sum_statistics_row,
            maximum_row,
            enabled,
        )
        return

    with torch.no_grad():
        if not bool(enabled):
            return
        value = value.detach()

        # One `log_stats()` call is one observation, including an empty tensor.
        sum_statistics_row[OBSERVATION_COUNT_INDEX].add_(1)
        if value.numel() == 0:
            return

        # Nonfinite values count toward `numel` and `nonfinite_count`, but not
        # zero counts, moments, or the finite absolute maximum.
        finite = torch.isfinite(value)
        value_fp32 = value.to(torch.float32)
        finite_value = torch.where(finite, value_fp32, 0.0)
        absolute = finite_value.abs()
        square = finite_value.square()

        sum_statistics_row[NUMEL_INDEX].add_(value.numel())
        sum_statistics_row[NONFINITE_COUNT_INDEX].add_(torch.count_nonzero(~finite))
        sum_statistics_row[ZERO_COUNT_INDEX].add_(
            torch.count_nonzero(finite & (value == 0))
        )

        sum_statistics_row[ABS_SUM_INDEX].add_(absolute.sum())
        sum_statistics_row[SQUARE_SUM_INDEX].add_(square.sum())
        sum_statistics_row[FOURTH_MOMENT_SUM_INDEX].add_(square.square().sum())

        finite_absolute = torch.where(finite, value_fp32.abs(), -torch.inf)
        updated_maximum = torch.maximum(maximum_row, finite_absolute.amax())
        maximum_row.copy_(updated_maximum)


@accumulate_tensor_statistics.register_fake
def _(
    value: torch.Tensor,
    sum_statistics: torch.Tensor,
    maximum: torch.Tensor,
    enabled: torch.Tensor,
    slot_index: torch.Tensor,
) -> None:
    return None
