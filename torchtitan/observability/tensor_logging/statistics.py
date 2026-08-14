# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn


NUMEL = 0
NONFINITE_COUNT = 1
ZERO_COUNT = 2
OBSERVATION_COUNT = 3
COUNT_FIELD_COUNT = 4

ABS_SUM = 0
SQUARE_SUM = 1
FOURTH_MOMENT_SUM = 2
SUM_FIELD_COUNT = 3


class StatisticBuffers(nn.Module):
    """Packed sufficient statistics with one row per registered metric.

    Example:

        counts[metric] = [numel, nonfinite_count, zero_count, observation_count]
        sums[metric] = [abs_sum, square_sum, fourth_moment_sum]
        maxima[metric] = abs_max
    """

    counts: torch.Tensor
    sums: torch.Tensor
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
            "counts",
            torch.zeros(
                (metric_count, COUNT_FIELD_COUNT), dtype=torch.int64, device=device
            ),
            persistent=False,
        )
        self.register_buffer(
            "sums",
            torch.zeros(
                (metric_count, SUM_FIELD_COUNT), dtype=torch.float32, device=device
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
        self.counts.zero_()
        self.sums.zero_()
        self.maxima.fill_(-torch.inf)


def _normalize_tensor_layout(value: torch.Tensor) -> torch.Tensor:
    """Reorder and collapse dimensions so common strided tensors scan as one view.

    Since statistic order is irrelevant, follow storage order and collapse
    contiguous runs; the caller copies only genuinely strided layouts.

    Example:

        value = torch.empty(2, 3, 4).transpose(0, 1)
        # shape=(3, 2, 4), stride=(4, 12, 1)
        normalized = _normalize_tensor_layout(value)
        # shape=(2, 3, 4), stride=(12, 4, 1), no copy
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
    mutates_args={"counts", "sums", "maximum"},
)
def accumulate_tensor_statistics(
    value: torch.Tensor,
    counts: torch.Tensor,
    sums: torch.Tensor,
    maximum: torch.Tensor,
    enabled: torch.Tensor,
) -> None:
    """Accumulate one tensor through an opaque, compile-safe operation.

    Example:

        # The output tensors are one preallocated metric row.
        accumulate_tensor_statistics(value, counts, sums, maximum, enabled)
    """

    if value.is_cuda:
        from .statistics_triton import accumulate_contiguous_tensor_statistics

        value = _normalize_tensor_layout(value)
        if not value.is_contiguous():
            value = value.contiguous()
        accumulate_contiguous_tensor_statistics(
            value,
            counts,
            sums,
            maximum,
            enabled,
        )
        return

    # CPU is the readable reference path for tests and non-CUDA execution.
    with torch.no_grad():
        if not bool(enabled):
            return
        value = value.detach()
        counts[OBSERVATION_COUNT].add_(1)
        if value.numel() == 0:
            return

        finite = torch.isfinite(value)
        value_fp32 = value.to(torch.float32)
        finite_value = torch.where(finite, value_fp32, 0.0)
        absolute = finite_value.abs()
        square = finite_value.square()

        counts[NUMEL].add_(value.numel())
        counts[NONFINITE_COUNT].add_(torch.count_nonzero(~finite))
        counts[ZERO_COUNT].add_(torch.count_nonzero(finite & (value == 0)))

        sums[ABS_SUM].add_(absolute.sum())
        sums[SQUARE_SUM].add_(square.sum())
        sums[FOURTH_MOMENT_SUM].add_(square.square().sum())

        finite_absolute = torch.where(finite, value_fp32.abs(), -torch.inf)
        updated_maximum = torch.maximum(maximum, finite_absolute.amax())
        maximum.copy_(updated_maximum)


@accumulate_tensor_statistics.register_fake
def _(
    value: torch.Tensor,
    counts: torch.Tensor,
    sums: torch.Tensor,
    maximum: torch.Tensor,
    enabled: torch.Tensor,
) -> None:
    return None
