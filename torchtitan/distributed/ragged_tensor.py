# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A variable number of values per token, padded to a fixed capacity.

Legend: T = tokens (rows), C = capacity (padded number of values), R = selected rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.export import register_dataclass


@dataclass(kw_only=True)
class RaggedTensor:
    """Row ``t`` holds token ``t``'s values: ``values[starts[t] : starts[t] + lengths[t]]``,
    with ``starts`` the exclusive cumulative sum of ``lengths``. The values in use come first;
    the rest, up to the capacity, is padding.

    The capacity is fixed by whoever builds the tensor, so the shapes stay the same from one
    microbatch to the next (accelerator graphs need that), and every operation here runs
    without a host sync. A model shards a ``RaggedTensor`` input along tokens like a ``[T]``
    tensor (`context_parallel.shard_tensors`), and a chunked loss splits it (`split`).

    Unlike ``torch.nested``'s jagged layout, it keeps a static capacity and needs no host
    sync to shard or split.

    Example (T = 3, C = 6):

        rows      [[5, 7], [], [2]]
        values    [5, 7, 2, 0, 0, 0]
        lengths   [2, 0, 1]
    """

    values: torch.Tensor  # [C]
    lengths: torch.Tensor  # [T], int64

    @classmethod
    def from_rows(
        cls, values: torch.Tensor, lengths: torch.Tensor, *, capacity: int
    ) -> RaggedTensor:
        """Pad the packed ``values`` of rows with ``lengths`` to ``capacity``.

        The number of values comes from the shape, so this needs no host sync.
        """
        if values.ndim != 1 or lengths.ndim != 1:
            raise ValueError(
                f"RaggedTensor values and lengths must be 1-D, got {tuple(values.shape)} "
                f"and {tuple(lengths.shape)}."
            )
        if values.shape[0] > capacity:
            raise ValueError(
                f"RaggedTensor has {values.shape[0]} values, more than its capacity "
                f"{capacity}."
            )
        padded = values.new_zeros(capacity)
        padded[: values.shape[0]] = values
        return cls(values=padded, lengths=lengths.long())

    def row_indices(self) -> torch.Tensor:
        """``[C]`` the row of each value; padding values get row ``T``, one past the last.

        Example: lengths [2, 0, 1], C = 6 -> [0, 0, 2, 3, 3, 3].
        """
        return _row_indices(self.lengths, self.values.shape[0])

    def take_rows(self, rows: torch.Tensor, *, capacity: int) -> RaggedTensor:
        """The rows ``rows[R]`` (in that order), packed and padded to ``capacity``.

        Raises on the device (asynchronously) if their values don't fit.

        Example: rows [[5, 7], [], [2], [4, 4]], take_rows([3, 0], capacity=4)
        -> values [4, 4, 5, 7], lengths [2, 2].
        """
        lengths = self.lengths.index_select(0, rows)
        torch._assert_async(
            lengths.sum() <= capacity,
            "RaggedTensor rows don't fit in the capacity they are packed into",
        )
        source_starts = (torch.cumsum(self.lengths, 0) - self.lengths).index_select(
            0, rows
        )
        row = _row_indices(lengths, capacity)  # [capacity], R for padding
        in_row = row < rows.shape[0]
        row = row.clamp_max(rows.shape[0] - 1)
        # Each value's position in its row, added to the row's start in ``self.values``.
        position_in_row = torch.arange(capacity, device=row.device) - (
            torch.cumsum(lengths, 0) - lengths
        ).index_select(0, row)
        source = (source_starts.index_select(0, row) + position_in_row).clamp(
            0, self.values.shape[0] - 1
        )
        values = torch.where(in_row, self.values.index_select(0, source), 0)
        return RaggedTensor(values=values, lengths=lengths)

    def split(self, num_splits: int) -> tuple[RaggedTensor, ...]:
        """Split the rows into ``num_splits`` equal parts, each packed into ``C // num_splits``
        values, e.g. for a loss computed in chunks of tokens.

        Example: lengths [2, 0, 1, 3], C = 8, 2 splits -> lengths [2, 0] and [1, 3], 4 values
        each.
        """
        num_rows, capacity = self.lengths.shape[0], self.values.shape[0]
        if num_rows % num_splits or capacity % num_splits:
            raise ValueError(
                f"RaggedTensor of {num_rows} rows and capacity {capacity} can't split into "
                f"{num_splits} equal parts."
            )
        rows = torch.arange(num_rows, device=self.lengths.device)
        return tuple(
            self.take_rows(part, capacity=capacity // num_splits)
            for part in rows.chunk(num_splits)
        )


def _row_indices(lengths: torch.Tensor, capacity: int) -> torch.Tensor:
    """``[capacity]`` the row of each value of rows with ``lengths``; padding gets ``T``."""
    num_rows = lengths.shape[0]
    # A last row holds the padding, so the repeats sum to the capacity and output_size needs
    # no host sync. Rows that overflow the capacity are cut off (callers check the sum).
    repeats = torch.cat([lengths, (capacity - lengths.sum()).clamp_min(0).view(1)])
    return torch.repeat_interleave(
        torch.arange(num_rows + 1, device=lengths.device),
        repeats,
        output_size=capacity,
    )


# Register as a pytree so accelerator graph replays copy the values and lengths.
register_dataclass(RaggedTensor)
