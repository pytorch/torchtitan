# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic load balancing across Muon compute partitions."""

from __future__ import annotations

import heapq
from collections.abc import Sequence


def balance_loads_across_partitions(
    loads: Sequence[tuple[int, int, str]],
    *,
    initial_cumulative_primary_loads: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Balance keyed loads with a deterministic LPT heuristic.

    Each load is ``(primary, secondary, stable_key)``. Assignments are
    partition indices aligned with those loads. Each call balances its primary
    and then secondary loads before cumulative primary load. Stable keys make
    ordering deterministic. This is not an exact partition optimum.
    """
    num_partitions = len(initial_cumulative_primary_loads)
    if not num_partitions:
        raise ValueError("at least one partition is required")

    assignments = [0] * len(loads)
    partition_loads = [
        (0, 0, cumulative_load, partition)
        for partition, cumulative_load in enumerate(initial_cumulative_primary_loads)
    ]
    heapq.heapify(partition_loads)
    ordered_loads = sorted(
        enumerate(loads),
        key=lambda indexed_load: (
            -indexed_load[1][0],
            -indexed_load[1][1],
            indexed_load[1][2],
        ),
    )
    for load_index, (primary, secondary, _stable_key) in ordered_loads:
        (
            current_primary,
            current_secondary,
            cumulative_primary,
            partition,
        ) = heapq.heappop(partition_loads)
        assignments[load_index] = partition
        heapq.heappush(
            partition_loads,
            (
                current_primary + primary,
                current_secondary + secondary,
                cumulative_primary + primary,
                partition,
            ),
        )

    updated_cumulative_primary_loads = [0] * num_partitions
    for _primary, _secondary, cumulative_primary, partition in partition_loads:
        updated_cumulative_primary_loads[partition] = cumulative_primary
    return tuple(assignments), tuple(updated_cumulative_primary_loads)
