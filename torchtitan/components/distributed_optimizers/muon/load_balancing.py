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
    loads: Sequence[tuple[int, str]],
    *,
    initial_partition_loads: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Balance keyed loads with a deterministic LPT heuristic.

    ``loads`` contains ``(load, stable_key)`` pairs. Assignments are partition
    indices aligned with those pairs. Each call balances its own loads first;
    ``initial_partition_loads`` breaks ties across a sequence of calls. Stable
    keys make ordering deterministic. This is not an exact partition optimum.
    """
    num_partitions = len(initial_partition_loads)
    if not num_partitions:
        raise ValueError("at least one partition is required")

    assignments = [0] * len(loads)
    partition_loads = [
        (0, cumulative_load, partition)
        for partition, cumulative_load in enumerate(initial_partition_loads)
    ]
    heapq.heapify(partition_loads)
    ordered_loads = sorted(
        enumerate(loads),
        key=lambda indexed_load: (
            -indexed_load[1][0],
            indexed_load[1][1],
        ),
    )
    for load_index, (load, _stable_key) in ordered_loads:
        current_load, cumulative_load, partition = heapq.heappop(partition_loads)
        assignments[load_index] = partition
        heapq.heappush(
            partition_loads,
            (
                current_load + load,
                cumulative_load + load,
                partition,
            ),
        )

    updated_loads = [0] * num_partitions
    for _current_load, cumulative_load, partition in partition_loads:
        updated_loads[partition] = cumulative_load
    return tuple(assignments), tuple(updated_loads)
