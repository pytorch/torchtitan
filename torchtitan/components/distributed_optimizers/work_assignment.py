# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic weighted work assignment for distributed optimizers."""

from __future__ import annotations

import heapq
from collections.abc import Callable, Sequence
from typing import TypeVar


_WorkT = TypeVar("_WorkT")


def assign_balanced_work(
    work: Sequence[_WorkT],
    *,
    num_partitions: int,
    initial_loads: Sequence[int],
    get_weight: Callable[[_WorkT], int],
    get_stable_key: Callable[[_WorkT], str],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Greedily assign heavier work while balancing current and prior load.

    Assignments are partition indices aligned with ``work``. Within one call,
    current load is the primary balancing key. ``initial_loads`` breaks ties
    across a sequence of calls, and ``get_stable_key`` makes work ordering
    deterministic.
    """
    if num_partitions <= 0 or len(initial_loads) != num_partitions:
        raise ValueError("partitions and initial loads must be aligned")

    assignments = [0] * len(work)
    partition_loads = [
        (0, cumulative_load, partition)
        for partition, cumulative_load in enumerate(initial_loads)
    ]
    heapq.heapify(partition_loads)
    ordered_work = sorted(
        enumerate(work),
        key=lambda indexed_work: (
            -get_weight(indexed_work[1]),
            get_stable_key(indexed_work[1]),
        ),
    )
    for work_index, item in ordered_work:
        current_load, cumulative_load, partition = heapq.heappop(partition_loads)
        assignments[work_index] = partition
        weight = get_weight(item)
        heapq.heappush(
            partition_loads,
            (
                current_load + weight,
                cumulative_load + weight,
                partition,
            ),
        )

    updated_loads = [0] * num_partitions
    for _current_load, cumulative_load, partition in partition_loads:
        updated_loads[partition] = cumulative_load
    return tuple(assignments), tuple(updated_loads)
