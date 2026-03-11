# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Layer 2: Load-Balanced Dispatch Solver

Two solvers:
- ``solve_dispatch``: assigns (Q,K) pairs to ranks (original, used by ring-pass/group-cast).
- ``solve_magi_dispatch``: assigns Q sub-chunks to ranks with equal-count constraint
  for true load-balanced redistribution (used by magi dispatch mode).
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, field

from torchtitan.distributed.varlen_cp.mask_primitives import (
    AttnSlice,
    split_slice_at_chunk_boundary,
)


@dataclass
class ChunkPair:
    """A (Q_chunk, K_chunk) pair with its associated attention sub-slices."""

    q_chunk_idx: int
    k_chunk_idx: int
    slices: list[AttnSlice] = field(default_factory=list)
    work: float = 0.0


@dataclass
class DispatchPlan:
    """Result of solve_dispatch: per-rank work assignments."""

    assignments: list[list[ChunkPair]]  # assignments[rank] = list of ChunkPairs
    chunk_size: int
    total_seqlen: int
    cp_world_size: int
    pad_size: int  # How much padding was added to make total_seqlen divisible

    def __post_init__(self):
        self._work_pairs: frozenset[tuple[int, int]] = frozenset(
            (cp.q_chunk_idx, cp.k_chunk_idx)
            for rank_assignments in self.assignments
            for cp in rank_assignments
        )

    def pair_has_work(self, q_chunk_idx: int, k_chunk_idx: int) -> bool:
        """Check if a (Q, K) chunk pair has any work (across all ranks)."""
        return (q_chunk_idx, k_chunk_idx) in self._work_pairs

    @property
    def num_chunks(self) -> int:
        return (self.total_seqlen + self.chunk_size - 1) // self.chunk_size

    def get_rank_work(self, rank: int) -> float:
        """Total estimated work for a rank."""
        return sum(cp.work for cp in self.assignments[rank])

    def has_work_for_pair(self, rank: int, q_chunk_idx: int, k_chunk_idx: int) -> bool:
        """Check if a rank has work for a given (Q_chunk, K_chunk) pair."""
        return any(
            cp.q_chunk_idx == q_chunk_idx and cp.k_chunk_idx == k_chunk_idx
            for cp in self.assignments[rank]
        )

    def get_pairs_for_rank(self, rank: int) -> list[tuple[int, int]]:
        """Get all (q_chunk_idx, k_chunk_idx) pairs assigned to a rank."""
        return [
            (cp.q_chunk_idx, cp.k_chunk_idx) for cp in self.assignments[rank]
        ]


def solve_dispatch(
    global_slices: list[AttnSlice],
    total_seqlen: int,
    chunk_size: int,
    cp_world_size: int,
) -> DispatchPlan:
    """Assign (Q_chunk, K_chunk) work pairs to CP ranks.

    Algorithm:
    1. Compute chunk boundaries from chunk_size
    2. For each global slice, split at chunk boundaries into sub-slices
    3. Group sub-slices by (q_chunk_idx, k_chunk_idx) into ChunkPairs
    4. Sort ChunkPairs by work descending
    5. Use min-heap to greedily assign each ChunkPair to the least-loaded rank

    Args:
        global_slices: List of global AttnSlices (from cu_seqlens_to_attn_slices).
        total_seqlen: Total packed sequence length (after padding if needed).
        chunk_size: Size of each chunk (typically total_seqlen // cp_world_size).
        cp_world_size: Number of CP ranks.

    Returns:
        DispatchPlan with per-rank assignments.
    """
    # Compute padding if needed
    padded_seqlen = total_seqlen
    pad_size = 0
    if total_seqlen % chunk_size != 0:
        padded_seqlen = ((total_seqlen + chunk_size - 1) // chunk_size) * chunk_size
        pad_size = padded_seqlen - total_seqlen

    # Split each global slice at chunk boundaries and group by (q_chunk, k_chunk)
    chunk_pair_map: dict[tuple[int, int], ChunkPair] = defaultdict(
        lambda: ChunkPair(q_chunk_idx=-1, k_chunk_idx=-1)
    )

    for global_slice in global_slices:
        sub_slices = split_slice_at_chunk_boundary(
            global_slice, chunk_size, padded_seqlen
        )
        for sub_slice in sub_slices:
            q_chunk_idx = sub_slice.q_start // chunk_size
            k_chunk_idx = sub_slice.k_start // chunk_size
            key = (q_chunk_idx, k_chunk_idx)

            if key not in chunk_pair_map:
                chunk_pair_map[key] = ChunkPair(
                    q_chunk_idx=q_chunk_idx, k_chunk_idx=k_chunk_idx
                )
            chunk_pair_map[key].slices.append(sub_slice)
            chunk_pair_map[key].work += sub_slice.work_estimate

    # Sort chunk pairs by work descending (largest first for better balancing)
    chunk_pairs = sorted(chunk_pair_map.values(), key=lambda cp: cp.work, reverse=True)

    # Min-heap greedy assignment: assign each ChunkPair to least-loaded rank
    # Heap entries: (total_work, rank_index)
    heap: list[tuple[float, int]] = [(0.0, rank) for rank in range(cp_world_size)]
    heapq.heapify(heap)

    assignments: list[list[ChunkPair]] = [[] for _ in range(cp_world_size)]

    for cp in chunk_pairs:
        total_work, rank = heapq.heappop(heap)
        assignments[rank].append(cp)
        heapq.heappush(heap, (total_work + cp.work, rank))

    return DispatchPlan(
        assignments=assignments,
        chunk_size=chunk_size,
        total_seqlen=padded_seqlen,
        cp_world_size=cp_world_size,
        pad_size=pad_size,
    )


# ---------------------------------------------------------------------------
# Magi-style dispatch: assign Q sub-chunks to ranks
# ---------------------------------------------------------------------------


@dataclass
class MagiDispatchPlan:
    """Magi-style dispatch plan: assigns Q sub-chunks to ranks.

    Unlike ``DispatchPlan`` which assigns (Q,K) pairs, this assigns whole
    Q sub-chunks. Each rank gets exactly ``sub_chunks_per_rank`` Q sub-chunks
    and is responsible for computing *all* attention for those Q sub-chunks
    (gathering the needed K/V from other ranks).
    """

    q_assignments: list[list[int]]  # [rank] -> sorted list of sub-chunk indices
    q_to_k_needs: dict[int, list[int]]  # q_sub_idx -> sorted [k_sub_idx, ...]
    sub_chunk_size: int  # tokens per sub-chunk
    num_sub_chunks: int  # = cp_world_size * sub_chunks_per_rank
    sub_chunks_per_rank: int
    chunk_size: int  # original chunk_size = total_seqlen / cp_world_size
    total_seqlen: int  # padded
    cp_world_size: int
    pad_size: int

    # Pre-computed for fast lookup (set in __post_init__)
    _q_assignment_sets: list[frozenset[int]] = field(
        default_factory=list, repr=False
    )
    _rank_k_needs: list[frozenset[int]] = field(
        default_factory=list, repr=False
    )

    def __post_init__(self):
        self._q_assignment_sets = [frozenset(qa) for qa in self.q_assignments]
        self._rank_k_needs = [
            frozenset(
                ki for qi in qa for ki in self.q_to_k_needs.get(qi, [])
            )
            for qa in self.q_assignments
        ]

    def owner_rank(self, sub_chunk_idx: int) -> int:
        """Which rank originally owns this sub-chunk (before redistribution)."""
        return sub_chunk_idx // self.sub_chunks_per_rank

    def is_assigned_to(self, rank: int, sub_chunk_idx: int) -> bool:
        return sub_chunk_idx in self._q_assignment_sets[rank]

    def rank_needs_k(self, rank: int, k_sub_idx: int) -> bool:
        return k_sub_idx in self._rank_k_needs[rank]


def solve_magi_dispatch(
    global_slices: list[AttnSlice],
    total_seqlen: int,
    chunk_size: int,
    cp_world_size: int,
    sub_chunks_per_rank: int = 2,
) -> MagiDispatchPlan:
    """Assign Q sub-chunks to CP ranks for load-balanced redistribution.

    Splits the sequence into ``cp_world_size * sub_chunks_per_rank`` sub-chunks.
    Each Q sub-chunk is assigned to exactly one rank, with each rank getting
    exactly ``sub_chunks_per_rank`` Q sub-chunks. Uses LPT (Longest Processing
    Time first) to minimize the maximum load across ranks.

    Args:
        global_slices: Global AttnSlices (from ``cu_seqlens_to_attn_slices``).
        total_seqlen: Total packed sequence length.
        chunk_size: Original chunk size (``total_seqlen // cp_world_size``).
        cp_world_size: Number of CP ranks.
        sub_chunks_per_rank: How many sub-chunks each rank processes (default 2).

    Returns:
        MagiDispatchPlan with per-rank Q sub-chunk assignments.
    """
    num_sub_chunks = cp_world_size * sub_chunks_per_rank
    sub_chunk_size = chunk_size // sub_chunks_per_rank
    assert chunk_size % sub_chunks_per_rank == 0, (
        f"chunk_size ({chunk_size}) must be divisible by "
        f"sub_chunks_per_rank ({sub_chunks_per_rank})"
    )

    padded_seqlen = total_seqlen
    pad_size = 0
    if total_seqlen % sub_chunk_size != 0:
        padded_seqlen = (
            (total_seqlen + sub_chunk_size - 1) // sub_chunk_size
        ) * sub_chunk_size
        pad_size = padded_seqlen - total_seqlen

    # Split global slices at sub-chunk boundaries and compute per-pair work
    pair_work: dict[tuple[int, int], float] = defaultdict(float)
    for global_slice in global_slices:
        sub_slices = split_slice_at_chunk_boundary(
            global_slice, sub_chunk_size, padded_seqlen
        )
        for sub_slice in sub_slices:
            q_idx = sub_slice.q_start // sub_chunk_size
            k_idx = sub_slice.k_start // sub_chunk_size
            pair_work[(q_idx, k_idx)] += sub_slice.work_estimate

    # Aggregate per-Q-sub-chunk: total work and needed K sub-chunks
    q_total_work: dict[int, float] = defaultdict(float)
    q_to_k_needs: dict[int, list[int]] = defaultdict(list)
    for (q_idx, k_idx), work in pair_work.items():
        q_total_work[q_idx] += work
        q_to_k_needs[q_idx].append(k_idx)
    for qi in q_to_k_needs:
        q_to_k_needs[qi].sort()

    # LPT assignment: sort Q sub-chunks by work descending, assign to
    # least-loaded rank that still has capacity.
    sorted_q = sorted(
        range(num_sub_chunks), key=lambda i: q_total_work.get(i, 0), reverse=True
    )

    assignments: list[list[int]] = [[] for _ in range(cp_world_size)]
    rank_load = [0.0] * cp_world_size
    rank_count = [0] * cp_world_size

    for q_idx in sorted_q:
        # Find least-loaded rank with capacity
        best_rank = -1
        best_load = float("inf")
        for r in range(cp_world_size):
            if rank_count[r] < sub_chunks_per_rank and rank_load[r] < best_load:
                best_load = rank_load[r]
                best_rank = r
        assert best_rank >= 0
        assignments[best_rank].append(q_idx)
        rank_load[best_rank] += q_total_work.get(q_idx, 0)
        rank_count[best_rank] += 1

    # Sort each rank's assignments for deterministic ordering
    for r in range(cp_world_size):
        assignments[r].sort()

    return MagiDispatchPlan(
        q_assignments=assignments,
        q_to_k_needs=dict(q_to_k_needs),
        sub_chunk_size=sub_chunk_size,
        num_sub_chunks=num_sub_chunks,
        sub_chunks_per_rank=sub_chunks_per_rank,
        chunk_size=chunk_size,
        total_seqlen=padded_seqlen,
        cp_world_size=cp_world_size,
        pad_size=pad_size,
    )
