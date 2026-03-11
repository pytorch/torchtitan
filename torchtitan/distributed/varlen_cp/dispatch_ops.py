# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Layer 3: Dispatch/Undispatch Tensor Operations

Handle redistribution of Q/K/V tensors across CP ranks. For the simple
chunking approach used in varlen CP, each rank gets a contiguous chunk of
the packed sequence. The key operation is computing local cu_seqlens for
each rank's chunk.
"""

from __future__ import annotations

import torch
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.distributed.varlen_cp.dispatch_solver import DispatchPlan


def compute_local_cu_seqlens(
    global_cu_seqlens: torch.Tensor,
    chunk_start: int,
    chunk_end: int,
) -> tuple[torch.Tensor, int]:
    """Compute local cu_seqlens for a chunk range by intersecting document boundaries.

    Given global cumulative sequence lengths and a chunk range [chunk_start, chunk_end),
    compute the local cu_seqlens that describe document boundaries within this chunk.

    Args:
        global_cu_seqlens: Global cumulative sequence lengths, shape (num_docs + 1,).
            e.g. [0, 128, 300, 512] for 3 documents.
        chunk_start: Start token index of the chunk (inclusive).
        chunk_end: End token index of the chunk (exclusive).

    Returns:
        Tuple of:
            - local_cu_seqlens: Tensor of local cumulative sequence lengths,
              starting at 0 and ending at chunk_end - chunk_start.
            - max_seqlen: Maximum document length within this chunk.
    """
    chunk_len = chunk_end - chunk_start
    device = global_cu_seqlens.device

    # Convert to list for iteration (global_cu_seqlens is typically small)
    boundaries = global_cu_seqlens.tolist()

    local_boundaries = [0]
    for b in boundaries:
        # Clamp boundary to chunk range and convert to local offset
        local_b = max(0, min(b, chunk_end) - chunk_start)
        if local_b > local_boundaries[-1] and local_b <= chunk_len:
            local_boundaries.append(local_b)

    # Ensure we end at chunk_len
    if local_boundaries[-1] != chunk_len:
        local_boundaries.append(chunk_len)

    local_cu_seqlens = torch.tensor(local_boundaries, dtype=torch.int32, device=device)

    # Compute max local document length
    if len(local_boundaries) > 1:
        diffs = torch.diff(local_cu_seqlens)
        max_seqlen = diffs.max().item()
    else:
        max_seqlen = 0

    return local_cu_seqlens, int(max_seqlen)


def shard_sequence(
    x: torch.Tensor,
    cp_rank: int,
    cp_world_size: int,
    seq_dim: int = 0,
) -> torch.Tensor:
    """Shard a tensor along the sequence dimension for a specific CP rank.

    Simple contiguous chunking: rank i gets tokens [i*chunk_size, (i+1)*chunk_size).

    Args:
        x: Input tensor with sequence along seq_dim.
        cp_rank: This rank's index in the CP group.
        cp_world_size: Number of CP ranks.
        seq_dim: Which dimension is the sequence dimension.

    Returns:
        Chunk of x for this rank.
    """
    seq_len = x.size(seq_dim)
    chunk_size = seq_len // cp_world_size
    assert seq_len % cp_world_size == 0, (
        f"Sequence length {seq_len} must be divisible by CP world size {cp_world_size}"
    )
    start = cp_rank * chunk_size
    end = start + chunk_size
    return x.narrow(seq_dim, start, end - start).contiguous()


def dispatch(
    x: torch.Tensor,
    plan: DispatchPlan,
    cp_mesh: DeviceMesh,
    seq_dim: int = 0,
) -> torch.Tensor:
    """Shard x along the sequence dimension for this CP rank.

    For the simple chunking approach, each rank gets a contiguous chunk.
    Padding is applied if needed to make the sequence divisible.

    Args:
        x: Input tensor with sequence along seq_dim.
        plan: DispatchPlan from solve_dispatch.
        cp_mesh: DeviceMesh for CP group.
        seq_dim: Sequence dimension index.

    Returns:
        Local chunk of x for this rank.
    """
    cp_rank = cp_mesh.get_local_rank()
    cp_world_size = cp_mesh.size(0)

    # Pad if needed
    if plan.pad_size > 0:
        pad_shape = list(x.shape)
        pad_shape[seq_dim] = plan.pad_size
        padding = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        x = torch.cat([x, padding], dim=seq_dim)

    return shard_sequence(x, cp_rank, cp_world_size, seq_dim)


def undispatch(
    x: torch.Tensor,
    plan: DispatchPlan,
    cp_mesh: DeviceMesh,
    original_total_tokens: int,
    seq_dim: int = 0,
) -> torch.Tensor:
    """Gather chunks from all CP ranks and reconstruct the full sequence.

    Args:
        x: Local chunk tensor.
        plan: DispatchPlan from solve_dispatch.
        cp_mesh: DeviceMesh for CP group.
        original_total_tokens: Original unpadded total token count.
        seq_dim: Sequence dimension index.

    Returns:
        Full sequence tensor with padding removed.
    """
    cp_world_size = cp_mesh.size(0)

    # All-gather along sequence dim
    gathered = [torch.empty_like(x) for _ in range(cp_world_size)]
    torch.distributed.all_gather(
        gathered,
        x,
        group=cp_mesh.get_group(),
    )
    full = torch.cat(gathered, dim=seq_dim)

    # Remove padding
    if plan.pad_size > 0:
        full = full.narrow(seq_dim, 0, original_total_tokens)

    return full
