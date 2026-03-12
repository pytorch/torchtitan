# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
AllToAll-V communication helpers for Magi Attention.

Based on: Men et al., "Magi Attention: Efficient Attention Mechanism for
Extreme-Scale LLM Training", https://arxiv.org/abs/2505.13211

Provides:
  - AllToAll-V wrapper (NCCL)
  - Q redistribution (Phase A)
  - Tensor redistribution to assigned ranks
  - Result return (Phase C)
  - dQ scatter (backward)

Env vars:
  TORCHTITAN_MAGI_SUB_CHUNKS=2   -- sub-chunks per rank (default 2)
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from torchtitan.distributed.varlen_cp.dispatch_solver import (
    MagiDispatchPlan,
)

_MAGI_SUB_CHUNKS = int(os.environ.get("TORCHTITAN_MAGI_SUB_CHUNKS", "2"))


# ---------------------------------------------------------------------------
# AllToAll-V helper
# ---------------------------------------------------------------------------


def _alltoall_v(
    send_buf: torch.Tensor,
    recv_splits: list[int],
    send_splits: list[int],
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Blocking AllToAll-V using NCCL."""
    total_recv = sum(recv_splits)
    recv_buf = torch.empty(total_recv, device=send_buf.device, dtype=send_buf.dtype)
    if total_recv > 0 or sum(send_splits) > 0:
        dist.all_to_all_single(
            recv_buf, send_buf, recv_splits, send_splits, group=group,
        )
    return recv_buf


# ---------------------------------------------------------------------------
# Phase A: Q redistribution
# ---------------------------------------------------------------------------


def _redistribute_q(
    q: torch.Tensor,
    plan: MagiDispatchPlan,
    cp_rank: int,
    cp_group: dist.ProcessGroup,
) -> dict[int, torch.Tensor]:
    """Redistribute Q sub-chunks to assigned ranks via AllToAll-V."""
    cp_world_size = plan.cp_world_size
    spr = plan.sub_chunks_per_rank
    sc_size = plan.sub_chunk_size
    n_heads, head_dim = q.shape[1], q.shape[2]
    sc_numel = sc_size * n_heads * head_dim

    my_range = range(cp_rank * spr, (cp_rank + 1) * spr)
    local_sub_chunks = list(q.split(sc_size, dim=0))

    send_splits: list[int] = []
    send_parts: list[torch.Tensor] = []
    recv_splits: list[int] = []

    for r in range(cp_world_size):
        count = 0
        for qi in my_range:
            if plan.is_assigned_to(r, qi):
                send_parts.append(local_sub_chunks[qi - cp_rank * spr].reshape(-1))
                count += 1
        send_splits.append(count * sc_numel)

        r_range = range(r * spr, (r + 1) * spr)
        rcount = sum(1 for qi in r_range if plan.is_assigned_to(cp_rank, qi))
        recv_splits.append(rcount * sc_numel)

    send_buf = torch.cat(send_parts) if send_parts else q.new_empty(0)
    recv_buf = _alltoall_v(send_buf, recv_splits, send_splits, cp_group)

    result: dict[int, torch.Tensor] = {}
    offset = 0
    for r in range(cp_world_size):
        r_range = range(r * spr, (r + 1) * spr)
        for qi in r_range:
            if plan.is_assigned_to(cp_rank, qi):
                result[qi] = recv_buf[offset : offset + sc_numel].reshape(
                    sc_size, n_heads, head_dim
                )
                offset += sc_numel

    return result


# ---------------------------------------------------------------------------
# Phase C: Return results (output + LSE)
# ---------------------------------------------------------------------------


def _return_results(
    results: dict[int, torch.Tensor],
    plan: MagiDispatchPlan,
    cp_rank: int,
    cp_group: dist.ProcessGroup,
    n_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Return computed output sub-chunks to original owners via AllToAll-V."""
    cp_world_size = plan.cp_world_size
    spr = plan.sub_chunks_per_rank
    sc_size = plan.sub_chunk_size
    sc_numel = sc_size * n_heads * head_dim

    my_assigned = plan.q_assignments[cp_rank]
    my_range = range(cp_rank * spr, (cp_rank + 1) * spr)

    send_splits: list[int] = []
    send_parts: list[torch.Tensor] = []
    recv_splits: list[int] = []

    for r in range(cp_world_size):
        r_range = range(r * spr, (r + 1) * spr)
        count = 0
        for qi in sorted(set(my_assigned) & set(r_range)):
            send_parts.append(results[qi].reshape(-1))
            count += 1
        send_splits.append(count * sc_numel)

        r_assigned = plan.q_assignments[r]
        rcount = sum(1 for qi in r_assigned if qi in set(my_range))
        recv_splits.append(rcount * sc_numel)

    send_buf = (
        torch.cat(send_parts) if send_parts else results[my_assigned[0]].new_empty(0)
    )
    recv_buf = _alltoall_v(send_buf, recv_splits, send_splits, cp_group)

    output_subs: dict[int, torch.Tensor] = {}
    offset = 0
    for r in range(cp_world_size):
        r_assigned = plan.q_assignments[r]
        for qi in sorted(qi for qi in r_assigned if qi in set(my_range)):
            output_subs[qi] = recv_buf[offset : offset + sc_numel].reshape(
                sc_size, n_heads, head_dim
            )
            offset += sc_numel

    parts = [output_subs[qi] for qi in sorted(output_subs.keys())]
    return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# Backward helpers: redistribute tensors to assigned ranks
# ---------------------------------------------------------------------------


def _redistribute_tensors_to_assigned(
    tensors: list[torch.Tensor],
    plan: MagiDispatchPlan,
    cp_rank: int,
    cp_group: dist.ProcessGroup,
) -> list[dict[int, torch.Tensor]]:
    """Redistribute multiple per-sub-chunk tensors to assigned ranks.

    Each tensor in `tensors` has shape (chunk_size, ...) and is split into
    sub-chunks of size sub_chunk_size. Sub-chunks are sent to their assigned
    rank per the MagiDispatchPlan (same routing as Phase A in forward).

    Returns a list of dicts, one per input tensor. Each dict maps
    sub-chunk index -> tensor at the assigned rank.
    """
    cp_world_size = plan.cp_world_size
    spr = plan.sub_chunks_per_rank
    sc_size = plan.sub_chunk_size
    my_range = range(cp_rank * spr, (cp_rank + 1) * spr)

    # Concatenate all tensors along the last dim for a single AllToAll-V
    # Each tensor: (chunk_size, ...) -> split into sub-chunks -> flatten
    n_tensors = len(tensors)
    per_sc_numels = []
    for t in tensors:
        # numel per sub-chunk for this tensor
        sub = t[:sc_size]
        per_sc_numels.append(sub.numel())
    total_sc_numel = sum(per_sc_numels)

    # Split each tensor into sub-chunks
    all_subs: list[list[torch.Tensor]] = []
    for t in tensors:
        all_subs.append(list(t.split(sc_size, dim=0)))

    send_splits: list[int] = []
    send_parts: list[torch.Tensor] = []
    recv_splits: list[int] = []

    for r in range(cp_world_size):
        count = 0
        for qi in my_range:
            if plan.is_assigned_to(r, qi):
                local_idx = qi - cp_rank * spr
                for ti in range(n_tensors):
                    send_parts.append(all_subs[ti][local_idx].reshape(-1))
                count += 1
        send_splits.append(count * total_sc_numel)

        r_range = range(r * spr, (r + 1) * spr)
        rcount = sum(1 for qi in r_range if plan.is_assigned_to(cp_rank, qi))
        recv_splits.append(rcount * total_sc_numel)

    send_buf = (
        torch.cat(send_parts) if send_parts else tensors[0].new_empty(0)
    )
    recv_buf = _alltoall_v(send_buf, recv_splits, send_splits, cp_group)

    # Unpack
    results: list[dict[int, torch.Tensor]] = [{} for _ in range(n_tensors)]
    offset = 0
    for r in range(cp_world_size):
        r_range = range(r * spr, (r + 1) * spr)
        for qi in r_range:
            if plan.is_assigned_to(cp_rank, qi):
                for ti in range(n_tensors):
                    numel = per_sc_numels[ti]
                    shape = list(tensors[ti][:sc_size].shape)
                    results[ti][qi] = recv_buf[offset : offset + numel].reshape(shape)
                    offset += numel

    return results


def _scatter_dq_to_owners(
    dq_by_qi: dict[int, torch.Tensor],
    plan: MagiDispatchPlan,
    cp_rank: int,
    cp_group: dist.ProcessGroup,
    q_shape: tuple[int, ...],
) -> torch.Tensor:
    """Scatter dQ sub-chunks back to their original Q owners.

    This is the reverse of Phase A. Each assigned rank sends dQ for
    Q sub-chunks back to the rank that originally owned them.

    Returns:
        Accumulated dQ for this rank's local Q, shape q_shape, float32.
    """
    cp_world_size = plan.cp_world_size
    spr = plan.sub_chunks_per_rank
    sc_size = plan.sub_chunk_size
    n_heads = q_shape[1]
    head_dim = q_shape[2]
    q_numel = sc_size * n_heads * head_dim

    my_range = range(cp_rank * spr, (cp_rank + 1) * spr)
    my_assigned = plan.q_assignments[cp_rank]

    send_splits: list[int] = []
    send_parts: list[torch.Tensor] = []
    recv_splits: list[int] = []

    for r in range(cp_world_size):
        # Send: dQ for Q sub-chunks owned by rank r that I computed
        r_q_range = range(r * spr, (r + 1) * spr)
        count = 0
        for qi in sorted(set(my_assigned) & set(r_q_range)):
            send_parts.append(dq_by_qi[qi].reshape(-1))
            count += 1
        send_splits.append(count * q_numel)

        # Recv: dQ from rank r for my Q sub-chunks that r computed
        r_assigned = plan.q_assignments[r]
        rcount = sum(1 for qi in r_assigned if qi in set(my_range))
        recv_splits.append(rcount * q_numel)

    send_buf = (
        torch.cat(send_parts)
        if send_parts
        else torch.empty(0, device=dq_by_qi[next(iter(dq_by_qi))].device,
                         dtype=torch.float32)
    )
    recv_buf = _alltoall_v(send_buf, recv_splits, send_splits, cp_group)

    # Accumulate received dQ into local buffer
    local_dq = torch.zeros(*q_shape, device=recv_buf.device, dtype=torch.float32)
    offset = 0
    for r in range(cp_world_size):
        r_assigned = plan.q_assignments[r]
        for qi in sorted(qi for qi in r_assigned if qi in set(my_range)):
            chunk = recv_buf[offset : offset + q_numel].reshape(
                sc_size, n_heads, head_dim,
            )
            local_idx_start = (qi - cp_rank * spr) * sc_size
            local_dq[local_idx_start : local_idx_start + sc_size] += chunk
            offset += q_numel

    return local_dq
