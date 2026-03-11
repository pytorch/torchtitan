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
  - AllToAll-V wrapper (NVSHMEM symmetric memory or NCCL fallback)
  - Q redistribution (Phase A)
  - Tensor redistribution to assigned ranks
  - Result return (Phase C)
  - dK/dV and dQ scatter (backward phases)

Env vars:
  TORCHTITAN_MAGI_SUB_CHUNKS=2   -- sub-chunks per rank (default 2)
  TORCHTITAN_DISABLE_NVSHMEM=1   -- force NCCL fallback (disable NVSHMEM)
"""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist

from torchtitan.distributed.varlen_cp.dispatch_solver import (
    MagiDispatchPlan,
)

logger = logging.getLogger(__name__)

_MAGI_SUB_CHUNKS = int(os.environ.get("TORCHTITAN_MAGI_SUB_CHUNKS", "2"))

# ---------------------------------------------------------------------------
# NVSHMEM symmetric memory support
# ---------------------------------------------------------------------------

_HAS_NVSHMEM = False
_NVSHMEM_DISABLED = os.environ.get("TORCHTITAN_DISABLE_NVSHMEM", "0") == "1"

if not _NVSHMEM_DISABLED:
    try:
        import torch.distributed._symmetric_memory as _symm_mem

        _HAS_NVSHMEM = _symm_mem.is_nvshmem_available()
    except Exception:
        pass


class _NvshmemBufferPool:
    """Manages pre-allocated NVSHMEM symmetric memory buffers for AllToAll-V.

    ALL four tensors passed to ``torch.ops.symm_mem.all_to_all_vdev`` must
    reside in symmetric memory.  Buffers are allocated collectively (all ranks
    participate) and cached for reuse.  The pool auto-grows when a larger
    buffer is needed (also collective).

    Per (group_name, dtype) we keep symmetric *send* and *recv* buffers.
    Per group_name we keep shared int64 buffers for splits / offsets metadata.
    """

    # Per-(group, dtype): symmetric send + recv buffers
    _send_bufs: dict[tuple[str, torch.dtype], torch.Tensor] = {}
    _recv_bufs: dict[tuple[str, torch.dtype], torch.Tensor] = {}
    _buf_sizes: dict[tuple[str, torch.dtype], int] = {}

    # Per-group: shared int64 metadata buffers (symmetric)
    _splits_bufs: dict[str, torch.Tensor] = {}   # in_splits
    _offsets_bufs: dict[str, torch.Tensor] = {}   # out_splits_offsets (2*W)

    _initialized_groups: set[str] = set()

    @classmethod
    def _ensure_group(cls, group_name: str) -> None:
        if group_name in cls._initialized_groups:
            return
        _symm_mem.set_backend("NVSHMEM")
        _symm_mem.enable_symm_mem_for_group(group_name)
        cls._initialized_groups.add(group_name)

    @classmethod
    def _get_splits_buf(
        cls, group_name: str, world_size: int, device: torch.device,
    ) -> torch.Tensor:
        """Get (or create) symmetric in_splits buffer (int64, world_size)."""
        if group_name not in cls._splits_bufs:
            buf = _symm_mem.empty(world_size, dtype=torch.int64, device=device)
            _symm_mem.rendezvous(buf, group=group_name)
            cls._splits_bufs[group_name] = buf
        return cls._splits_bufs[group_name]

    @classmethod
    def _get_offsets_buf(
        cls, group_name: str, world_size: int, device: torch.device,
    ) -> torch.Tensor:
        """Get (or create) symmetric out_splits_offsets buffer (int64, 2*W)."""
        if group_name not in cls._offsets_bufs:
            buf = _symm_mem.empty(2 * world_size, dtype=torch.int64, device=device)
            _symm_mem.rendezvous(buf, group=group_name)
            cls._offsets_bufs[group_name] = buf
        return cls._offsets_bufs[group_name]

    @classmethod
    def _ensure_data_bufs(
        cls,
        group_name: str,
        max_numel: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get (or grow) symmetric send + recv buffers for (group, dtype)."""
        key = (group_name, dtype)
        current_size = cls._buf_sizes.get(key, 0)
        if current_size >= max_numel:
            return cls._send_bufs[key], cls._recv_bufs[key]

        # Grow: allocate new symmetric buffers (collective!)
        new_size = max(max_numel, current_size * 2)  # at least double
        send = _symm_mem.empty(new_size, dtype=dtype, device=device)
        _symm_mem.rendezvous(send, group=group_name)
        recv = _symm_mem.empty(new_size, dtype=dtype, device=device)
        _symm_mem.rendezvous(recv, group=group_name)
        cls._send_bufs[key] = send
        cls._recv_bufs[key] = recv
        cls._buf_sizes[key] = new_size
        logger.debug(
            "NVSHMEM buffers grown: group=%s dtype=%s size=%d",
            group_name, dtype, new_size,
        )
        return send, recv

    @classmethod
    def alltoall_v(
        cls,
        send_buf: torch.Tensor,
        recv_splits: list[int],
        send_splits: list[int],
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        """AllToAll-V using NVSHMEM symmetric memory."""
        group_name = group.group_name
        world_size = group.size()
        device = send_buf.device
        dtype = send_buf.dtype

        cls._ensure_group(group_name)

        total_send = send_buf.numel()
        total_recv = sum(recv_splits)

        if total_recv == 0 and total_send == 0:
            return torch.empty(0, device=device, dtype=dtype)

        # Ensure symmetric data buffers are large enough
        needed = max(total_send, total_recv, 1)
        sym_send, sym_recv = cls._ensure_data_bufs(
            group_name, needed, dtype, device,
        )

        # Copy send data into symmetric buffer
        if total_send > 0:
            sym_send[:total_send].copy_(send_buf)

        # Prepare in_splits (symmetric)
        in_splits = cls._get_splits_buf(group_name, world_size, device)
        in_splits.copy_(
            torch.tensor(send_splits, dtype=torch.int64, device=device),
        )

        # Prepare out_splits_offsets (symmetric, auto-filled by op)
        out_so = cls._get_offsets_buf(group_name, world_size, device)
        out_so_view = out_so.view(2, world_size)

        torch.ops.symm_mem.all_to_all_vdev(
            sym_send[:max(total_send, 1)],
            sym_recv[:max(total_recv, 1)],
            in_splits,
            out_so_view,
            group_name,
        )

        # Copy result out of symmetric buffer into a regular tensor
        if total_recv > 0:
            result = sym_recv[:total_recv].clone()
        else:
            result = torch.empty(0, device=device, dtype=dtype)
        return result


# ---------------------------------------------------------------------------
# AllToAll-V helpers
# ---------------------------------------------------------------------------


def _alltoall_v(
    send_buf: torch.Tensor,
    recv_splits: list[int],
    send_splits: list[int],
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Blocking AllToAll-V: NVSHMEM symmetric memory when available, NCCL fallback."""
    if _HAS_NVSHMEM:
        return _NvshmemBufferPool.alltoall_v(send_buf, recv_splits, send_splits, group)

    # NCCL fallback
    total_recv = sum(recv_splits)
    recv_buf = torch.empty(total_recv, device=send_buf.device, dtype=send_buf.dtype)
    if total_recv > 0 or sum(send_splits) > 0:
        dist.all_to_all_single(
            recv_buf, send_buf, recv_splits, send_splits, group=group,
        )
    return recv_buf


def _alltoall_v_nccl(
    send_buf: torch.Tensor,
    recv_splits: list[int],
    send_splits: list[int],
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Blocking AllToAll-V using NCCL only (no shared NVSHMEM buffers).

    Each call gets its own fresh recv buffer, making it safe for pipelined
    use on a separate CUDA stream where multiple recv buffers must coexist.
    Always calls the collective even with zero splits to avoid deadlock.
    """
    total_recv = sum(recv_splits)
    recv_buf = torch.empty(total_recv, device=send_buf.device, dtype=send_buf.dtype)
    # Always call the collective — even with all-zero splits — so all ranks
    # participate.  Skipping would deadlock if another rank has non-zero data.
    dist.all_to_all_single(
        recv_buf, send_buf, recv_splits, send_splits, group=group,
    )
    return recv_buf


def _alltoall_v_async(
    send_buf: torch.Tensor,
    recv_splits: list[int],
    send_splits: list[int],
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, dist.Work | None]:
    """Non-blocking NCCL AllToAll-V. Returns (recv_buf, work_handle).

    The caller must call ``work.wait()`` before reading from recv_buf.
    Falls back to blocking when NVSHMEM is used (NVSHMEM ops are already
    stream-async when launched on a non-default stream).
    """
    total_recv = sum(recv_splits)
    total_send = sum(send_splits)
    recv_buf = torch.empty(
        max(total_recv, 1), device=send_buf.device, dtype=send_buf.dtype,
    )
    if total_recv == 0 and total_send == 0:
        return recv_buf[:0], None

    work = dist.all_to_all_single(
        recv_buf, send_buf, recv_splits, send_splits, group=group,
        async_op=True,
    )
    return recv_buf[:total_recv], work


def preinit_nvshmem_buffers(
    group: dist.ProcessGroup,
    max_numel_bf16: int,
    max_numel_f32: int,
    device: torch.device,
) -> None:
    """Pre-allocate NVSHMEM symmetric buffers for AllToAll-V.

    Must be called collectively by all ranks in the group.  This avoids
    repeated lazy growth during the first training step.

    Args:
        group: CP process group.
        max_numel_bf16: Max elements for bf16 send buffers (forward path).
        max_numel_f32: Max elements for float32 send buffers (backward path).
        device: CUDA device.
    """
    if not _HAS_NVSHMEM:
        return
    group_name = group.group_name
    world_size = group.size()
    _NvshmemBufferPool._ensure_group(group_name)
    _NvshmemBufferPool._get_splits_buf(group_name, world_size, device)
    _NvshmemBufferPool._get_offsets_buf(group_name, world_size, device)
    if max_numel_bf16 > 0:
        _NvshmemBufferPool._ensure_data_bufs(
            group_name, max_numel_bf16, torch.bfloat16, device,
        )
    if max_numel_f32 > 0:
        _NvshmemBufferPool._ensure_data_bufs(
            group_name, max_numel_f32, torch.float32, device,
        )
    logger.info(
        "NVSHMEM buffers pre-allocated: bf16=%d f32=%d elements",
        max_numel_bf16, max_numel_f32,
    )


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


def _return_lse(
    lse_results: dict[int, torch.Tensor],
    plan: MagiDispatchPlan,
    cp_rank: int,
    cp_group: dist.ProcessGroup,
    n_heads: int,
) -> torch.Tensor:
    """Return computed LSE sub-chunks to original owners via AllToAll-V.

    LSE shape per sub-chunk: (n_heads, sub_chunk_size).
    """
    cp_world_size = plan.cp_world_size
    spr = plan.sub_chunks_per_rank
    sc_size = plan.sub_chunk_size
    sc_numel = n_heads * sc_size

    my_assigned = plan.q_assignments[cp_rank]
    my_range = range(cp_rank * spr, (cp_rank + 1) * spr)

    send_splits: list[int] = []
    send_parts: list[torch.Tensor] = []
    recv_splits: list[int] = []

    for r in range(cp_world_size):
        r_range = range(r * spr, (r + 1) * spr)
        count = 0
        for qi in sorted(set(my_assigned) & set(r_range)):
            send_parts.append(lse_results[qi].reshape(-1))
            count += 1
        send_splits.append(count * sc_numel)

        r_assigned = plan.q_assignments[r]
        rcount = sum(1 for qi in r_assigned if qi in set(my_range))
        recv_splits.append(rcount * sc_numel)

    send_buf = (
        torch.cat(send_parts)
        if send_parts
        else lse_results[my_assigned[0]].new_empty(0)
    )
    recv_buf = _alltoall_v(send_buf, recv_splits, send_splits, cp_group)

    lse_subs: dict[int, torch.Tensor] = {}
    offset = 0
    for r in range(cp_world_size):
        r_assigned = plan.q_assignments[r]
        for qi in sorted(qi for qi in r_assigned if qi in set(my_range)):
            lse_subs[qi] = recv_buf[offset : offset + sc_numel].reshape(
                n_heads, sc_size,
            )
            offset += sc_numel

    parts = [lse_subs[qi] for qi in sorted(lse_subs.keys())]
    return torch.cat(parts, dim=1)  # (n_heads, chunk_size)


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


def _scatter_dkv_to_owners(
    dkv_by_ki: dict[int, torch.Tensor],
    plan: MagiDispatchPlan,
    cp_rank: int,
    cp_group: dist.ProcessGroup,
    kv_shape: tuple[int, ...],
) -> torch.Tensor:
    """Scatter dK/dV sub-chunks back to their original K/V owners.

    This is the reverse of Phase B (gather K/V). Each assigned rank
    sends partial dK/dV gradients to the rank that owns that K sub-chunk.

    Args:
        dkv_by_ki: dict mapping K sub-chunk index -> dKV tensor (float32).
        plan: MagiDispatchPlan.
        cp_rank: Local rank.
        cp_group: Process group.
        kv_shape: Shape of local KV (chunk_size, n_kv_heads, kv_head_dim*2).

    Returns:
        Accumulated dKV for this rank's local K/V, shape kv_shape, float32.
    """
    cp_world_size = plan.cp_world_size
    spr = plan.sub_chunks_per_rank
    sc_size = plan.sub_chunk_size
    n_kv_heads = kv_shape[1]
    kv_head_dim = kv_shape[2]
    kv_numel = sc_size * n_kv_heads * kv_head_dim

    my_range = range(cp_rank * spr, (cp_rank + 1) * spr)

    # Send: for each K sub-chunk I computed backward for, send to its owner
    # Recv: from each rank that computed backward for my K sub-chunks
    send_splits: list[int] = []
    send_parts: list[torch.Tensor] = []
    recv_splits: list[int] = []

    for r in range(cp_world_size):
        # What I send to rank r: dKV for K sub-chunks owned by rank r
        r_k_range = range(r * spr, (r + 1) * spr)
        count = 0
        for ki in sorted(r_k_range):
            if ki in dkv_by_ki:
                send_parts.append(dkv_by_ki[ki].reshape(-1))
                count += 1
        send_splits.append(count * kv_numel)

        # What I recv from rank r: dKV for my K sub-chunks that rank r computed
        rcount = 0
        for ki in my_range:
            if plan.rank_needs_k(r, ki):
                rcount += 1
        recv_splits.append(rcount * kv_numel)

    send_buf = (
        torch.cat(send_parts)
        if send_parts
        else torch.empty(0, device=dkv_by_ki[next(iter(dkv_by_ki))].device,
                         dtype=torch.float32)
    )
    recv_buf = _alltoall_v(send_buf, recv_splits, send_splits, cp_group)

    # Accumulate received dKV into local buffer
    local_dkv = torch.zeros(*kv_shape, device=recv_buf.device, dtype=torch.float32)
    offset = 0
    for r in range(cp_world_size):
        for ki in my_range:
            if plan.rank_needs_k(r, ki):
                chunk = recv_buf[offset : offset + kv_numel].reshape(
                    sc_size, n_kv_heads, kv_head_dim,
                )
                local_idx_start = (ki - cp_rank * spr) * sc_size
                local_dkv[local_idx_start : local_idx_start + sc_size] += chunk
                offset += kv_numel

    return local_dkv


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
