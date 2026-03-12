# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Magi Attention for Context Parallelism.

Implementation based on:
  Magi Attention: Efficient Attention Mechanism for Extreme-Scale LLM Training
  Rui Men, Ke Yang, et al., 2025.
  https://arxiv.org/abs/2505.13211

Architecture:
  Phase A: Redistribute Q sub-chunks to assigned ranks via LPT load balancing
  Phase B: Gather K/V from owner ranks with per-doc packing (zero-redundancy)
  Compute: Batched FFA kernel per Q sub-chunk (fallback to varlen_attn)
  Phase C: Return output sub-chunks to original owners
  Optional: Multi-stage K/V fetch overlapped with compute (Section 4.4)

Backward:
  Reuses saved packed K/V from forward (no K/V re-gather needed):
  - Redistribute grad_out to assigned ranks (AllToAll-V)
  - Backward kernel with packed K/V (same layout as forward)
  - Scatter dK/dV and dQ back to owners (AllToAll-V)

Env vars:
  TORCHTITAN_MAGI_SUB_CHUNKS=2   -- sub-chunks per rank (default 2)
  TORCHTITAN_OVERLAP_STAGES=1    -- overlap stages for K/V fetch (1=no overlap)
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.distributed.varlen_cp.dispatch_solver import (
    MagiDispatchPlan,
    solve_magi_dispatch,
)

# PyTorch-native flex_attention wrappers (replace magi-attention FFA kernel)
from torchtitan.distributed.varlen_cp.flex_attn_kernels import (
    flex_attn_backward,
    flex_attn_forward,
)

# Reuse magi helpers that are unchanged
from torchtitan.distributed.varlen_cp.magi_dispatch import (  # noqa: E402
    _alltoall_v,
    _alltoall_v_nccl,
    _MAGI_SUB_CHUNKS,
    _redistribute_q,
    _redistribute_tensors_to_assigned,
    _return_lse,
    _return_results,
    _scatter_dq_to_owners,
    preinit_nvshmem_buffers,
)
from torchtitan.distributed.varlen_cp.mask_primitives import cu_seqlens_to_attn_slices
from torchtitan.distributed.varlen_cp.ring_attention import (
    _backward_step,
    _build_ring_step_cu_seqlens,
    _compute_and_merge_step,
    merge_with_lse,
)

_NUM_OVERLAP_STAGES = int(os.environ.get("TORCHTITAN_OVERLAP_STAGES", "1"))


# ---------------------------------------------------------------------------
# Cross-layer metadata cache
# ---------------------------------------------------------------------------
# Everything derived from global_cu_seqlens + plan (but NOT from Q/K/V values)
# is invariant across transformer layers within one forward pass. Cache it.


class _MagiMetadataCache:
    """Cache for metadata that doesn't change across layers.

    Keyed on (cu_seqlens_tuple, cp_world_size, sub_chunks_per_rank).
    Stores: magi_plan, AllToAll-V split sizes, FFA range descriptors, etc.
    """

    def __init__(self):
        self._key: tuple | None = None
        self.magi_plan: MagiDispatchPlan | None = None
        # Per-rank: (send_splits, recv_splits, recv_layout) for _gather_kv_packed
        self.gather_send_splits: dict[int, list[int]] = {}
        self.gather_recv_splits: dict[int, list[int]] = {}
        self.gather_recv_layout: dict[int, list[tuple[int, int]]] = {}
        # Per-rank: send metadata for K/V packing
        # gather_send_meta[rank][(target_rank, ki)] = list of (local_s, local_e) slices
        self.gather_send_meta: dict[
            int, list[tuple[int, int, list[tuple[int, int]]]]
        ] = {}
        # Per-(rank, ki): packed doc ranges
        self.packed_doc_ranges: dict[tuple[int, int], list[tuple[int, int]]] = {}
        # Per-(rank, qi): FFA ranges (q_ranges, k_ranges, attn_types) or None
        self.ffa_ranges: dict[tuple[int, int], tuple | None] = {}
        # Per-(rank, qi): q_coverage mask (bool tensor on CPU, transfer to GPU once)
        self.q_coverage: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def is_valid(self, key: tuple) -> bool:
        return self._key == key

    def set_key(self, key: tuple) -> None:
        self._key = key
        self.magi_plan = None
        self.gather_send_splits.clear()
        self.gather_recv_splits.clear()
        self.gather_recv_layout.clear()
        self.gather_send_meta.clear()
        self.packed_doc_ranges.clear()
        self.ffa_ranges.clear()
        self.q_coverage.clear()


_metadata_cache = _MagiMetadataCache()


def _precompute_gather_metadata(
    plan: MagiDispatchPlan,
    global_cu_seqlens: torch.Tensor,
    original_total_seqlen: int,
    cp_rank: int,
    n_kv_heads: int,
    kv_head_dim: int,
) -> tuple[
    list[int],
    list[int],
    list[tuple[int, int]],
    list[tuple[int, int, list[tuple[int, int]]]],
]:
    """Pre-compute AllToAll-V split sizes and send slice metadata for K/V gather.

    Returns:
        send_splits, recv_splits, recv_layout,
        send_meta: list of (target_rank_idx, ki_local_offset, slices) per send chunk
    """
    cp_world_size = plan.cp_world_size
    spr = plan.sub_chunks_per_rank
    sc_size = plan.sub_chunk_size
    my_range = range(cp_rank * spr, (cp_rank + 1) * spr)

    send_splits: list[int] = []
    recv_splits: list[int] = []
    recv_layout: list[tuple[int, int]] = []
    # send_meta: for each (r, ki) that we send, store the local slice positions
    send_meta: list[tuple[int, int, list[tuple[int, int]]]] = []

    for r in range(cp_world_size):
        send_count = 0
        for ki in my_range:
            if plan.rank_needs_k(r, ki):
                qi_list = plan.q_assignments[r]
                doc_ranges = _packed_k_doc_ranges(
                    global_cu_seqlens,
                    qi_list,
                    ki,
                    sc_size,
                    original_total_seqlen,
                )
                k_start = ki * sc_size
                ki_local_offset = (ki - cp_rank * spr) * sc_size
                slices = []
                n_tokens = 0
                for k_s, k_e in doc_ranges:
                    local_s = ki_local_offset + (k_s - k_start)
                    local_e = ki_local_offset + (k_e - k_start)
                    slices.append((local_s, local_e))
                    n_tokens += k_e - k_s
                send_meta.append((r, ki_local_offset, slices))
                send_count += n_tokens * n_kv_heads * kv_head_dim
        send_splits.append(send_count)

        recv_count = 0
        r_range = range(r * spr, (r + 1) * spr)
        for ki in r_range:
            if plan.rank_needs_k(cp_rank, ki):
                qi_list = plan.q_assignments[cp_rank]
                n_tokens = _packed_k_token_count(
                    global_cu_seqlens,
                    qi_list,
                    ki,
                    sc_size,
                    original_total_seqlen,
                )
                recv_count += n_tokens * n_kv_heads * kv_head_dim
                recv_layout.append((ki, n_tokens))
        recv_splits.append(recv_count)

    return send_splits, recv_splits, recv_layout, send_meta


def _precompute_ffa_ranges(
    plan: MagiDispatchPlan,
    global_cu_seqlens: torch.Tensor,
    original_total_seqlen: int,
    cp_rank: int,
    total_seqlen: int,
    device: torch.device,
    packed_doc_ranges_cache: dict[tuple[int, int], list[tuple[int, int]]],
) -> dict[int, tuple | None]:
    """Pre-compute FFA range descriptors for all qi assigned to cp_rank.

    packed_doc_ranges_cache must be populated first.
    Returns dict[qi] -> (q_ranges, k_ranges, attn_types) or None.
    """
    sc_size = plan.sub_chunk_size
    result: dict[int, tuple | None] = {}

    for qi in plan.q_assignments[cp_rank]:
        ki_list = plan.q_to_k_needs.get(qi, [])
        # We need packed_kv shapes to compute k_offsets, but those depend on
        # the actual recv layout. Instead, compute using the cached token counts.
        # Build a mock packed_kv with correct shapes (no actual data needed).
        mock_packed_kv: dict[int, torch.Tensor] = {}
        for ki in ki_list:
            key = (cp_rank, ki)
            if key in packed_doc_ranges_cache:
                n_tokens = sum(e - s for s, e in packed_doc_ranges_cache[key])
            else:
                n_tokens = 0
            # Only need shape[0] for range building
            mock_packed_kv[ki] = torch.empty(n_tokens, 0, 0)

        mock_doc_ranges = {
            ki: packed_doc_ranges_cache.get((cp_rank, ki), []) for ki in ki_list
        }

        ranges = _build_batched_ffa_ranges_packed(
            qi,
            ki_list,
            mock_packed_kv,
            mock_doc_ranges,
            global_cu_seqlens,
            sc_size,
            total_seqlen,
            original_total_seqlen,
            device,
        )
        result[qi] = ranges

    return result


def _get_or_build_cache(
    global_cu_seqlens: torch.Tensor,
    plan,  # DispatchPlan
    cp_rank: int,
    cp_world_size: int,
    n_kv_heads: int,
    kv_head_dim: int,
    total_seqlen: int,
    device: torch.device,
) -> tuple[MagiDispatchPlan, _MagiMetadataCache]:
    """Get or build the cross-layer metadata cache."""
    global _metadata_cache

    cache_key = (
        tuple(global_cu_seqlens.tolist()),
        cp_world_size,
        _MAGI_SUB_CHUNKS,
    )

    if _metadata_cache.is_valid(cache_key) and _metadata_cache.magi_plan is not None:
        return _metadata_cache.magi_plan, _metadata_cache

    # Cache miss — recompute everything
    _metadata_cache.set_key(cache_key)

    original_total_seqlen = total_seqlen - plan.pad_size
    chunk_size = plan.chunk_size

    global_slices = cu_seqlens_to_attn_slices(global_cu_seqlens)
    magi_plan = solve_magi_dispatch(
        global_slices,
        total_seqlen,
        chunk_size,
        cp_world_size,
        sub_chunks_per_rank=_MAGI_SUB_CHUNKS,
    )
    _metadata_cache.magi_plan = magi_plan
    sc_size = magi_plan.sub_chunk_size

    # Pre-compute gather metadata
    send_splits, recv_splits, recv_layout, send_meta = _precompute_gather_metadata(
        magi_plan,
        global_cu_seqlens,
        original_total_seqlen,
        cp_rank,
        n_kv_heads,
        kv_head_dim,
    )
    _metadata_cache.gather_send_splits[cp_rank] = send_splits
    _metadata_cache.gather_recv_splits[cp_rank] = recv_splits
    _metadata_cache.gather_recv_layout[cp_rank] = recv_layout
    _metadata_cache.gather_send_meta[cp_rank] = send_meta

    # Pre-compute packed doc ranges for recv side
    qi_list = magi_plan.q_assignments[cp_rank]
    spr = magi_plan.sub_chunks_per_rank
    for ki, _ in recv_layout:
        doc_ranges = _packed_k_doc_ranges(
            global_cu_seqlens,
            qi_list,
            ki,
            sc_size,
            original_total_seqlen,
        )
        _metadata_cache.packed_doc_ranges[(cp_rank, ki)] = doc_ranges

    # Pre-compute FFA ranges
    ffa = _precompute_ffa_ranges(
        magi_plan,
        global_cu_seqlens,
        original_total_seqlen,
        cp_rank,
        total_seqlen,
        device,
        _metadata_cache.packed_doc_ranges,
    )
    for qi, ranges in ffa.items():
        _metadata_cache.ffa_ranges[(cp_rank, qi)] = ranges
        # Pre-compute q_coverage ranges (for masking uncovered positions)
        if ranges is not None:
            _metadata_cache.q_coverage[(cp_rank, qi)] = ranges[0]  # q_ranges_l

    return magi_plan, _metadata_cache


# ---------------------------------------------------------------------------
# Per-doc packed K/V helpers (generalized from group_cast)
# ---------------------------------------------------------------------------


def _packed_k_doc_ranges(
    global_cu_seqlens: torch.Tensor,
    qi_list: list[int],
    ki: int,
    sc_size: int,
    original_total_seqlen: int,
) -> list[tuple[int, int]]:
    """Compute the union of K document ranges needed by ANY qi in qi_list.

    For each document, check if it overlaps the K sub-chunk [ki*sc_size, (ki+1)*sc_size)
    AND overlaps at least one Q sub-chunk in qi_list. If so, include its K range.

    Returns:
        Sorted list of (k_global_start, k_global_end) pairs.
    """
    k_start = ki * sc_size
    k_end = min(k_start + sc_size, original_total_seqlen)
    if k_end <= k_start:
        return []

    global_cu = global_cu_seqlens.tolist()
    num_docs = len(global_cu) - 1

    result: list[tuple[int, int]] = []
    for d in range(num_docs):
        doc_start = int(global_cu[d])
        doc_end = min(int(global_cu[d + 1]), original_total_seqlen)

        # K range within this doc
        k_s = max(doc_start, k_start)
        k_e = min(doc_end, k_end)
        if k_e <= k_s:
            continue

        # Check if ANY qi overlaps this doc
        found = False
        for qi in qi_list:
            q_start = qi * sc_size
            q_end = min(q_start + sc_size, original_total_seqlen)
            q_s = max(doc_start, q_start)
            q_e = min(doc_end, q_end)
            if q_e > q_s:
                # Also check causal: k_chunk_idx <= q_chunk_idx
                k_chunk_idx = k_start // sc_size
                q_chunk_idx = q_start // sc_size
                if k_chunk_idx <= q_chunk_idx:
                    found = True
                    break
        if found:
            result.append((k_s, k_e))

    return result


def _packed_k_token_count(
    global_cu_seqlens: torch.Tensor,
    qi_list: list[int],
    ki: int,
    sc_size: int,
    original_total_seqlen: int,
) -> int:
    """Count packed K tokens for ki needed by the union of qi_list."""
    ranges = _packed_k_doc_ranges(
        global_cu_seqlens,
        qi_list,
        ki,
        sc_size,
        original_total_seqlen,
    )
    return sum(e - s for s, e in ranges)


def _pack_kv_for_rank(
    local_kv: torch.Tensor,
    global_cu_seqlens: torch.Tensor,
    qi_list: list[int],
    ki: int,
    ki_local_offset: int,
    sc_size: int,
    original_total_seqlen: int,
) -> torch.Tensor:
    """Pack K/V tokens from sub-chunk ki needed by target rank's qi_list.

    Args:
        local_kv: This rank's full KV chunk, shape (chunk_size, n_kv_heads, kv_head_dim).
        global_cu_seqlens: Global cumulative sequence lengths.
        qi_list: Q sub-chunk indices assigned to the target rank.
        ki: K sub-chunk index (global).
        ki_local_offset: Local offset of ki within this rank's chunk.
        sc_size: Sub-chunk size.
        original_total_seqlen: Original total seqlen before padding.

    Returns:
        Flat 1-D packed tensor of needed K/V tokens.
    """
    doc_ranges = _packed_k_doc_ranges(
        global_cu_seqlens,
        qi_list,
        ki,
        sc_size,
        original_total_seqlen,
    )
    if not doc_ranges:
        return local_kv.new_empty(0)

    k_start = ki * sc_size
    slices: list[torch.Tensor] = []
    for k_s, k_e in doc_ranges:
        local_s = ki_local_offset + (k_s - k_start)
        local_e = ki_local_offset + (k_e - k_start)
        slices.append(local_kv[local_s:local_e].reshape(-1))

    return torch.cat(slices)


# ---------------------------------------------------------------------------
# Phase B: K/V gathering with per-doc packing
# ---------------------------------------------------------------------------


def _build_stage_gather_args(
    k: torch.Tensor,
    v: torch.Tensor,
    plan: MagiDispatchPlan,
    global_cu_seqlens: torch.Tensor,
    original_total_seqlen: int,
    cp_rank: int,
    ki_set: set[int],
) -> tuple[torch.Tensor, list[int], list[int], list[tuple[int, int]]]:
    """Build AllToAll-V args for gathering a specific set of K sub-chunks.

    Returns (send_buf, recv_splits, send_splits, recv_layout) where
    recv_layout is a list of (ki, n_packed_tokens) for unpacking.
    """
    cp_world_size = plan.cp_world_size
    spr = plan.sub_chunks_per_rank
    sc_size = plan.sub_chunk_size
    n_kv_heads = k.shape[1]
    kv_head_dim = k.shape[2] * 2

    local_kv = torch.cat([k, v], dim=-1)
    my_range = range(cp_rank * spr, (cp_rank + 1) * spr)

    send_splits: list[int] = []
    send_parts: list[torch.Tensor] = []
    recv_splits: list[int] = []
    recv_layout: list[tuple[int, int]] = []

    for r in range(cp_world_size):
        send_count = 0
        for ki in my_range:
            if ki not in ki_set:
                continue
            if plan.rank_needs_k(r, ki):
                qi_list = plan.q_assignments[r]
                packed = _pack_kv_for_rank(
                    local_kv,
                    global_cu_seqlens,
                    qi_list,
                    ki,
                    (ki - cp_rank * spr) * sc_size,
                    sc_size,
                    original_total_seqlen,
                )
                send_parts.append(packed)
                send_count += packed.numel()
        send_splits.append(send_count)

        recv_count = 0
        r_range = range(r * spr, (r + 1) * spr)
        for ki in r_range:
            if ki not in ki_set:
                continue
            if plan.rank_needs_k(cp_rank, ki):
                qi_list = plan.q_assignments[cp_rank]
                n_tokens = _packed_k_token_count(
                    global_cu_seqlens,
                    qi_list,
                    ki,
                    sc_size,
                    original_total_seqlen,
                )
                recv_count += n_tokens * n_kv_heads * kv_head_dim
                recv_layout.append((ki, n_tokens))
        recv_splits.append(recv_count)

    send_buf = torch.cat(send_parts) if send_parts else k.new_empty(0)
    return send_buf, recv_splits, send_splits, recv_layout


def _unpack_kv_recv(
    recv_buf: torch.Tensor,
    recv_layout: list[tuple[int, int]],
    plan: MagiDispatchPlan,
    global_cu_seqlens: torch.Tensor,
    original_total_seqlen: int,
    cp_rank: int,
    n_kv_heads: int,
    kv_head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[int, torch.Tensor], dict[int, list[tuple[int, int]]]]:
    """Unpack AllToAll-V recv buffer into packed_kv and packed_doc_ranges dicts."""
    sc_size = plan.sub_chunk_size
    packed_kv: dict[int, torch.Tensor] = {}
    packed_doc_ranges: dict[int, list[tuple[int, int]]] = {}

    offset = 0
    for ki, n_tokens in recv_layout:
        numel = n_tokens * n_kv_heads * kv_head_dim
        if numel > 0:
            packed_kv[ki] = recv_buf[offset : offset + numel].reshape(
                n_tokens,
                n_kv_heads,
                kv_head_dim,
            )
        else:
            packed_kv[ki] = torch.empty(
                0, n_kv_heads, kv_head_dim, device=device, dtype=dtype
            )
        offset += numel

        qi_list = plan.q_assignments[cp_rank]
        packed_doc_ranges[ki] = _packed_k_doc_ranges(
            global_cu_seqlens,
            qi_list,
            ki,
            sc_size,
            original_total_seqlen,
        )

    return packed_kv, packed_doc_ranges


def _gather_kv_packed(
    k: torch.Tensor,
    v: torch.Tensor,
    plan: MagiDispatchPlan,
    global_cu_seqlens: torch.Tensor,
    original_total_seqlen: int,
    cp_rank: int,
    cp_group: dist.ProcessGroup,
) -> tuple[
    dict[int, torch.Tensor],  # packed_kv[ki] = (n_packed, n_kv_heads, kv_head_dim)
    dict[int, list[tuple[int, int]]],  # packed_doc_ranges[ki] = [(k_gs, k_ge), ...]
]:
    """Gather needed K/V sub-chunks with per-document packing via AllToAll-V.

    Unlike the original magi _gather_kv which sends full sub-chunks, this
    only sends K/V tokens from documents that overlap the target rank's
    assigned Q sub-chunks. Zero-redundancy communication.

    Returns:
        packed_kv: dict[ki] -> packed KV tensor
        packed_doc_ranges: dict[ki] -> list of global (k_start, k_end) doc ranges
    """
    cp_world_size = plan.cp_world_size
    spr = plan.sub_chunks_per_rank
    sc_size = plan.sub_chunk_size
    n_kv_heads = k.shape[1]
    kv_head_dim = k.shape[2] * 2

    local_kv = torch.cat([k, v], dim=-1)
    my_range = range(cp_rank * spr, (cp_rank + 1) * spr)

    # Pre-compute: for each (target_rank, ki from my range), packed token counts
    send_splits: list[int] = []
    send_parts: list[torch.Tensor] = []
    recv_splits: list[int] = []
    # Track recv layout: list of (ki, n_packed_tokens) for unpacking
    recv_layout: list[tuple[int, int]] = []

    for r in range(cp_world_size):
        send_count = 0
        for ki in my_range:
            if plan.rank_needs_k(r, ki):
                qi_list = plan.q_assignments[r]
                packed = _pack_kv_for_rank(
                    local_kv,
                    global_cu_seqlens,
                    qi_list,
                    ki,
                    (ki - cp_rank * spr) * sc_size,
                    sc_size,
                    original_total_seqlen,
                )
                send_parts.append(packed)
                send_count += packed.numel()
        send_splits.append(send_count)

        recv_count = 0
        r_range = range(r * spr, (r + 1) * spr)
        for ki in r_range:
            if plan.rank_needs_k(cp_rank, ki):
                qi_list = plan.q_assignments[cp_rank]
                n_tokens = _packed_k_token_count(
                    global_cu_seqlens,
                    qi_list,
                    ki,
                    sc_size,
                    original_total_seqlen,
                )
                recv_count += n_tokens * n_kv_heads * kv_head_dim
                recv_layout.append((ki, n_tokens))
        recv_splits.append(recv_count)

    send_buf = torch.cat(send_parts) if send_parts else k.new_empty(0)
    recv_buf = _alltoall_v(send_buf, recv_splits, send_splits, cp_group)

    # Unpack received packed K/V
    half = kv_head_dim // 2
    packed_kv: dict[int, torch.Tensor] = {}
    packed_doc_ranges: dict[int, list[tuple[int, int]]] = {}

    offset = 0
    for ki, n_tokens in recv_layout:
        numel = n_tokens * n_kv_heads * kv_head_dim
        if numel > 0:
            kv_flat = recv_buf[offset : offset + numel]
            packed_kv[ki] = kv_flat.reshape(n_tokens, n_kv_heads, kv_head_dim)
        else:
            packed_kv[ki] = k.new_empty(0, n_kv_heads, kv_head_dim)
        offset += numel

        # Compute doc ranges for this ki (same as what the sender packed)
        qi_list = plan.q_assignments[cp_rank]
        packed_doc_ranges[ki] = _packed_k_doc_ranges(
            global_cu_seqlens,
            qi_list,
            ki,
            sc_size,
            original_total_seqlen,
        )

    return packed_kv, packed_doc_ranges


def _gather_kv_packed_cached(
    k: torch.Tensor,
    v: torch.Tensor,
    plan: MagiDispatchPlan,
    cp_rank: int,
    cp_group: dist.ProcessGroup,
    cache: _MagiMetadataCache,
) -> tuple[dict[int, torch.Tensor], dict[int, list[tuple[int, int]]]]:
    """Cached version of _gather_kv_packed — uses pre-computed splits and metadata."""
    n_kv_heads = k.shape[1]
    kv_head_dim = k.shape[2] * 2

    local_kv = torch.cat([k, v], dim=-1)

    # Build send buffer using cached slice metadata
    send_parts: list[torch.Tensor] = []
    for _r, _ki_offset, slices in cache.gather_send_meta[cp_rank]:
        for local_s, local_e in slices:
            send_parts.append(local_kv[local_s:local_e].reshape(-1))

    send_buf = torch.cat(send_parts) if send_parts else k.new_empty(0)
    send_splits = cache.gather_send_splits[cp_rank]
    recv_splits = cache.gather_recv_splits[cp_rank]
    recv_layout = cache.gather_recv_layout[cp_rank]

    recv_buf = _alltoall_v(send_buf, recv_splits, send_splits, cp_group)

    # Unpack using cached doc ranges
    packed_kv: dict[int, torch.Tensor] = {}
    packed_doc_ranges: dict[int, list[tuple[int, int]]] = {}
    device = k.device
    dtype = k.dtype

    offset = 0
    for ki, n_tokens in recv_layout:
        numel = n_tokens * n_kv_heads * kv_head_dim
        if numel > 0:
            packed_kv[ki] = recv_buf[offset : offset + numel].reshape(
                n_tokens,
                n_kv_heads,
                kv_head_dim,
            )
        else:
            packed_kv[ki] = torch.empty(
                0, n_kv_heads, kv_head_dim, device=device, dtype=dtype
            )
        offset += numel
        packed_doc_ranges[ki] = cache.packed_doc_ranges.get((cp_rank, ki), [])

    return packed_kv, packed_doc_ranges


# ---------------------------------------------------------------------------
# Batched FFA with packed K/V range descriptors
# ---------------------------------------------------------------------------


def _build_batched_ffa_ranges_packed(
    qi: int,
    ki_list: list[int],
    packed_kv: dict[int, torch.Tensor],
    packed_doc_ranges: dict[int, list[tuple[int, int]]],
    global_cu_seqlens: torch.Tensor,
    sc_size: int,
    total_seqlen: int,
    original_total_seqlen: int,
    device: torch.device,
) -> tuple[list[list[int]], list[list[int]], list[int]] | None:
    """Build FFA range descriptors for one Q sub-chunk vs packed K sub-chunks.

    Unlike _build_batched_ffa_ranges (magi mode) which assumes full sub-chunks,
    this uses per-doc packed K/V layout. The K offset for each ki is the
    cumulative packed token count of preceding ki's in ki_list.

    Returns (q_ranges, k_ranges, attn_types) or None if no work.
    """
    q_start = qi * sc_size
    q_end = min(q_start + sc_size, total_seqlen)

    all_q_ranges: list[list[int]] = []
    all_k_ranges: list[list[int]] = []
    all_attn_types: list[int] = []

    # Compute cumulative packed offset for each ki in ki_list
    k_offset = 0
    for ki in ki_list:
        k_global_start = ki * sc_size
        k_global_end = min(k_global_start + sc_size, total_seqlen)

        (
            _,
            _,
            q_doc_ranges,
            k_doc_ranges,
            num_real_q,
            is_causal,
        ) = _build_ring_step_cu_seqlens(
            global_cu_seqlens,
            q_start,
            q_end,
            k_global_start,
            k_global_end,
            original_total_seqlen,
            sc_size,
            device,
        )

        if num_real_q == 0 or not k_doc_ranges:
            if ki in packed_kv:
                k_offset += packed_kv[ki].shape[0]
            continue

        # Map k_doc_ranges from global coords to packed coords
        # packed_doc_ranges[ki] is the list of (k_gs, k_ge) that were packed
        doc_ranges_for_ki = packed_doc_ranges.get(ki, [])
        # Build a mapping from global K position to packed offset
        # The packed buffer for ki has docs concatenated in order
        packed_cu = [0]
        for dr_s, dr_e in doc_ranges_for_ki:
            packed_cu.append(packed_cu[-1] + (dr_e - dr_s))

        attn_type = 1 if is_causal else 0
        for (qs, qe), (ks, ke) in zip(q_doc_ranges, k_doc_ranges):
            # Find which packed doc this k_range belongs to
            packed_k_start = None
            for di, (dr_s, dr_e) in enumerate(doc_ranges_for_ki):
                if ks >= dr_s and ke <= dr_e:
                    # This k_range is within packed doc di
                    packed_k_start = k_offset + packed_cu[di] + (ks - dr_s)
                    packed_k_end = packed_k_start + (ke - ks)
                    break
            if packed_k_start is None:
                continue  # Should not happen if packing is correct

            all_q_ranges.append([qs - q_start, qe - q_start])
            all_k_ranges.append([packed_k_start, packed_k_end])
            all_attn_types.append(attn_type)

        if ki in packed_kv:
            k_offset += packed_kv[ki].shape[0]

    if not all_q_ranges:
        return None

    return all_q_ranges, all_k_ranges, all_attn_types


# ---------------------------------------------------------------------------
# Scatter packed dK/dV back to K/V owners
# ---------------------------------------------------------------------------


def _scatter_packed_dkv_to_owners(
    dkv_by_ki: dict[int, torch.Tensor],
    packed_doc_ranges: dict[int, list[tuple[int, int]]],
    global_cu_seqlens: torch.Tensor,
    plan: MagiDispatchPlan,
    cp_rank: int,
    cp_group: dist.ProcessGroup,
    chunk_size: int,
    n_kv_heads: int,
    kv_head_dim: int,
    device: torch.device,
    original_total_seqlen: int,
) -> torch.Tensor:
    """Scatter packed dK/dV back to K/V owners via AllToAll-V.

    Each dkv_by_ki[ki] is in packed layout (per-doc ranges). We send the
    packed gradients to the rank that owns ki, which unpacks and accumulates
    into the full sub-chunk positions.

    Returns:
        Local dKV, shape (chunk_size, n_kv_heads, kv_head_dim), float32.
    """
    cp_world_size = plan.cp_world_size
    spr = plan.sub_chunks_per_rank
    sc_size = plan.sub_chunk_size
    my_range = range(cp_rank * spr, (cp_rank + 1) * spr)

    send_splits: list[int] = []
    send_parts: list[torch.Tensor] = []
    recv_splits: list[int] = []
    recv_layout: list[tuple[int, list[tuple[int, int]]]] = []

    for r in range(cp_world_size):
        # Send: packed dKV for K sub-chunks owned by rank r
        r_k_range = range(r * spr, (r + 1) * spr)
        send_count = 0
        for ki in sorted(r_k_range):
            if ki in dkv_by_ki:
                flat = dkv_by_ki[ki].reshape(-1)
                send_parts.append(flat)
                send_count += flat.numel()
        send_splits.append(send_count)

        # Recv: packed dKV from rank r for my K sub-chunks that rank r computed
        recv_count = 0
        for ki in my_range:
            if plan.rank_needs_k(r, ki):
                qi_list_r = plan.q_assignments[r]
                n_tokens = _packed_k_token_count(
                    global_cu_seqlens,
                    qi_list_r,
                    ki,
                    sc_size,
                    original_total_seqlen,
                )
                numel = n_tokens * n_kv_heads * kv_head_dim
                recv_count += numel
                doc_ranges = _packed_k_doc_ranges(
                    global_cu_seqlens,
                    qi_list_r,
                    ki,
                    sc_size,
                    original_total_seqlen,
                )
                recv_layout.append((ki, doc_ranges))
        recv_splits.append(recv_count)

    send_buf = (
        torch.cat(send_parts)
        if send_parts
        else torch.empty(0, device=device, dtype=torch.float32)
    )
    recv_buf = _alltoall_v(send_buf, recv_splits, send_splits, cp_group)

    # Unpack and accumulate
    local_dkv = torch.zeros(
        chunk_size,
        n_kv_heads,
        kv_head_dim,
        device=device,
        dtype=torch.float32,
    )
    offset = 0
    for ki, doc_ranges in recv_layout:
        k_global_start = ki * sc_size
        local_ki_offset = (ki - cp_rank * spr) * sc_size
        for dr_s, dr_e in doc_ranges:
            n_tokens = dr_e - dr_s
            numel = n_tokens * n_kv_heads * kv_head_dim
            if numel > 0:
                chunk = recv_buf[offset : offset + numel].reshape(
                    n_tokens,
                    n_kv_heads,
                    kv_head_dim,
                )
                local_s = local_ki_offset + (dr_s - k_global_start)
                local_dkv[local_s : local_s + n_tokens] += chunk
            offset += numel

    return local_dkv


# ---------------------------------------------------------------------------
# Pipelined forward: multi-stage K/V fetch + FFA compute overlap
# ---------------------------------------------------------------------------


def _partition_ki_into_stages(
    plan: MagiDispatchPlan,
    num_stages: int,
) -> list[set[int]]:
    """Partition ALL K sub-chunks into ``num_stages`` groups.

    The partition is GLOBAL (same across all ranks) so that AllToAll-V
    collectives are called consistently by every rank.  Uses round-robin
    over the full range [0, total_sub_chunks).
    """
    total_sub_chunks = plan.cp_world_size * plan.sub_chunks_per_rank
    stages: list[set[int]] = [set() for _ in range(num_stages)]
    for ki in range(total_sub_chunks):
        stages[ki % num_stages].add(ki)
    return stages


def _ffa_compute_stage(
    assigned_q: dict[int, torch.Tensor],
    packed_kv: dict[int, torch.Tensor],
    packed_doc_ranges: dict[int, list[tuple[int, int]]],
    plan: MagiDispatchPlan,
    global_cu_seqlens: torch.Tensor,
    cp_rank: int,
    ki_set: set[int],
    sc_size: int,
    total_seqlen: int,
    original_total_seqlen: int,
    n_heads: int,
    head_dim: int,
    half_hd: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Compute FFA for assigned Q sub-chunks against a subset of K sub-chunks.

    Returns per-qi (output, LSE) dicts for this stage only.
    """
    out_results: dict[int, torch.Tensor] = {}
    lse_results: dict[int, torch.Tensor] = {}

    for qi in plan.q_assignments[cp_rank]:
        q_sub = assigned_q[qi]
        ki_list = [ki for ki in plan.q_to_k_needs.get(qi, []) if ki in ki_set]
        if not ki_list:
            continue

        ranges = _build_batched_ffa_ranges_packed(
            qi,
            ki_list,
            packed_kv,
            packed_doc_ranges,
            global_cu_seqlens,
            sc_size,
            total_seqlen,
            original_total_seqlen,
            device,
        )
        if ranges is None:
            continue

        q_ranges_l, k_ranges_l, attn_types_l = ranges
        k_parts = [
            packed_kv[ki][..., :half_hd]
            for ki in ki_list
            if ki in packed_kv and packed_kv[ki].shape[0] > 0
        ]
        v_parts = [
            packed_kv[ki][..., half_hd:]
            for ki in ki_list
            if ki in packed_kv and packed_kv[ki].shape[0] > 0
        ]
        if not k_parts:
            continue

        big_k = torch.cat(k_parts, dim=0)
        big_v = torch.cat(v_parts, dim=0)
        q_ranges_t = torch.tensor(q_ranges_l, dtype=torch.int32, device=device)
        k_ranges_t = torch.tensor(k_ranges_l, dtype=torch.int32, device=device)
        attn_type_map = torch.tensor(attn_types_l, dtype=torch.int32, device=device)

        step_out, step_meta = flex_attn_forward(
            q_sub,
            big_k,
            big_v,
            q_ranges_t,
            k_ranges_t,
            attn_type_map,
        )
        step_lse = step_meta.lse.transpose(0, 1)

        # Mask uncovered Q positions
        q_coverage = torch.zeros(sc_size, dtype=torch.bool, device=device)
        for qr in q_ranges_l:
            q_coverage[qr[0] : qr[1]] = True
        uncovered = ~q_coverage
        if uncovered.any():
            step_lse[:, uncovered] = float("-inf")
            step_out[uncovered] = 0

        out_results[qi] = step_out.to(dtype)
        lse_results[qi] = step_lse

    return out_results, lse_results


def _forward_pipelined(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    global_cu_seqlens: torch.Tensor,
    assigned_q: dict[int, torch.Tensor],
    magi_plan: MagiDispatchPlan,
    original_total_seqlen: int,
    cp_rank: int,
    cp_group: dist.ProcessGroup,
    num_stages: int,
    n_heads: int,
    head_dim: int,
    n_kv_heads: int,
    kv_head_dim: int,
    half_hd: int,
    sc_size: int,
    total_seqlen: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[
    dict[int, torch.Tensor],  # out_results
    dict[int, torch.Tensor],  # lse_results
    dict[int, torch.Tensor],  # packed_kv (all stages merged)
    dict[int, list[tuple[int, int]]],  # packed_doc_ranges
]:
    """Pipelined forward: multi-stage K/V fetch overlapped with FFA compute.

    Stage i's FFA compute overlaps with stage i+1's K/V AllToAll-V.

    Timeline::

        comm_stream:  [fetch s0] [fetch s1]         [fetch s2]         ...
        default:               [FFA s0  ]  [FFA s1  ]  [FFA s2  ]
    """
    stage_ki = _partition_ki_into_stages(magi_plan, num_stages)

    # Initialize accumulators
    all_qi = magi_plan.q_assignments[cp_rank]
    accum_out: dict[int, torch.Tensor] = {}
    accum_lse: dict[int, torch.Tensor] = {}
    for qi in all_qi:
        accum_out[qi] = torch.zeros(
            sc_size,
            n_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        accum_lse[qi] = torch.full(
            (n_heads, sc_size),
            float("-inf"),
            device=device,
            dtype=torch.float32,
        )

    all_packed_kv: dict[int, torch.Tensor] = {}
    all_packed_doc_ranges: dict[int, list[tuple[int, int]]] = {}

    # Pre-build send args for all stages (local K/V doesn't change).
    # IMPORTANT: Even empty stages must produce valid zero-split args
    # so that ALL ranks call AllToAll-V collectively for every stage.
    stage_args: list[tuple] = []
    cp_world_size = magi_plan.cp_world_size
    for s in range(num_stages):
        if not stage_ki[s]:
            # Empty stage: all-zero splits, still call the collective
            zero_splits = [0] * cp_world_size
            stage_args.append((k.new_empty(0), zero_splits, zero_splits, []))
            continue
        args = _build_stage_gather_args(
            k,
            v,
            magi_plan,
            global_cu_seqlens,
            original_total_seqlen,
            cp_rank,
            stage_ki[s],
        )
        stage_args.append(args)

    # Use a separate CUDA stream for communication.  The blocking
    # _alltoall_v (NVSHMEM or NCCL) runs on comm_stream; FFA compute
    # runs on the default stream.  The two overlap naturally.
    comm_stream = torch.cuda.Stream(device=device)

    # Pre-fetch results storage (one per stage, no double-buffer needed
    # since we wait for each stage before reusing)
    stage_recv: list[torch.Tensor | None] = [None] * num_stages
    stage_layout: list[list[tuple[int, int]]] = [[] for _ in range(num_stages)]

    def _fetch_on_comm_stream(s: int) -> None:
        send_buf, recv_splits, send_splits, recv_layout = stage_args[s]
        # Always call AllToAll-V — even with zero splits — so all ranks
        # participate in the collective (skipping would deadlock).
        with torch.cuda.stream(comm_stream):
            stage_recv[s] = _alltoall_v_nccl(
                send_buf,
                recv_splits,
                send_splits,
                cp_group,
            )
        stage_layout[s] = recv_layout

    # === Pipeline loop ===
    # Launch stage 0 fetch
    _fetch_on_comm_stream(0)

    for s in range(num_stages):
        # Wait for current stage's K/V to arrive
        torch.cuda.current_stream().wait_stream(comm_stream)

        # Launch NEXT stage's fetch (overlaps with FFA compute below)
        if s + 1 < num_stages:
            # comm_stream must wait for default stream to finish using
            # the previous recv buffer before we can reuse NVSHMEM pool
            comm_stream.wait_stream(torch.cuda.current_stream())
            _fetch_on_comm_stream(s + 1)

        # Unpack current stage's K/V
        recv = stage_recv[s]
        if recv is not None and recv.numel() > 0:
            stage_kv, stage_doc_ranges = _unpack_kv_recv(
                recv,
                stage_layout[s],
                magi_plan,
                global_cu_seqlens,
                original_total_seqlen,
                cp_rank,
                n_kv_heads,
                kv_head_dim,
                device,
                dtype,
            )
        else:
            stage_kv, stage_doc_ranges = {}, {}

        # Compute FFA for this stage (on default stream)
        if stage_kv:
            stage_out, stage_lse = _ffa_compute_stage(
                assigned_q,
                stage_kv,
                stage_doc_ranges,
                magi_plan,
                global_cu_seqlens,
                cp_rank,
                stage_ki[s],
                sc_size,
                total_seqlen,
                original_total_seqlen,
                n_heads,
                head_dim,
                half_hd,
                device,
                dtype,
            )

            # Merge into accumulators
            for qi in stage_out:
                accum_out[qi], accum_lse[qi] = merge_with_lse(
                    accum_out[qi],
                    accum_lse[qi],
                    stage_out[qi],
                    stage_lse[qi],
                )

        # Save packed K/V for backward
        all_packed_kv.update(stage_kv)
        all_packed_doc_ranges.update(stage_doc_ranges)

    return accum_out, accum_lse, all_packed_kv, all_packed_doc_ranges


# ---------------------------------------------------------------------------
# Custom autograd.Function
# ---------------------------------------------------------------------------


class _VarlenMagiFunc(torch.autograd.Function):
    """Magi Attention: Q redistribution + per-doc packed K/V.

    Forward: LPT Q redistribution + packed K/V gather + batched FFA compute.
    Backward: Redistribute grad_out, compute with saved packed K/V, scatter dK/dV and dQ.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        global_cu_seqlens,
        plan,  # DispatchPlan (original)
        cp_group,
        cp_rank,
        cp_world_size,
    ):
        chunk_size = plan.chunk_size
        total_seqlen = plan.total_seqlen
        original_total_seqlen = total_seqlen - plan.pad_size

        n_heads = q.shape[1]
        head_dim = q.shape[2]
        n_kv_heads = k.shape[1]
        kv_head_dim = k.shape[2] * 2
        half_hd = k.shape[2]
        dtype = q.dtype
        device = q.device

        # Pre-allocate NVSHMEM symmetric memory buffers (collective).
        max_feat_bf16 = max(n_heads * head_dim, n_kv_heads * kv_head_dim)
        max_feat_f32 = max_feat_bf16
        preinit_nvshmem_buffers(
            cp_group,
            max_numel_bf16=chunk_size * max_feat_bf16,
            max_numel_f32=chunk_size * max_feat_f32,
            device=device,
        )

        # Build or retrieve cached magi plan + metadata
        magi_plan, cache = _get_or_build_cache(
            global_cu_seqlens,
            plan,
            cp_rank,
            cp_world_size,
            n_kv_heads,
            kv_head_dim,
            total_seqlen,
            device,
        )
        sc_size = magi_plan.sub_chunk_size

        with torch.no_grad():
            # Phase A: Redistribute Q
            assigned_q = _redistribute_q(q, magi_plan, cp_rank, cp_group)

            # Phase B + Compute: pipelined or non-pipelined path
            num_stages = _NUM_OVERLAP_STAGES

            block_mask_cache: dict[int, object] = {}
            pipelined_ok = False
            if num_stages > 1:
                try:
                    (
                        out_results,
                        lse_results,
                        packed_kv,
                        packed_doc_ranges,
                    ) = _forward_pipelined(
                        q,
                        k,
                        v,
                        global_cu_seqlens,
                        assigned_q,
                        magi_plan,
                        original_total_seqlen,
                        cp_rank,
                        cp_group,
                        num_stages,
                        n_heads,
                        head_dim,
                        n_kv_heads,
                        kv_head_dim,
                        half_hd,
                        sc_size,
                        total_seqlen,
                        dtype,
                        device,
                    )
                    batched_ok = True
                    pipelined_ok = True
                except RuntimeError:
                    pass

            if not pipelined_ok:
                # Use cached gather when available
                if cache.gather_send_splits.get(cp_rank) is not None:
                    packed_kv, packed_doc_ranges = _gather_kv_packed_cached(
                        k,
                        v,
                        magi_plan,
                        cp_rank,
                        cp_group,
                        cache,
                    )
                else:
                    packed_kv, packed_doc_ranges = _gather_kv_packed(
                        k,
                        v,
                        magi_plan,
                        global_cu_seqlens,
                        original_total_seqlen,
                        cp_rank,
                        cp_group,
                    )

                out_results: dict[int, torch.Tensor] = {}
                lse_results: dict[int, torch.Tensor] = {}

                batched_ok = False
                try:
                    for qi in magi_plan.q_assignments[cp_rank]:
                        q_sub = assigned_q[qi]
                        ki_list = magi_plan.q_to_k_needs.get(qi, [])

                        # Use cached FFA ranges when available
                        cache_key_qi = (cp_rank, qi)
                        if cache_key_qi in cache.ffa_ranges:
                            ranges = cache.ffa_ranges[cache_key_qi]
                        else:
                            ranges = _build_batched_ffa_ranges_packed(
                                qi,
                                ki_list,
                                packed_kv,
                                packed_doc_ranges,
                                global_cu_seqlens,
                                sc_size,
                                total_seqlen,
                                original_total_seqlen,
                                device,
                            )
                        if ranges is None:
                            out_results[qi] = torch.zeros(
                                sc_size,
                                n_heads,
                                head_dim,
                                device=device,
                                dtype=dtype,
                            )
                            lse_results[qi] = torch.full(
                                (n_heads, sc_size),
                                float("-inf"),
                                device=device,
                                dtype=torch.float32,
                            )
                            continue

                        q_ranges_l, k_ranges_l, attn_types_l = ranges
                        k_parts = [
                            packed_kv[ki][..., :half_hd]
                            for ki in ki_list
                            if ki in packed_kv and packed_kv[ki].shape[0] > 0
                        ]
                        v_parts = [
                            packed_kv[ki][..., half_hd:]
                            for ki in ki_list
                            if ki in packed_kv and packed_kv[ki].shape[0] > 0
                        ]
                        if not k_parts:
                            out_results[qi] = torch.zeros(
                                sc_size,
                                n_heads,
                                head_dim,
                                device=device,
                                dtype=dtype,
                            )
                            lse_results[qi] = torch.full(
                                (n_heads, sc_size),
                                float("-inf"),
                                device=device,
                                dtype=torch.float32,
                            )
                            continue

                        big_k = torch.cat(k_parts, dim=0)
                        big_v = torch.cat(v_parts, dim=0)
                        q_ranges_t = torch.tensor(
                            q_ranges_l,
                            dtype=torch.int32,
                            device=device,
                        )
                        k_ranges_t = torch.tensor(
                            k_ranges_l,
                            dtype=torch.int32,
                            device=device,
                        )
                        attn_type_map = torch.tensor(
                            attn_types_l,
                            dtype=torch.int32,
                            device=device,
                        )

                        step_out, step_meta = flex_attn_forward(
                            q_sub,
                            big_k,
                            big_v,
                            q_ranges_t,
                            k_ranges_t,
                            attn_type_map,
                        )
                        step_lse = step_meta.lse.transpose(0, 1)

                        # Cache BlockMask for backward reuse
                        if step_meta.block_mask is not None:
                            block_mask_cache[qi] = step_meta.block_mask

                        q_coverage = torch.zeros(
                            sc_size,
                            dtype=torch.bool,
                            device=device,
                        )
                        for qr in q_ranges_l:
                            q_coverage[qr[0] : qr[1]] = True
                        uncovered = ~q_coverage
                        if uncovered.any():
                            step_lse[:, uncovered] = float("-inf")
                            step_out[uncovered] = 0

                        out_results[qi] = step_out.to(dtype)
                        lse_results[qi] = step_lse

                    batched_ok = True
                except RuntimeError:
                    out_results.clear()
                    lse_results.clear()

                if not batched_ok:
                    # Fallback: per-(qi, ki) pair loop with unpacked K/V
                    for qi in magi_plan.q_assignments[cp_rank]:
                        q_sub = assigned_q[qi]
                        q_start = qi * sc_size
                        q_end = min(q_start + sc_size, total_seqlen)

                        accum_out = torch.zeros(
                            sc_size,
                            n_heads,
                            head_dim,
                            device=device,
                            dtype=dtype,
                        )
                        accum_lse = torch.full(
                            (n_heads, sc_size),
                            float("-inf"),
                            device=device,
                            dtype=torch.float32,
                        )

                        for ki in magi_plan.q_to_k_needs.get(qi, []):
                            if ki not in packed_kv or packed_kv[ki].shape[0] == 0:
                                continue
                            full_k = torch.zeros(
                                sc_size,
                                n_kv_heads,
                                half_hd,
                                device=device,
                                dtype=dtype,
                            )
                            full_v = torch.zeros(
                                sc_size,
                                n_kv_heads,
                                half_hd,
                                device=device,
                                dtype=dtype,
                            )
                            k_global_start = ki * sc_size
                            doc_ranges = packed_doc_ranges.get(ki, [])
                            packed_offset = 0
                            for dr_s, dr_e in doc_ranges:
                                n_tok = dr_e - dr_s
                                local_s = dr_s - k_global_start
                                full_k[local_s : local_s + n_tok] = packed_kv[ki][
                                    packed_offset : packed_offset + n_tok, :, :half_hd
                                ]
                                full_v[local_s : local_s + n_tok] = packed_kv[ki][
                                    packed_offset : packed_offset + n_tok, :, half_hd:
                                ]
                                packed_offset += n_tok

                            k_start = ki * sc_size
                            k_end = min(k_start + sc_size, total_seqlen)
                            accum_out, accum_lse = _compute_and_merge_step(
                                q_sub,
                                full_k,
                                full_v,
                                global_cu_seqlens,
                                q_start,
                                q_end,
                                k_start,
                                k_end,
                                original_total_seqlen,
                                sc_size,
                                accum_out,
                                accum_lse,
                            )

                        out_results[qi] = accum_out.to(dtype)
                        lse_results[qi] = accum_lse

            # Phase C: Return output to original owners (1 AllToAll-V)
            output = _return_results(
                out_results,
                magi_plan,
                cp_rank,
                cp_group,
                n_heads,
                head_dim,
            )

        merged_out = output.contiguous()

        # Save for backward: assigned Q, packed K/V, per-qi out+lse, doc ranges
        assigned_q_keys = sorted(assigned_q.keys())
        packed_kv_keys = sorted(packed_kv.keys())
        assigned_out_keys = sorted(out_results.keys())

        save_list = [
            q,
            k,
            v,
            global_cu_seqlens,
            # Assigned Q sub-chunks
            *[assigned_q[qi] for qi in assigned_q_keys],
            # Packed K/V (already concatenated K+V in dim=-1)
            *[packed_kv[ki] for ki in packed_kv_keys],
            # Per-qi output and LSE
            *[out_results[qi] for qi in assigned_out_keys],
            *[lse_results[qi] for qi in assigned_out_keys],
        ]
        ctx.save_for_backward(*save_list)
        ctx.n_assigned_q = len(assigned_q_keys)
        ctx.n_packed_kv = len(packed_kv_keys)
        ctx.n_assigned_out = len(assigned_out_keys)
        ctx.assigned_q_keys = assigned_q_keys
        ctx.packed_kv_keys = packed_kv_keys
        ctx.assigned_out_keys = assigned_out_keys
        ctx.packed_doc_ranges = packed_doc_ranges
        ctx.plan = plan
        ctx.magi_plan = magi_plan
        ctx.cp_group = cp_group
        ctx.cp_rank = cp_rank
        ctx.cp_world_size = cp_world_size
        # Cache BlockMask per qi for backward reuse (avoids create_block_mask)
        ctx.block_mask_cache = block_mask_cache if batched_ok else {}

        return merged_out

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        q = saved[0]
        k = saved[1]
        v = saved[2]
        global_cu_seqlens = saved[3]

        base = 4
        nq = ctx.n_assigned_q
        nkv = ctx.n_packed_kv
        nout = ctx.n_assigned_out

        assigned_q = dict(zip(ctx.assigned_q_keys, saved[base : base + nq]))
        base += nq
        packed_kv = dict(zip(ctx.packed_kv_keys, saved[base : base + nkv]))
        base += nkv
        assigned_out = dict(zip(ctx.assigned_out_keys, saved[base : base + nout]))
        assigned_lse = dict(
            zip(
                ctx.assigned_out_keys,
                saved[base + nout : base + 2 * nout],
            )
        )

        packed_doc_ranges = ctx.packed_doc_ranges
        plan = ctx.plan
        magi_plan = ctx.magi_plan
        cp_group = ctx.cp_group
        cp_rank = ctx.cp_rank

        sc_size = magi_plan.sub_chunk_size
        total_seqlen = plan.total_seqlen
        original_total_seqlen = total_seqlen - plan.pad_size
        n_kv_heads = k.shape[1]
        half_hd = k.shape[2]
        kv_head_dim = half_hd * 2
        dtype = q.dtype
        device = q.device
        chunk_size = plan.chunk_size

        with torch.no_grad():
            # Phase A': Redistribute grad_out to assigned ranks
            assigned_grad_out = _redistribute_tensors_to_assigned(
                [grad_output],
                magi_plan,
                cp_rank,
                cp_group,
            )[0]

            # Compute backward for each assigned Q sub-chunk
            dq_by_qi: dict[int, torch.Tensor] = {}
            # dkv_by_ki: packed layout (matching forward packed_kv)
            dkv_by_ki: dict[int, torch.Tensor] = {}

            block_mask_cache = ctx.block_mask_cache
            batched_bwd_ok = False
            try:
                for qi in magi_plan.q_assignments[cp_rank]:
                    q_sub = assigned_q[qi]
                    grad_out_sub = assigned_grad_out[qi]
                    out_sub = assigned_out[qi]
                    lse_sub = assigned_lse[qi]
                    ki_list = magi_plan.q_to_k_needs.get(qi, [])

                    ranges = _build_batched_ffa_ranges_packed(
                        qi,
                        ki_list,
                        packed_kv,
                        packed_doc_ranges,
                        global_cu_seqlens,
                        sc_size,
                        total_seqlen,
                        original_total_seqlen,
                        device,
                    )
                    if ranges is None:
                        dq_by_qi[qi] = torch.zeros_like(
                            q_sub,
                            dtype=torch.float32,
                        )
                        continue

                    q_ranges_l, k_ranges_l, attn_types_l = ranges

                    k_parts = [
                        packed_kv[ki][..., :half_hd]
                        for ki in ki_list
                        if ki in packed_kv and packed_kv[ki].shape[0] > 0
                    ]
                    v_parts = [
                        packed_kv[ki][..., half_hd:]
                        for ki in ki_list
                        if ki in packed_kv and packed_kv[ki].shape[0] > 0
                    ]
                    if not k_parts:
                        dq_by_qi[qi] = torch.zeros_like(
                            q_sub,
                            dtype=torch.float32,
                        )
                        continue

                    big_k = torch.cat(k_parts, dim=0)
                    big_v = torch.cat(v_parts, dim=0)
                    q_ranges_t = torch.tensor(
                        q_ranges_l,
                        dtype=torch.int32,
                        device=device,
                    )
                    k_ranges_t = torch.tensor(
                        k_ranges_l,
                        dtype=torch.int32,
                        device=device,
                    )
                    attn_type_map = torch.tensor(
                        attn_types_l,
                        dtype=torch.int32,
                        device=device,
                    )

                    ffa_lse = lse_sub.transpose(0, 1).contiguous()

                    dq, dk_big, dv_big = flex_attn_backward(
                        grad_out_sub.contiguous(),
                        q_sub.contiguous(),
                        big_k.contiguous(),
                        big_v.contiguous(),
                        out_sub.contiguous(),
                        ffa_lse,
                        q_ranges_t,
                        k_ranges_t,
                        attn_type_map,
                        block_mask=block_mask_cache.get(qi),
                    )

                    dq_by_qi[qi] = dq

                    # Split dk_big/dv_big per ki (packed layout)
                    k_offset = 0
                    for ki in ki_list:
                        if ki not in packed_kv or packed_kv[ki].shape[0] == 0:
                            continue
                        n_tok = packed_kv[ki].shape[0]
                        if ki not in dkv_by_ki:
                            dkv_by_ki[ki] = torch.zeros(
                                n_tok,
                                n_kv_heads,
                                kv_head_dim,
                                device=device,
                                dtype=torch.float32,
                            )
                        dkv_by_ki[ki][..., :half_hd] += dk_big[
                            k_offset : k_offset + n_tok
                        ]
                        dkv_by_ki[ki][..., half_hd:] += dv_big[
                            k_offset : k_offset + n_tok
                        ]
                        k_offset += n_tok

                batched_bwd_ok = True
            except RuntimeError:
                dq_by_qi.clear()
                dkv_by_ki.clear()

            if not batched_bwd_ok:
                # Fallback: per-(qi, ki) pair loop
                for qi in magi_plan.q_assignments[cp_rank]:
                    q_sub = assigned_q[qi]
                    grad_out_sub = assigned_grad_out[qi]
                    out_sub = assigned_out[qi]
                    lse_sub = assigned_lse[qi]
                    q_start = qi * sc_size
                    q_end = min(q_start + sc_size, total_seqlen)
                    grad_q_sub = torch.zeros_like(q_sub, dtype=torch.float32)

                    for ki in magi_plan.q_to_k_needs.get(qi, []):
                        if ki not in packed_kv or packed_kv[ki].shape[0] == 0:
                            continue
                        # Unpack to full sub-chunk for fallback
                        full_k = torch.zeros(
                            sc_size,
                            n_kv_heads,
                            half_hd,
                            device=device,
                            dtype=dtype,
                        )
                        full_v = torch.zeros(
                            sc_size,
                            n_kv_heads,
                            half_hd,
                            device=device,
                            dtype=dtype,
                        )
                        k_global_start = ki * sc_size
                        doc_ranges = packed_doc_ranges.get(ki, [])
                        packed_offset = 0
                        for dr_s, dr_e in doc_ranges:
                            n_tok = dr_e - dr_s
                            local_s = dr_s - k_global_start
                            full_k[local_s : local_s + n_tok] = packed_kv[ki][
                                packed_offset : packed_offset + n_tok, :, :half_hd
                            ]
                            full_v[local_s : local_s + n_tok] = packed_kv[ki][
                                packed_offset : packed_offset + n_tok, :, half_hd:
                            ]
                            packed_offset += n_tok

                        k_start = ki * sc_size
                        k_end = min(k_start + sc_size, total_seqlen)
                        # Need full sub-chunk dkv for scatter
                        if ki not in dkv_by_ki:
                            dkv_by_ki[ki] = torch.zeros(
                                sc_size,
                                n_kv_heads,
                                kv_head_dim,
                                device=device,
                                dtype=torch.float32,
                            )

                        _backward_step(
                            grad_out_sub,
                            q_sub,
                            full_k,
                            full_v,
                            out_sub,
                            lse_sub,
                            global_cu_seqlens,
                            q_start,
                            q_end,
                            k_start,
                            k_end,
                            original_total_seqlen,
                            sc_size,
                            grad_q_sub,
                            dkv_by_ki[ki],
                        )

                    dq_by_qi[qi] = grad_q_sub

            # Phase B': Scatter dK/dV back to K/V owners
            if batched_bwd_ok:
                # Packed layout: use packed scatter
                local_dkv = _scatter_packed_dkv_to_owners(
                    dkv_by_ki,
                    packed_doc_ranges,
                    global_cu_seqlens,
                    magi_plan,
                    cp_rank,
                    cp_group,
                    chunk_size,
                    n_kv_heads,
                    kv_head_dim,
                    device,
                    original_total_seqlen,
                )
            else:
                # Full sub-chunk layout: use original scatter
                from torchtitan.distributed.varlen_cp.magi_dispatch import (
                    _scatter_dkv_to_owners,
                )

                kv_shape = (chunk_size, n_kv_heads, kv_head_dim)
                local_dkv = _scatter_dkv_to_owners(
                    dkv_by_ki,
                    magi_plan,
                    cp_rank,
                    cp_group,
                    kv_shape,
                )

            # Phase A': Scatter dQ back to Q owners
            q_shape = q.shape
            local_dq = _scatter_dq_to_owners(
                dq_by_qi,
                magi_plan,
                cp_rank,
                cp_group,
                q_shape,
            )

        half = kv_head_dim // 2
        return (
            local_dq.to(dtype).contiguous(),
            local_dkv[:, :, :half].to(dtype).contiguous(),
            local_dkv[:, :, half:].to(dtype).contiguous(),
            None,
            None,
            None,
            None,
            None,
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def varlen_magi_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    global_cu_seqlens: torch.Tensor,
    plan: "DispatchPlan",
    cp_mesh: DeviceMesh,
) -> torch.Tensor:
    """Magi Attention for context parallelism.

    LPT load balancing with zero-redundancy K/V communication.
    See: Men et al., "Magi Attention", https://arxiv.org/abs/2505.13211

    Args:
        q: Local Q chunk, shape (chunk_size, n_heads, head_dim).
        k: Local K chunk, shape (chunk_size, n_kv_heads, head_dim).
        v: Local V chunk, shape (chunk_size, n_kv_heads, head_dim).
        global_cu_seqlens: Global cumulative sequence lengths.
        plan: Original DispatchPlan (used only for metadata).
        cp_mesh: DeviceMesh for CP group.

    Returns:
        Output tensor, shape (chunk_size, n_heads, head_dim).
    """
    cp_rank = cp_mesh.get_local_rank()
    cp_group = cp_mesh.get_group()
    cp_world_size = cp_mesh.size(0)

    return _VarlenMagiFunc.apply(
        q,
        k,
        v,
        global_cu_seqlens,
        plan,
        cp_group,
        cp_rank,
        cp_world_size,
    )
