# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Varlen Attention with Context Parallelism (Magi Attention).

Based on: Men et al., "Magi Attention: Efficient Attention Mechanism for
Extreme-Scale LLM Training", https://arxiv.org/abs/2505.13211

Entry point: varlen_ring_attention() delegates to varlen_magi_dispatch().
Also provides shared utilities: FFA kernel wrappers, LSE merge, backward steps.
"""

from __future__ import annotations

import torch
from torch.distributed.device_mesh import DeviceMesh
from torchtitan.distributed.varlen_cp.dispatch_solver import DispatchPlan

# PyTorch-native flex_attention wrappers for range-based attention.
# These replace the external magi-attention FFA kernel dependency with
# torch.nn.attention.flex_attention (compiled Triton kernel).
from torchtitan.distributed.varlen_cp.flex_attn_kernels import (
    flex_attn_backward,
    flex_attn_forward,
)


# ---------------------------------------------------------------------------
# LSE merge
# ---------------------------------------------------------------------------


def _safe_subtract(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Subtract two tensors, returning -inf (not NaN) for -inf - (-inf)."""
    two_neg_inf = (a == b) & (a == float("-inf"))
    return (a - b).masked_fill(two_neg_inf, float("-inf"))


def merge_with_lse(
    out1: torch.Tensor,
    lse1: torch.Tensor,
    out2: torch.Tensor,
    lse2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Numerically stable merge of two partial attention outputs using LSE.

    Uses the softplus formulation from magi-attention for improved numerical
    stability::

        merged_lse = max(lse1, lse2) + softplus(min(lse1, lse2) - max(lse1, lse2))

    All intermediate computation is done in fp32 to avoid bf16 precision loss.

    Args:
        out1: First partial output, shape (total_tokens, n_heads, head_dim).
        lse1: First partial log-sum-exp, shape (n_heads, total_tokens).
        out2: Second partial output, shape (total_tokens, n_heads, head_dim).
        lse2: Second partial log-sum-exp, shape (n_heads, total_tokens).

    Returns:
        Tuple of merged (output, lse) with same shapes.
    """
    # Upconvert to fp32 for intermediate precision
    lse1_t = lse1.transpose(0, 1).float()  # (total_tokens, n_heads)
    lse2_t = lse2.transpose(0, 1).float()

    # softplus formula: log(exp(a) + exp(b)) = max + softplus(min - max)
    min_lse = torch.minimum(lse1_t, lse2_t)
    max_lse = torch.maximum(lse1_t, lse2_t)
    merged_lse = max_lse + torch.nn.functional.softplus(
        _safe_subtract(min_lse, max_lse)
    )

    # Rescale weights: w_i = exp(lse_i - merged_lse)
    w1 = torch.exp(_safe_subtract(lse1_t, merged_lse)).unsqueeze(-1)
    w2 = torch.exp(_safe_subtract(lse2_t, merged_lse)).unsqueeze(-1)

    out = w1 * out1.float() + w2 * out2.float()

    return out.to(out1.dtype), merged_lse.transpose(0, 1).to(lse1.dtype)


# ---------------------------------------------------------------------------
# cu_seqlens helpers
# ---------------------------------------------------------------------------


def _build_ring_step_cu_seqlens(
    global_cu_seqlens: torch.Tensor,
    q_chunk_start: int,
    q_chunk_end: int,
    k_chunk_start: int,
    k_chunk_end: int,
    original_total_seqlen: int,
    chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]], list[tuple[int, int]], int, bool]:
    """Build cu_seqlens for a (Q_chunk, K_chunk) pair in ring attention.

    For each document overlapping both the Q chunk and K chunk:
    - k_chunk < q_chunk (below diagonal): Q attends to full K intersection.
      ``is_causal=False`` because all K positions precede all Q positions.
    - k_chunk == q_chunk (diagonal): Causal within the chunk.
      ``is_causal=True`` with matching Q/K lengths.
    - k_chunk > q_chunk (above diagonal): Skip (causal masking).

    Args:
        global_cu_seqlens: Global cumulative sequence lengths.
        q_chunk_start: Start of Q chunk (inclusive).
        q_chunk_end: End of Q chunk (exclusive).
        k_chunk_start: Start of K chunk (inclusive).
        k_chunk_end: End of K chunk (exclusive).
        original_total_seqlen: Original total sequence length (before padding).
        chunk_size: Chunk size (for computing chunk indices).
        device: Device for output tensors.

    Returns:
        Tuple of (cu_seqlens_q, cu_seqlens_k, q_doc_ranges, k_doc_ranges,
                  num_real_q_tokens, is_causal).
        q_doc_ranges and k_doc_ranges are lists of (global_start, global_end) pairs.
    """
    q_chunk_idx = q_chunk_start // chunk_size
    k_chunk_idx = k_chunk_start // chunk_size

    empty_cu = torch.tensor([0], dtype=torch.int32, device=device)

    if k_chunk_idx > q_chunk_idx:
        # Above diagonal: skip
        return empty_cu, empty_cu, [], [], 0, False

    is_causal = k_chunk_idx == q_chunk_idx

    global_cu = global_cu_seqlens.tolist()
    num_docs = len(global_cu) - 1

    cu_q_list = [0]
    cu_k_list = [0]
    q_doc_ranges: list[tuple[int, int]] = []
    k_doc_ranges: list[tuple[int, int]] = []
    num_real_q_tokens = 0

    for d in range(num_docs):
        doc_start = int(global_cu[d])
        doc_end = min(int(global_cu[d + 1]), original_total_seqlen)

        # Q portion: intersection of doc with Q chunk
        q_s = max(doc_start, q_chunk_start)
        q_e = min(doc_end, q_chunk_end)
        q_len = q_e - q_s
        if q_len <= 0:
            continue

        # K portion: intersection of doc with K chunk
        k_s = max(doc_start, k_chunk_start)
        k_e = min(doc_end, k_chunk_end)
        k_len = k_e - k_s
        if k_len <= 0:
            continue

        cu_q_list.append(cu_q_list[-1] + q_len)
        cu_k_list.append(cu_k_list[-1] + k_len)
        q_doc_ranges.append((q_s, q_e))
        k_doc_ranges.append((k_s, k_e))
        num_real_q_tokens += q_len

    cu_seqlens_q = torch.tensor(cu_q_list, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor(cu_k_list, dtype=torch.int32, device=device)

    return cu_seqlens_q, cu_seqlens_k, q_doc_ranges, k_doc_ranges, num_real_q_tokens, is_causal



# ---------------------------------------------------------------------------
# FFA (Flex-Flash-Attention) compute path
# ---------------------------------------------------------------------------


def _compute_step_ffa(
    q: torch.Tensor,
    cur_k: torch.Tensor,
    cur_v: torch.Tensor,
    q_doc_ranges: list[tuple[int, int]],
    k_doc_ranges: list[tuple[int, int]],
    is_causal: bool,
    q_chunk_start: int,
    k_chunk_start: int,
    chunk_size: int,
    n_heads: int,
    head_dim: int,
    accum_out: torch.Tensor,
    accum_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one (Q,K) step using magi-attention's FFA kernel and merge.

    Operates on full chunk-level tensors with range descriptors.  Each doc's
    Q/K intersection becomes one range pair for flex_attention.

    Args:
        q: Full Q chunk, shape (chunk_size, n_heads, head_dim).
        cur_k: Full K chunk, shape (chunk_size, n_kv_heads, head_dim).
        cur_v: Full V chunk, shape (chunk_size, n_kv_heads, head_dim).
        q_doc_ranges: Global (start, end) pairs for Q docs.
        k_doc_ranges: Global (start, end) pairs for K docs.
        is_causal: True for diagonal blocks, False for below-diagonal.
        q_chunk_start: Global start of the Q chunk.
        k_chunk_start: Global start of the K chunk.
        chunk_size: Tokens per chunk.
        n_heads: Number of Q attention heads.
        head_dim: Dimension per head.
        accum_out: Running accumulated output.
        accum_lse: Running accumulated LSE.

    Returns:
        Updated (accum_out, accum_lse).
    """
    device = q.device

    # Convert global doc ranges to chunk-local coordinates for FFA
    q_ranges = torch.tensor(
        [[qs - q_chunk_start, qe - q_chunk_start] for qs, qe in q_doc_ranges],
        dtype=torch.int32,
        device=device,
    )
    k_ranges = torch.tensor(
        [[ks - k_chunk_start, ke - k_chunk_start] for ks, ke in k_doc_ranges],
        dtype=torch.int32,
        device=device,
    )
    # MaskType mapping: 0=FULL, 1=CAUSAL — matches FFA's attn_type_map
    attn_type = 1 if is_causal else 0
    attn_type_map = torch.full(
        (len(q_doc_ranges),), attn_type, dtype=torch.int32, device=device
    )

    step_out, step_meta = flex_attn_forward(
        q,
        cur_k,
        cur_v,
        q_ranges,
        k_ranges,
        attn_type_map,
    )

    # flex_attn LSE shape: (chunk_size, n_heads) float32 -> (n_heads, chunk_size)
    step_lse = step_meta.lse.transpose(0, 1)

    # Positions outside q_ranges have undefined output/LSE from FFA.
    # Mask them to -inf LSE / zero output so they don't affect the merge.
    q_coverage = torch.zeros(chunk_size, dtype=torch.bool, device=device)
    for qs, qe in q_doc_ranges:
        q_coverage[qs - q_chunk_start : qe - q_chunk_start] = True
    uncovered = ~q_coverage
    if uncovered.any():
        step_lse[:, uncovered] = float("-inf")
        step_out[uncovered] = 0

    return merge_with_lse(accum_out, accum_lse, step_out, step_lse)


# ---------------------------------------------------------------------------
# Shared per-step attention helper
# ---------------------------------------------------------------------------


def _compute_and_merge_step(
    q: torch.Tensor,
    cur_k: torch.Tensor,
    cur_v: torch.Tensor,
    global_cu_seqlens: torch.Tensor,
    q_chunk_start: int,
    q_chunk_end: int,
    k_chunk_start: int,
    k_chunk_end: int,
    original_total_seqlen: int,
    chunk_size: int,
    accum_out: torch.Tensor,
    accum_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute partial attention for one (Q_chunk, K_chunk) pair and merge.

    Builds cu_seqlens for the (Q,K) pair and calls flex_attention via
    ``_compute_step_ffa``, merging into the running accumulator.

    Args:
        q: Local Q chunk, shape (chunk_size, n_heads, head_dim).
        cur_k: K for this step, shape (chunk_size, n_kv_heads, head_dim).
        cur_v: V for this step, shape (chunk_size, n_kv_heads, head_dim).
        global_cu_seqlens: Global cumulative sequence lengths.
        q_chunk_start: Global start of Q chunk.
        q_chunk_end: Global end of Q chunk.
        k_chunk_start: Global start of K chunk.
        k_chunk_end: Global end of K chunk.
        original_total_seqlen: Total sequence length before padding.
        chunk_size: Chunk size.
        accum_out: Running accumulated output.
        accum_lse: Running accumulated LSE.

    Returns:
        Updated (accum_out, accum_lse).
    """
    device = q.device
    dtype = q.dtype
    n_heads = q.shape[1]
    head_dim = q.shape[2]

    (
        cu_q,
        cu_k,
        q_doc_ranges,
        k_doc_ranges,
        num_real_q,
        is_causal,
    ) = _build_ring_step_cu_seqlens(
        global_cu_seqlens,
        q_chunk_start,
        q_chunk_end,
        k_chunk_start,
        k_chunk_end,
        original_total_seqlen,
        chunk_size,
        device,
    )

    if num_real_q == 0 or not k_doc_ranges:
        return accum_out, accum_lse

    return _compute_step_ffa(
        q,
        cur_k,
        cur_v,
        q_doc_ranges,
        k_doc_ranges,
        is_causal,
        q_chunk_start,
        k_chunk_start,
        chunk_size,
        n_heads,
        head_dim,
        accum_out,
        accum_lse,
    )


# ---------------------------------------------------------------------------
# Backward helper (flex_attention backward kernel)
# ---------------------------------------------------------------------------


def _backward_step_ffa(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    cur_k: torch.Tensor,
    cur_v: torch.Tensor,
    merged_out: torch.Tensor,
    merged_lse: torch.Tensor,
    global_cu_seqlens: torch.Tensor,
    q_chunk_start: int,
    q_chunk_end: int,
    k_chunk_start: int,
    k_chunk_end: int,
    original_total_seqlen: int,
    chunk_size: int,
    grad_q: torch.Tensor,
    grad_kv: torch.Tensor,
) -> None:
    """Compute backward for one (Q,K) step using flex_attention backward.

    Operates on full chunk-level tensors with range descriptors (no pack/unpack).

    Args:
        Same as ``_backward_step``.
    """
    device = q.device
    n_kv_head_dim = cur_k.shape[2]

    cu_q, cu_k, q_doc_ranges, k_doc_ranges, num_real_q, is_causal = (
        _build_ring_step_cu_seqlens(
            global_cu_seqlens,
            q_chunk_start, q_chunk_end,
            k_chunk_start, k_chunk_end,
            original_total_seqlen, chunk_size, device,
        )
    )

    if num_real_q == 0 or not k_doc_ranges:
        return

    # Convert doc ranges to FFA format (chunk-local coordinates)
    q_ranges = torch.tensor(
        [[qs - q_chunk_start, qe - q_chunk_start] for qs, qe in q_doc_ranges],
        dtype=torch.int32, device=device,
    )
    k_ranges = torch.tensor(
        [[ks - k_chunk_start, ke - k_chunk_start] for ks, ke in k_doc_ranges],
        dtype=torch.int32, device=device,
    )
    attn_type = 1 if is_causal else 0
    attn_type_map = torch.full(
        (len(q_doc_ranges),), attn_type, dtype=torch.int32, device=device,
    )

    # LSE shape: (n_heads, chunk_size) -> (chunk_size, n_heads) for backward
    ffa_lse = merged_lse.transpose(0, 1).contiguous()

    dq, dk, dv = flex_attn_backward(
        grad_output.contiguous(),
        q.contiguous(),
        cur_k.contiguous(),
        cur_v.contiguous(),
        merged_out.contiguous(),
        ffa_lse,
        q_ranges,
        k_ranges,
        attn_type_map,
    )

    # Accumulate into chunk-level gradient buffers
    grad_q += dq
    grad_kv[..., :n_kv_head_dim] += dk
    grad_kv[..., n_kv_head_dim:] += dv


def _backward_step(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    cur_k: torch.Tensor,
    cur_v: torch.Tensor,
    merged_out: torch.Tensor,
    merged_lse: torch.Tensor,
    global_cu_seqlens: torch.Tensor,
    q_chunk_start: int,
    q_chunk_end: int,
    k_chunk_start: int,
    k_chunk_end: int,
    original_total_seqlen: int,
    chunk_size: int,
    grad_q: torch.Tensor,
    grad_kv: torch.Tensor,
) -> None:
    """Compute backward for one (Q,K) step using flex_attention backward."""
    _backward_step_ffa(
        grad_output, q, cur_k, cur_v, merged_out, merged_lse,
        global_cu_seqlens,
        q_chunk_start, q_chunk_end, k_chunk_start, k_chunk_end,
        original_total_seqlen, chunk_size,
        grad_q, grad_kv,
    )



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def varlen_ring_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    global_cu_seqlens: torch.Tensor,
    plan: DispatchPlan,
    cp_mesh: DeviceMesh,
) -> torch.Tensor:
    """Varlen attention with context parallelism via Magi Attention.

    See: Men et al., "Magi Attention", https://arxiv.org/abs/2505.13211

    Args:
        q: Local Q chunk, shape (chunk_size, n_heads, head_dim).
        k: Local K chunk, shape (chunk_size, n_kv_heads, head_dim).
        v: Local V chunk, shape (chunk_size, n_kv_heads, head_dim).
        global_cu_seqlens: Global cumulative sequence lengths.
        plan: DispatchPlan (used for chunk_size and padding info).
        cp_mesh: DeviceMesh for CP group.

    Returns:
        Output tensor, shape (chunk_size, n_heads, head_dim).
    """
    from torchtitan.distributed.varlen_cp.magi_attention import (
        varlen_magi_dispatch,
    )

    return varlen_magi_dispatch(
        q, k, v, global_cu_seqlens, plan, cp_mesh
    )


