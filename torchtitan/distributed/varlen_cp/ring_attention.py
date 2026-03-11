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
from torch.nn.attention.varlen import AuxRequest, varlen_attn

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
# Token extraction / scattering helpers
# ---------------------------------------------------------------------------


def _extract_tokens_for_docs(
    chunk_tensor: torch.Tensor,
    doc_ranges: list[tuple[int, int]],
    chunk_start: int,
) -> torch.Tensor:
    """Extract tokens from a chunk tensor given global doc ranges.

    Converts global (start, end) ranges to chunk-local offsets and
    concatenates the extracted slices.

    Args:
        chunk_tensor: Tensor of shape (chunk_size, ...).
        doc_ranges: List of (global_start, global_end) pairs.
        chunk_start: Global start position of the chunk.

    Returns:
        Packed tensor with tokens from all doc ranges.
    """
    if not doc_ranges:
        return chunk_tensor[:0]

    slices = []
    for gs, ge in doc_ranges:
        local_s = gs - chunk_start
        local_e = ge - chunk_start
        slices.append(chunk_tensor[local_s:local_e])

    return torch.cat(slices, dim=0)



def _scatter_to_chunk(
    packed_out: torch.Tensor,
    packed_lse: torch.Tensor,
    q_doc_ranges: list[tuple[int, int]],
    q_chunk_start: int,
    chunk_size: int,
    n_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter packed varlen_attn output back to full chunk positions.

    Positions not covered by any document get zero output and ``-inf`` LSE
    so they don't affect the LSE merge.

    Args:
        packed_out: Packed attention output, shape (total_packed_q, n_heads, head_dim).
        packed_lse: Packed LSE, shape (n_heads, total_packed_q).
        q_doc_ranges: List of (global_start, global_end) for Q documents.
        q_chunk_start: Global start of the Q chunk.
        chunk_size: Chunk size.
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
        device: Device.
        dtype: Data type for output tensor.

    Returns:
        Tuple of (full_out, full_lse) with shapes
        (chunk_size, n_heads, head_dim) and (n_heads, chunk_size).
    """
    full_out = torch.zeros(chunk_size, n_heads, head_dim, device=device, dtype=dtype)
    full_lse = torch.full(
        (n_heads, chunk_size), float("-inf"), device=device, dtype=torch.float32
    )

    offset = 0
    for q_s, q_e in q_doc_ranges:
        q_len = q_e - q_s
        local_start = q_s - q_chunk_start
        full_out[local_start : local_start + q_len] = packed_out[offset : offset + q_len]
        full_lse[:, local_start : local_start + q_len] = packed_lse[
            :, offset : offset + q_len
        ]
        offset += q_len

    return full_out, full_lse


def _extract_lse_for_docs(
    lse: torch.Tensor,
    doc_ranges: list[tuple[int, int]],
    chunk_start: int,
) -> torch.Tensor:
    """Extract LSE values for Q document positions.

    Mirrors ``_extract_tokens_for_docs`` but for LSE shape (n_heads, chunk_size).

    Args:
        lse: LSE tensor, shape (n_heads, chunk_size).
        doc_ranges: Global (start, end) pairs for documents.
        chunk_start: Global start of the chunk.

    Returns:
        Packed LSE, shape (n_heads, total_packed_tokens).
    """
    if not doc_ranges:
        return lse[:, :0]
    parts = []
    for gs, ge in doc_ranges:
        parts.append(lse[:, gs - chunk_start : ge - chunk_start])
    return torch.cat(parts, dim=1)


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

    FFA operates on full chunk-level tensors with range descriptors, avoiding
    the pack/unpack overhead of varlen_attn.  Each doc's Q/K intersection
    becomes one FFA range pair.

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

    Builds cu_seqlens, extracts tokens, calls varlen_attn, scatters output
    back to chunk positions, and merges with the running accumulator.
    Used by both ring-pass and group-cast paths.

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

    # --- FFA path: single kernel call on chunk-level tensors (no pack/unpack) ---
    try:
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
    except RuntimeError:
        pass  # Fall through to varlen_attn path

    # --- varlen_attn fallback: extract packed tokens, compute, scatter back ---

    # Extract packed Q and K/V tokens for participating documents
    q_packed = _extract_tokens_for_docs(q, q_doc_ranges, q_chunk_start)
    k_packed = _extract_tokens_for_docs(cur_k, k_doc_ranges, k_chunk_start)
    v_packed = _extract_tokens_for_docs(cur_v, k_doc_ranges, k_chunk_start)

    # Compute max sequence lengths for this step
    q_diffs = torch.diff(cu_q)
    k_diffs = torch.diff(cu_k)
    max_q = int(q_diffs.max().item()) if q_diffs.numel() > 0 else 0
    max_k = int(k_diffs.max().item()) if k_diffs.numel() > 0 else 0

    # Compute partial attention with LSE
    step_out, step_lse = varlen_attn(
        q_packed,
        k_packed,
        v_packed,
        cu_q,
        cu_k,
        max_q,
        max_k,
        is_causal=is_causal,
        return_aux=AuxRequest(lse=True),
    )

    # Scatter packed output to full chunk positions
    full_step_out, full_step_lse = _scatter_to_chunk(
        step_out,
        step_lse,
        q_doc_ranges,
        q_chunk_start,
        chunk_size,
        n_heads,
        head_dim,
        device,
        dtype,
    )

    # Merge with accumulated result
    return merge_with_lse(accum_out, accum_lse, full_step_out, full_step_lse)


# ---------------------------------------------------------------------------
# Custom backward helpers (flash attention backward kernel)
# ---------------------------------------------------------------------------


def _scatter_grad_to_chunk(
    packed_grad: torch.Tensor,
    chunk_grad: torch.Tensor,
    doc_ranges: list[tuple[int, int]],
    chunk_start: int,
) -> None:
    """Scatter packed gradients back to chunk positions (in-place accumulate).

    Args:
        packed_grad: Packed gradient, shape (num_packed, ...).
        chunk_grad: Chunk gradient accumulator, shape (chunk_size, ...).
        doc_ranges: Global (start, end) pairs for documents.
        chunk_start: Global start of the chunk.
    """
    offset = 0
    for gs, ge in doc_ranges:
        length = ge - gs
        local_s = gs - chunk_start
        chunk_grad[local_s : local_s + length] += packed_grad[offset : offset + length]
        offset += length


def _backward_step_flash(
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
    """Compute backward for one (Q_chunk, K_chunk) step using flash backward.

    Calls ``_flash_attention_backward`` with the merged output and LSE from
    the full forward pass. This gives correct K/V gradients because the
    softmax denominator (encoded in the merged LSE) accounts for all K/V
    chunks, not just the current step.

    Accumulates dQ into ``grad_q`` and dK/dV into ``grad_kv`` in-place.

    Args:
        grad_output: Upstream gradient, shape (chunk_size, n_heads, head_dim).
        q: Local Q chunk.
        cur_k: K for this step.
        cur_v: V for this step.
        merged_out: Merged attention output from forward.
        merged_lse: Merged LSE from forward, shape (n_heads, chunk_size), float32.
        global_cu_seqlens: Global cumulative sequence lengths.
        q_chunk_start, q_chunk_end: Q chunk range (global).
        k_chunk_start, k_chunk_end: K chunk range (global).
        original_total_seqlen: Total seqlen before padding.
        chunk_size: Chunk size.
        grad_q: Gradient accumulator for Q, shape (chunk_size, n_heads, head_dim).
        grad_kv: Gradient accumulator for K/V, shape (chunk_size, n_kv_heads, 2*head_dim).
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

    # Pack tensors the same way as forward
    q_packed = _extract_tokens_for_docs(q, q_doc_ranges, q_chunk_start)
    k_packed = _extract_tokens_for_docs(cur_k, k_doc_ranges, k_chunk_start)
    v_packed = _extract_tokens_for_docs(cur_v, k_doc_ranges, k_chunk_start)
    out_packed = _extract_tokens_for_docs(merged_out, q_doc_ranges, q_chunk_start)
    grad_packed = _extract_tokens_for_docs(grad_output, q_doc_ranges, q_chunk_start)
    lse_packed = _extract_lse_for_docs(merged_lse, q_doc_ranges, q_chunk_start)

    q_diffs = torch.diff(cu_q)
    k_diffs = torch.diff(cu_k)
    max_q = int(q_diffs.max().item()) if q_diffs.numel() > 0 else 0
    max_k = int(k_diffs.max().item()) if k_diffs.numel() > 0 else 0

    dq_packed, dk_packed, dv_packed = torch.ops.aten._flash_attention_backward(
        grad_packed.contiguous(),
        q_packed.contiguous(),
        k_packed.contiguous(),
        v_packed.contiguous(),
        out_packed.contiguous(),
        lse_packed.contiguous(),
        cu_q, cu_k, max_q, max_k,
        0.0,  # dropout_p
        is_causal,
        torch.empty(2, dtype=torch.uint64, device=device),  # dummy rng_state
        torch.empty(0, device=device),  # unused
    )

    # Scatter dQ back to grad_q
    _scatter_grad_to_chunk(dq_packed.float(), grad_q, q_doc_ranges, q_chunk_start)

    # Scatter dK/dV back to grad_kv
    offset = 0
    for k_s, k_e in k_doc_ranges:
        k_len = k_e - k_s
        local_s = k_s - k_chunk_start
        grad_kv[local_s : local_s + k_len, :, :n_kv_head_dim] += (
            dk_packed[offset : offset + k_len].float()
        )
        grad_kv[local_s : local_s + k_len, :, n_kv_head_dim:] += (
            dv_packed[offset : offset + k_len].float()
        )
        offset += k_len


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
    """Compute backward for one (Q,K) step using FFA CUTLASS backward kernel.

    Drop-in replacement for ``_backward_step_flash`` when FFA is available.
    Operates on full chunk-level tensors with range descriptors (no pack/unpack).

    Args:
        Same as ``_backward_step_flash``.
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
    """Dispatch to flex_attention backward, fall back to varlen_attn if needed."""
    try:
        return _backward_step_ffa(
            grad_output, q, cur_k, cur_v, merged_out, merged_lse,
            global_cu_seqlens,
            q_chunk_start, q_chunk_end, k_chunk_start, k_chunk_end,
            original_total_seqlen, chunk_size,
            grad_q, grad_kv,
        )
    except RuntimeError:
        pass  # Fall through to varlen_attn path
    _backward_step_flash(
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


