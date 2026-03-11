# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Layer 1: Mask Slice Primitives

Atomic building blocks for decomposing variable-length attention masks into
chunk-level sub-slices. Each AttnSlice describes a rectangular region of the
attention matrix with a specific mask type (FULL, CAUSAL, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class MaskType(IntEnum):
    """Type of attention mask for a sub-slice of the attention matrix."""

    FULL = 0  # All Q attend to all K (dense block)
    CAUSAL = 1  # Lower-triangle, bottom-right aligned
    INVCAUSAL = 2  # Upper-triangle, top-left aligned
    BICAUSAL = 3  # Intersection of causal + inv-causal


@dataclass(frozen=True)
class AttnSlice:
    """A rectangular region of the global attention matrix with a mask type.

    Coordinates are in global token space (not chunk-local).
    """

    q_start: int
    q_end: int
    k_start: int
    k_end: int
    mask_type: MaskType

    @property
    def q_len(self) -> int:
        return self.q_end - self.q_start

    @property
    def k_len(self) -> int:
        return self.k_end - self.k_start

    @property
    def work_estimate(self) -> float:
        """Exact attended-element count based on mask type.

        Uses precise trapezoid/triangle area instead of ``q*k/2`` approximation.
        Reference: magi-attention ``make_global_bucket_from_qk_ranges``.
        """
        q, k = self.q_len, self.k_len
        if self.mask_type == MaskType.FULL:
            return max(float(q * k), 1.0)
        elif self.mask_type in (MaskType.CAUSAL, MaskType.INVCAUSAL):
            # Bottom-right (CAUSAL) or top-left (INVCAUSAL) triangle/trapezoid.
            # Exact area: rows of length k, k-1, ..., k-min(q,k)+1
            m = min(q, k)
            area = (2 * k - m + 1) * m / 2.0
            return max(area, 1.0)
        else:  # BICAUSAL
            m = min(q, k)
            area = (2 * k - m + 1) * m / 4.0
            return max(area, 1.0)


def cu_seqlens_to_attn_slices(
    cu_seqlens: torch.Tensor | list[int],
    is_causal: bool = True,
) -> list[AttnSlice]:
    """Convert cumulative sequence lengths to a list of AttnSlices.

    Each document in the packed sequence produces one diagonal CAUSAL (or FULL)
    block on the attention matrix diagonal.

    Args:
        cu_seqlens: Cumulative sequence lengths, e.g. [0, 128, 256, 512].
            Length is num_docs + 1.
        is_causal: If True, each document uses CAUSAL masking. If False, FULL.

    Returns:
        List of AttnSlice, one per document.
    """
    if isinstance(cu_seqlens, torch.Tensor):
        cu_seqlens = cu_seqlens.tolist()

    mask_type = MaskType.CAUSAL if is_causal else MaskType.FULL
    slices = []
    for i in range(len(cu_seqlens) - 1):
        start = int(cu_seqlens[i])
        end = int(cu_seqlens[i + 1])
        if end > start:
            slices.append(
                AttnSlice(
                    q_start=start,
                    q_end=end,
                    k_start=start,
                    k_end=end,
                    mask_type=mask_type,
                )
            )
    return slices


def split_slice_at_chunk_boundary(
    attn_slice: AttnSlice,
    chunk_size: int,
    total_seqlen: int,
) -> list[AttnSlice]:
    """Split a global AttnSlice into sub-slices at chunk boundaries.

    When a document spans multiple chunks, the diagonal CAUSAL block is
    decomposed:
    - Diagonal sub-blocks (q_chunk == k_chunk for same doc) stay CAUSAL
    - Below-diagonal sub-blocks (q_chunk > k_chunk) become FULL
      (Q in later chunk attends to all K in earlier chunk of same doc)
    - Above-diagonal sub-blocks are dropped (causal masking)

    For FULL mask type, all sub-blocks are FULL.

    Args:
        attn_slice: The global slice to split.
        chunk_size: Size of each chunk.
        total_seqlen: Total sequence length (for computing chunk count).

    Returns:
        List of sub-AttnSlices, one per (q_chunk, k_chunk) pair that
        has non-zero overlap with the original slice.
    """
    num_chunks = (total_seqlen + chunk_size - 1) // chunk_size
    result = []

    for q_chunk_idx in range(num_chunks):
        q_chunk_start = q_chunk_idx * chunk_size
        q_chunk_end = min((q_chunk_idx + 1) * chunk_size, total_seqlen)

        # Intersect Q range with the slice's Q range
        q_start = max(attn_slice.q_start, q_chunk_start)
        q_end = min(attn_slice.q_end, q_chunk_end)
        if q_start >= q_end:
            continue

        for k_chunk_idx in range(num_chunks):
            k_chunk_start = k_chunk_idx * chunk_size
            k_chunk_end = min((k_chunk_idx + 1) * chunk_size, total_seqlen)

            # Intersect K range with the slice's K range
            k_start = max(attn_slice.k_start, k_chunk_start)
            k_end = min(attn_slice.k_end, k_chunk_end)
            if k_start >= k_end:
                continue

            # Determine mask type for this sub-block
            if attn_slice.mask_type == MaskType.FULL:
                sub_mask = MaskType.FULL
            elif attn_slice.mask_type == MaskType.CAUSAL:
                if q_chunk_idx == k_chunk_idx:
                    # Diagonal block: the causal mask still applies
                    sub_mask = MaskType.CAUSAL
                elif q_chunk_idx > k_chunk_idx:
                    # Below diagonal: Q is after K, full attention
                    sub_mask = MaskType.FULL
                else:
                    # Above diagonal: Q is before K, no attention (causal)
                    continue
            elif attn_slice.mask_type == MaskType.INVCAUSAL:
                if q_chunk_idx == k_chunk_idx:
                    sub_mask = MaskType.INVCAUSAL
                elif q_chunk_idx < k_chunk_idx:
                    sub_mask = MaskType.FULL
                else:
                    continue
            else:  # BICAUSAL
                if q_chunk_idx == k_chunk_idx:
                    sub_mask = MaskType.BICAUSAL
                else:
                    continue

            result.append(
                AttnSlice(
                    q_start=q_start,
                    q_end=q_end,
                    k_start=k_start,
                    k_end=k_end,
                    mask_type=sub_mask,
                )
            )

    return result


def make_slice_mask(
    q_len: int,
    k_len: int,
    mask_type: MaskType,
    dtype: torch.dtype = torch.bool,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Materialize a dense boolean mask for a single slice.

    The mask is bottom-right aligned for CAUSAL, top-left for INVCAUSAL.

    Args:
        q_len: Number of query positions.
        k_len: Number of key positions.
        mask_type: The mask type.
        dtype: Output dtype (default: bool).
        device: Output device.

    Returns:
        Tensor of shape (q_len, k_len).
    """
    if mask_type == MaskType.FULL:
        return torch.ones(q_len, k_len, dtype=dtype, device=device)

    # For causal: bottom-right aligned triangle
    # mask[i, j] = True iff (i - (q_len - 1)) <= (j - (k_len - 1))
    # Equivalently: j <= i - q_len + k_len
    # Or: i + k_len - q_len >= j
    q_idx = torch.arange(q_len, device=device).unsqueeze(1)
    k_idx = torch.arange(k_len, device=device).unsqueeze(0)

    if mask_type == MaskType.CAUSAL:
        # Bottom-right aligned: offset = k_len - q_len
        mask = (q_idx + k_len - q_len) >= k_idx
    elif mask_type == MaskType.INVCAUSAL:
        # Top-left aligned: upper triangle
        mask = q_idx <= k_idx
    elif mask_type == MaskType.BICAUSAL:
        causal = (q_idx + k_len - q_len) >= k_idx
        invcausal = q_idx <= k_idx
        mask = causal & invcausal
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")

    return mask.to(dtype=dtype)
