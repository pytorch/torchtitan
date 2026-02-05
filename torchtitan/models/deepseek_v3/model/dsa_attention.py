# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
DSA (DeepSeek Sparse Attention) Sparse Attention Wrappers.

These modules compute attention only on selected positions identified by
the light indexer, reducing complexity from O(L²) to O(L × topk).

Two implementations are provided:
1. DSASparseAttentionWrapper: For standard batched sequences
2. DSAVarlenSparseAttention: For packed/variable-length sequences
"""

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSAVarlenMetadata(NamedTuple):
    """Metadata for variable-length DSA attention."""

    cu_seqlens_q: torch.Tensor  # Cumulative sequence lengths for queries
    cu_seqlens_k: torch.Tensor  # Cumulative sequence lengths for keys
    max_seqlen_q: int
    max_seqlen_k: int
    topk_indices: torch.Tensor  # (total_tokens, topk) - selected positions
    topk_scores: torch.Tensor | None  # (total_tokens, topk) - optional selection scores


class DSASparseAttentionWrapper(nn.Module):
    """
    Sparse Attention wrapper that computes attention only for selected positions.

    This implements the sparse attention phase of DSA where we:
    1. Gather selected K/V based on top-k indices
    2. Compute attention only over selected positions
    3. Optionally weight by indexer scores

    This is more memory efficient than full attention for long sequences
    when topk << sequence_length.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_scores: torch.Tensor | None = None,
        scale: float | None = None,
    ) -> torch.Tensor:
        """
        Compute sparse attention using pre-selected indices.

        Args:
            q: Query tensor (bsz, n_heads, seqlen, head_dim).
            k: Key tensor (bsz, n_heads, seqlen, head_dim).
            v: Value tensor (bsz, n_heads, seqlen, v_head_dim).
            topk_indices: Indices of selected K/V positions (bsz, seqlen, topk).
            topk_scores: Optional pre-computed scores for weighting (bsz, seqlen, topk).
            scale: Attention scaling factor.

        Returns:
            output: Attention output (bsz, n_heads, seqlen, v_head_dim).
        """
        bsz, n_heads, seqlen, head_dim = q.shape
        _, _, _, v_head_dim = v.shape
        topk = topk_indices.shape[-1]

        if scale is None:
            scale = head_dim**-0.5

        # Expand indices for multi-head gathering
        # topk_indices: (bsz, seqlen, topk) -> need (bsz, n_heads, seqlen, topk, head_dim)
        indices_for_k = (
            topk_indices.unsqueeze(1).unsqueeze(-1).expand(bsz, n_heads, seqlen, topk, head_dim)
        )
        indices_for_v = (
            topk_indices.unsqueeze(1).unsqueeze(-1).expand(bsz, n_heads, seqlen, topk, v_head_dim)
        )

        # Gather selected keys and values for each query position
        # k shape: (bsz, n_heads, seqlen, head_dim)
        # We need to expand k to (bsz, n_heads, seqlen, seqlen, head_dim) then gather
        k_expanded = k.unsqueeze(2).expand(bsz, n_heads, seqlen, seqlen, head_dim)
        v_expanded = v.unsqueeze(2).expand(bsz, n_heads, seqlen, seqlen, v_head_dim)

        # Gather: (bsz, n_heads, seqlen, topk, head_dim)
        k_gathered = torch.gather(k_expanded, dim=3, index=indices_for_k)
        v_gathered = torch.gather(v_expanded, dim=3, index=indices_for_v)

        # Compute attention scores for selected positions
        # q: (bsz, n_heads, seqlen, head_dim)
        # k_gathered: (bsz, n_heads, seqlen, topk, head_dim)
        # scores: (bsz, n_heads, seqlen, topk)
        q_expanded = q.unsqueeze(-2)  # (bsz, n_heads, seqlen, 1, head_dim)
        attn_scores = torch.matmul(
            q_expanded, k_gathered.transpose(-2, -1)
        )  # (bsz, n_heads, seqlen, 1, topk)
        attn_scores = attn_scores.squeeze(-2) * scale  # (bsz, n_heads, seqlen, topk)

        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Optionally combine with indexer scores
        if topk_scores is not None:
            # topk_scores: (bsz, seqlen, topk) -> (bsz, 1, seqlen, topk)
            topk_scores_expanded = topk_scores.unsqueeze(1)
            attn_weights = attn_weights * topk_scores_expanded
            # Renormalize
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-12)

        # Compute output
        # attn_weights: (bsz, n_heads, seqlen, topk)
        # v_gathered: (bsz, n_heads, seqlen, topk, v_head_dim)
        attn_weights_expanded = attn_weights.unsqueeze(-1)  # (bsz, n_heads, seqlen, topk, 1)
        output = (attn_weights_expanded * v_gathered).sum(
            dim=-2
        )  # (bsz, n_heads, seqlen, v_head_dim)

        return output


class DSAVarlenSparseAttention(nn.Module):
    """
    Variable-length sparse attention for packed sequences.

    This supports document-packed batches where multiple documents are
    concatenated into a single sequence, with different valid context
    ranges for each position.

    The input tensors are "packed" format: (total_tokens, n_heads, head_dim)
    rather than (bsz, n_heads, seqlen, head_dim).
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        q_packed: torch.Tensor,
        k_packed: torch.Tensor,
        v_packed: torch.Tensor,
        metadata: DSAVarlenMetadata,
        scale: float | None = None,
    ) -> torch.Tensor:
        """
        Compute sparse attention for variable-length packed sequences.

        Args:
            q_packed: Query tensor (total_tokens, n_heads, head_dim).
            k_packed: Key tensor (total_tokens, n_heads, head_dim).
            v_packed: Value tensor (total_tokens, n_heads, v_head_dim).
            metadata: DSAVarlenMetadata containing cu_seqlens and topk_indices.
            scale: Attention scaling factor.

        Returns:
            output: Attention output (total_tokens, n_heads, v_head_dim).
        """
        total_tokens, n_heads, head_dim = q_packed.shape
        _, _, v_head_dim = v_packed.shape
        topk = metadata.topk_indices.shape[-1]

        if scale is None:
            scale = head_dim**-0.5

        # topk_indices: (total_tokens, topk) - absolute positions in packed sequence
        topk_indices = metadata.topk_indices

        # Gather K/V for each token's selected positions
        # Expand indices for all heads: (total_tokens, topk) -> (total_tokens, n_heads, topk, head_dim)
        indices_for_k = (
            topk_indices.unsqueeze(1).unsqueeze(-1).expand(total_tokens, n_heads, topk, head_dim)
        )
        indices_for_v = (
            topk_indices.unsqueeze(1).unsqueeze(-1).expand(total_tokens, n_heads, topk, v_head_dim)
        )

        # Expand k_packed and v_packed to enable gathering
        # k_packed: (total_tokens, n_heads, head_dim)
        # We need to gather from all positions for each token
        # Create expanded views for gathering
        k_for_gather = k_packed.unsqueeze(0).expand(
            total_tokens, total_tokens, n_heads, head_dim
        )  # (total_tokens, total_tokens, n_heads, head_dim)
        v_for_gather = v_packed.unsqueeze(0).expand(
            total_tokens, total_tokens, n_heads, v_head_dim
        )

        # Transpose for proper indexing: (total_tokens, n_heads, total_tokens, head_dim)
        k_for_gather = k_for_gather.permute(0, 2, 1, 3)
        v_for_gather = v_for_gather.permute(0, 2, 1, 3)

        # Gather: (total_tokens, n_heads, topk, head_dim)
        k_gathered = torch.gather(k_for_gather, dim=2, index=indices_for_k)
        v_gathered = torch.gather(v_for_gather, dim=2, index=indices_for_v)

        # Compute attention scores
        # q_packed: (total_tokens, n_heads, head_dim) -> (total_tokens, n_heads, 1, head_dim)
        q_expanded = q_packed.unsqueeze(2)
        attn_scores = torch.matmul(
            q_expanded, k_gathered.transpose(-2, -1)
        )  # (total_tokens, n_heads, 1, topk)
        attn_scores = attn_scores.squeeze(2) * scale  # (total_tokens, n_heads, topk)

        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Optionally combine with indexer scores
        if metadata.topk_scores is not None:
            # topk_scores: (total_tokens, topk) -> (total_tokens, 1, topk)
            topk_scores_expanded = metadata.topk_scores.unsqueeze(1)
            attn_weights = attn_weights * topk_scores_expanded
            # Renormalize
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-12)

        # Compute output
        # attn_weights: (total_tokens, n_heads, topk)
        # v_gathered: (total_tokens, n_heads, topk, v_head_dim)
        attn_weights_expanded = attn_weights.unsqueeze(-1)  # (total_tokens, n_heads, topk, 1)
        output = (attn_weights_expanded * v_gathered).sum(
            dim=2
        )  # (total_tokens, n_heads, v_head_dim)

        return output


class DSAVarlenSparseAttentionOptimized(nn.Module):
    """
    Optimized variable-length sparse attention using per-document processing.

    This version iterates over documents to avoid the O(total_tokens²) memory
    overhead of the naive implementation. More memory efficient for long
    packed sequences.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        q_packed: torch.Tensor,
        k_packed: torch.Tensor,
        v_packed: torch.Tensor,
        metadata: DSAVarlenMetadata,
        scale: float | None = None,
    ) -> torch.Tensor:
        """
        Compute sparse attention with per-document optimization.

        Args:
            q_packed: Query tensor (total_tokens, n_heads, head_dim).
            k_packed: Key tensor (total_tokens, n_heads, head_dim).
            v_packed: Value tensor (total_tokens, n_heads, v_head_dim).
            metadata: DSAVarlenMetadata containing cu_seqlens and topk_indices.
            scale: Attention scaling factor.

        Returns:
            output: Attention output (total_tokens, n_heads, v_head_dim).
        """
        total_tokens, n_heads, head_dim = q_packed.shape
        _, _, v_head_dim = v_packed.shape
        topk = metadata.topk_indices.shape[-1]

        if scale is None:
            scale = head_dim**-0.5

        cu_seqlens = metadata.cu_seqlens_q
        num_seqs = len(cu_seqlens) - 1

        # Output buffer
        output = torch.zeros(
            total_tokens, n_heads, v_head_dim, dtype=q_packed.dtype, device=q_packed.device
        )

        # Process each document separately
        for i in range(num_seqs):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            seq_len = end - start

            # Extract this document's Q/K/V
            q_doc = q_packed[start:end]  # (seq_len, n_heads, head_dim)
            k_doc = k_packed[start:end]  # (seq_len, n_heads, head_dim)
            v_doc = v_packed[start:end]  # (seq_len, n_heads, v_head_dim)

            # Extract this document's indices (adjusted to local positions)
            doc_indices = metadata.topk_indices[start:end]  # (seq_len, topk)
            # Adjust indices to be relative to document start
            doc_indices_local = doc_indices - start
            # Clamp to valid range within document
            doc_indices_local = doc_indices_local.clamp(0, seq_len - 1)

            # Get effective topk for this document
            effective_topk = min(topk, seq_len)
            doc_indices_local = doc_indices_local[:, :effective_topk]

            # Expand indices for gathering
            indices_for_k = (
                doc_indices_local.unsqueeze(1)
                .unsqueeze(-1)
                .expand(seq_len, n_heads, effective_topk, head_dim)
            )
            indices_for_v = (
                doc_indices_local.unsqueeze(1)
                .unsqueeze(-1)
                .expand(seq_len, n_heads, effective_topk, v_head_dim)
            )

            # Expand K/V for gathering
            k_for_gather = k_doc.unsqueeze(0).expand(seq_len, seq_len, n_heads, head_dim)
            v_for_gather = v_doc.unsqueeze(0).expand(seq_len, seq_len, n_heads, v_head_dim)

            # Transpose for proper indexing
            k_for_gather = k_for_gather.permute(0, 2, 1, 3)
            v_for_gather = v_for_gather.permute(0, 2, 1, 3)

            # Gather
            k_gathered = torch.gather(k_for_gather, dim=2, index=indices_for_k)
            v_gathered = torch.gather(v_for_gather, dim=2, index=indices_for_v)

            # Compute attention
            q_expanded = q_doc.unsqueeze(2)
            attn_scores = torch.matmul(q_expanded, k_gathered.transpose(-2, -1)).squeeze(2) * scale

            # Softmax
            attn_weights = F.softmax(attn_scores, dim=-1)

            # Optional indexer score weighting
            if metadata.topk_scores is not None:
                doc_scores = metadata.topk_scores[start:end, :effective_topk]
                attn_weights = attn_weights * doc_scores.unsqueeze(1)
                attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-12)

            # Compute output
            attn_weights_expanded = attn_weights.unsqueeze(-1)
            doc_output = (attn_weights_expanded * v_gathered).sum(dim=2)

            output[start:end] = doc_output

        return output
