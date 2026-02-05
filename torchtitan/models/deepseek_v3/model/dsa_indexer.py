# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
DSA (DeepSeek Sparse Attention) Light Indexer Module.

The light indexer computes top-k indices for sparse attention by:
1. Projecting Q/K to a compressed indexer space
2. Applying Hadamard transform for better locality
3. Using FP8 quantization for efficient scoring
4. Selecting top-k positions per query
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DSAConfig:
    """Configuration for DeepSeek Sparse Attention."""

    enabled: bool = False
    indexer_dim: int = 128
    topk: int = 2048
    use_fp8: bool = True
    hadamard_transform: bool = True
    temperature: float = 1.0
    start_layer: int = 0  # First layer to apply DSA (0 = all MLA layers)
    combine_with_indexer_scores: bool = True


def _generate_hadamard_matrix(dim: int) -> torch.Tensor:
    """
    Generate a normalized Hadamard matrix of size dim x dim.

    The Hadamard matrix is an orthogonal matrix with entries ±1/sqrt(dim).
    It provides an efficient orthogonal transformation for scoring.

    Args:
        dim: Dimension of the matrix (must be power of 2).

    Returns:
        Hadamard matrix of shape (dim, dim).
    """
    if dim == 1:
        return torch.ones(1, 1)

    # Recursively build Hadamard matrix using Sylvester construction
    h_prev = _generate_hadamard_matrix(dim // 2)
    h = torch.cat(
        [torch.cat([h_prev, h_prev], dim=1), torch.cat([h_prev, -h_prev], dim=1)], dim=0
    )
    return h


class DSALightIndexer(nn.Module):
    """
    Light Indexer for DeepSeek Sparse Attention (DSA).

    Computes compressed similarity scores between queries and keys
    using optional FP8 quantization and Hadamard transform for efficiency.

    The indexer operates on the compressed representations from MLA:
    - q_compressed: Output of wq_a (before wq_b expansion)
    - kv_compressed: Output of wkv_a before kv_norm (the kv_lora_rank part)

    This allows efficient scoring without expanding to full attention dimensions.
    """

    def __init__(
        self,
        q_dim: int,
        kv_lora_rank: int,
        indexer_dim: int,
        n_heads: int,
        topk: int,
        use_fp8: bool = True,
        hadamard_transform: bool = True,
        temperature: float = 1.0,
        norm_eps: float = 1e-5,
    ) -> None:
        """
        Initialize the DSA Light Indexer.

        Args:
            q_dim: Query dimension (typically q_lora_rank or dim).
            kv_lora_rank: KV compressed dimension from MLA (e.g., 512).
            indexer_dim: Compressed dimension for indexer (e.g., 128).
            n_heads: Number of attention heads.
            topk: Number of positions to select per query.
            use_fp8: Whether to use FP8 quantization for scoring.
            hadamard_transform: Whether to apply Hadamard transform.
            temperature: Temperature for scoring (lower = sharper selection).
            norm_eps: Epsilon for layer normalization.
        """
        super().__init__()
        self.indexer_dim = indexer_dim
        self.topk = topk
        self.use_fp8 = use_fp8
        self.hadamard_transform = hadamard_transform
        self.temperature = temperature
        self.n_heads = n_heads

        # Query projection: compress Q to indexer space
        self.wq_indexer = nn.Linear(q_dim, indexer_dim, bias=False)

        # Key projection: compress KV to indexer space
        self.wk_indexer = nn.Linear(kv_lora_rank, indexer_dim, bias=False)

        # Normalization for indexer keys
        self.k_norm = nn.RMSNorm(indexer_dim, eps=norm_eps)

        # Pre-compute Hadamard matrix if needed
        if self.hadamard_transform:
            # Ensure dimension is power of 2
            if indexer_dim & (indexer_dim - 1) != 0:
                raise ValueError(
                    f"Hadamard transform requires power-of-2 indexer_dim, got {indexer_dim}"
                )
            h_matrix = _generate_hadamard_matrix(indexer_dim)
            # Normalize for orthogonality
            h_matrix = h_matrix / (indexer_dim**0.5)
            self.register_buffer("hadamard_matrix", h_matrix, persistent=False)

    def _apply_hadamard(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Hadamard transform to last dimension."""
        return torch.matmul(x, self.hadamard_matrix)

    def _quantize_to_fp8(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to FP8 for efficient scoring.

        Uses per-row dynamic scaling for better numerical accuracy.

        Args:
            x: Input tensor to quantize.

        Returns:
            Tuple of (quantized tensor in FP8, scale tensor).
        """
        # Compute per-row scale based on absmax
        absmax = x.abs().amax(dim=-1, keepdim=True)
        # FP8 e4m3 has max value of 448
        scale = absmax / 448.0
        scale = torch.clamp(scale, min=1e-12)

        # Scale and quantize
        x_scaled = x / scale
        x_fp8 = x_scaled.to(torch.float8_e4m3fn)

        return x_fp8, scale

    def forward(
        self,
        q_compressed: torch.Tensor,
        kv_compressed: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute top-k indices for sparse attention.

        Args:
            q_compressed: Compressed query representation (bsz, q_len, q_dim).
            kv_compressed: Compressed key-value representation (bsz, kv_len, kv_lora_rank).
            attention_mask: Optional boolean mask (bsz, q_len, kv_len) where True = attend.
            cu_seqlens: Optional cumulative sequence lengths for varlen batching.

        Returns:
            topk_indices: Indices of top-k positions (bsz, q_len, topk).
            topk_scores: Softmax scores for selected positions (bsz, q_len, topk).
        """
        bsz, q_len, _ = q_compressed.shape
        _, kv_len, _ = kv_compressed.shape

        # Project to indexer space
        q_idx = self.wq_indexer(q_compressed)  # (bsz, q_len, indexer_dim)
        k_idx = self.wk_indexer(kv_compressed)  # (bsz, kv_len, indexer_dim)

        # Normalize keys for stable scoring
        k_idx = self.k_norm(k_idx)

        # Apply Hadamard transform for better locality
        if self.hadamard_transform:
            q_idx = self._apply_hadamard(q_idx)
            k_idx = self._apply_hadamard(k_idx)

        # Compute similarity scores
        if self.use_fp8:
            # FP8 quantization path for efficiency
            q_fp8, q_scale = self._quantize_to_fp8(q_idx)
            k_fp8, k_scale = self._quantize_to_fp8(k_idx)

            # Compute scores with scale compensation
            # Convert back to compute dtype for bmm
            scores = torch.bmm(
                q_fp8.to(q_compressed.dtype), k_fp8.to(q_compressed.dtype).transpose(-2, -1)
            )
            # Apply scales
            scores = scores * (q_scale * k_scale.transpose(-2, -1))
        else:
            # Standard dot product scoring
            scores = torch.bmm(q_idx, k_idx.transpose(-2, -1))

        # Temperature scaling
        scores = scores / self.temperature  # (bsz, q_len, kv_len)

        # Apply causal masking
        if attention_mask is not None:
            # attention_mask: True = attend, False = mask out
            scores = scores.masked_fill(~attention_mask, float("-inf"))
        else:
            # Default: apply causal mask
            causal_mask = torch.triu(
                torch.ones(q_len, kv_len, dtype=torch.bool, device=scores.device),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        # Handle varlen batching with cumulative sequence lengths
        if cu_seqlens is not None:
            scores = self._apply_varlen_mask(scores, cu_seqlens)

        # Select top-k positions
        effective_topk = min(self.topk, kv_len)
        topk_scores_raw, topk_indices = torch.topk(
            scores, k=effective_topk, dim=-1, sorted=False
        )

        # Apply softmax to selected scores for weighting
        topk_scores = F.softmax(topk_scores_raw, dim=-1)

        return topk_indices, topk_scores

    def _apply_varlen_mask(
        self, scores: torch.Tensor, cu_seqlens: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply document boundary masking for variable-length sequences.

        Prevents attention across document boundaries in packed batches.

        Args:
            scores: Attention scores (bsz, q_len, kv_len).
            cu_seqlens: Cumulative sequence lengths defining document boundaries.

        Returns:
            Masked scores with -inf for cross-document positions.
        """
        bsz, q_len, kv_len = scores.shape
        device = scores.device

        # Create position indices
        q_pos = torch.arange(q_len, device=device).unsqueeze(0).expand(bsz, -1)
        kv_pos = torch.arange(kv_len, device=device).unsqueeze(0).expand(bsz, -1)

        # Determine document ID for each position
        # cu_seqlens: [0, len1, len1+len2, ...] -> segment each position
        cu_seqlens = cu_seqlens.to(device)
        q_doc_ids = torch.bucketize(q_pos, cu_seqlens[1:-1], right=False)
        kv_doc_ids = torch.bucketize(kv_pos, cu_seqlens[1:-1], right=False)

        # Mask out cross-document attention
        doc_mask = q_doc_ids.unsqueeze(-1) == kv_doc_ids.unsqueeze(-2)
        scores = scores.masked_fill(~doc_mask, float("-inf"))

        return scores

    def init_weights(self, init_std: float = 0.02) -> None:
        """Initialize weights with truncated normal distribution."""
        nn.init.trunc_normal_(self.wq_indexer.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.wk_indexer.weight, mean=0.0, std=init_std)
        self.k_norm.reset_parameters()
