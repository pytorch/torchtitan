# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
DSA (DeepSeek Sparse Attention) with Context Parallelism Support.

This module provides ring-style communication for DSA when using Context Parallelism.
The key idea is:
1. Q is sharded across CP ranks (each rank has a local chunk)
2. K/V (compressed) are communicated via ring pattern to all ranks
3. The indexer computes top-k indices for local Q against all K positions
4. Sparse attention gathers selected K/V and computes output

Since KV is compressed (kv_lora_rank dimension), the communication overhead is acceptable.
"""

from typing import NamedTuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh


class DSACPMetadata(NamedTuple):
    """Metadata for DSA with Context Parallelism."""

    cp_mesh: DeviceMesh
    local_seq_len: int  # Sequence length on this rank
    global_seq_len: int  # Total sequence length across all ranks
    cp_rank: int
    cp_world_size: int


class DSARingIndexer(nn.Module):
    """
    DSA Light Indexer with ring communication for Context Parallelism.

    This module performs the indexing phase of DSA with CP support:
    1. Each rank has local Q (sharded) and local compressed KV
    2. Compressed KV is communicated via ring pattern
    3. Indexer computes scores for local Q against all K positions
    4. Returns global top-k indices for sparse attention
    """

    def __init__(
        self,
        indexer: nn.Module,  # The underlying DSALightIndexer
    ) -> None:
        super().__init__()
        self.indexer = indexer

    def forward(
        self,
        q_compressed: torch.Tensor,
        kv_compressed: torch.Tensor,
        cp_mesh: DeviceMesh,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute top-k indices using ring communication for all KV.

        Args:
            q_compressed: Local compressed Q (bsz, local_seq_len, q_dim)
            kv_compressed: Local compressed KV (bsz, local_seq_len, kv_lora_rank)
            cp_mesh: Device mesh for context parallel

        Returns:
            topk_indices: Global indices of top-k positions (bsz, local_seq_len, topk)
            topk_scores: Softmax scores for selected positions (bsz, local_seq_len, topk)
        """
        cp_world_size = cp_mesh.size(0)
        cp_rank = cp_mesh.get_local_rank()
        cp_group = cp_mesh.get_group()

        bsz, local_seq_len, kv_dim = kv_compressed.shape
        global_seq_len = local_seq_len * cp_world_size

        # Gather all KV from all ranks using all_gather
        # Shape: (bsz, global_seq_len, kv_dim)
        all_kv_list = [
            torch.empty_like(kv_compressed) for _ in range(cp_world_size)
        ]
        dist.all_gather(all_kv_list, kv_compressed, group=cp_group)
        all_kv = torch.cat(all_kv_list, dim=1)

        # Now compute indexer scores for local Q against all K
        # The indexer will apply causal masking internally
        # We need to adjust positions for proper causal masking
        topk_indices, topk_scores = self._compute_with_global_kv(
            q_compressed, all_kv, cp_rank, local_seq_len, global_seq_len
        )

        return topk_indices, topk_scores

    def _compute_with_global_kv(
        self,
        q_compressed: torch.Tensor,
        all_kv: torch.Tensor,
        cp_rank: int,
        local_seq_len: int,
        global_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute indexer scores with proper causal masking for CP.

        For CP, each rank's Q positions are offset by cp_rank * local_seq_len.
        We need to apply causal masking where Q at global position g_q can only
        attend to K at positions <= g_q.
        """
        bsz = q_compressed.shape[0]
        device = q_compressed.device
        dtype = q_compressed.dtype

        # Project Q and K to indexer space
        q_idx = self.indexer.wq_indexer(q_compressed)
        k_idx = self.indexer.wk_indexer(all_kv)
        k_idx = self.indexer.k_norm(k_idx)

        # Apply Hadamard transform if enabled
        if self.indexer.hadamard_transform:
            q_idx = self.indexer._apply_hadamard(q_idx)
            k_idx = self.indexer._apply_hadamard(k_idx)

        # Compute similarity scores
        if self.indexer.use_fp8:
            q_fp8, q_scale = self.indexer._quantize_to_fp8(q_idx)
            k_fp8, k_scale = self.indexer._quantize_to_fp8(k_idx)
            scores = torch.bmm(q_fp8.to(dtype), k_fp8.to(dtype).transpose(-2, -1))
            scores = scores * (q_scale * k_scale.transpose(-2, -1))
        else:
            scores = torch.bmm(q_idx, k_idx.transpose(-2, -1))

        # Temperature scaling
        scores = scores / self.indexer.temperature

        # Apply causal mask with CP position offset
        # Local Q position i maps to global position (cp_rank * local_seq_len + i)
        q_offset = cp_rank * local_seq_len
        q_positions = torch.arange(local_seq_len, device=device) + q_offset
        kv_positions = torch.arange(global_seq_len, device=device)

        # Causal mask: Q at position q can attend to K at positions <= q
        causal_mask = q_positions.unsqueeze(1) >= kv_positions.unsqueeze(0)
        causal_mask = causal_mask.unsqueeze(0).expand(bsz, -1, -1)
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        # Select top-k positions
        effective_topk = min(self.indexer.topk, global_seq_len)
        topk_scores_raw, topk_indices = torch.topk(
            scores, k=effective_topk, dim=-1, sorted=False
        )

        # Apply softmax to selected scores
        topk_scores = F.softmax(topk_scores_raw, dim=-1)

        return topk_indices, topk_scores


class DSASparseAttentionCP(nn.Module):
    """
    DSA Sparse Attention with Context Parallelism support.

    This module performs sparse attention with ring communication:
    1. Gathers full K/V from all CP ranks
    2. Uses global top-k indices to select relevant K/V
    3. Computes sparse attention for local Q against selected global K/V
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_scores: torch.Tensor | None,
        cp_mesh: DeviceMesh,
        scale: float | None = None,
    ) -> torch.Tensor:
        """
        Compute sparse attention with CP ring communication.

        Args:
            q: Local query tensor (bsz, n_heads, local_seq_len, head_dim)
            k: Local key tensor (bsz, n_heads, local_seq_len, head_dim)
            v: Local value tensor (bsz, n_heads, local_seq_len, v_head_dim)
            topk_indices: Global indices from indexer (bsz, local_seq_len, topk)
            topk_scores: Optional indexer scores (bsz, local_seq_len, topk)
            cp_mesh: Device mesh for context parallel
            scale: Attention scaling factor

        Returns:
            output: Attention output (bsz, n_heads, local_seq_len, v_head_dim)
        """
        cp_world_size = cp_mesh.size(0)
        cp_group = cp_mesh.get_group()

        bsz, n_heads, local_seq_len, head_dim = q.shape
        _, _, _, v_head_dim = v.shape
        topk = topk_indices.shape[-1]

        if scale is None:
            scale = head_dim**-0.5

        # Gather all K and V from all ranks
        # Shape after gather: (bsz, n_heads, global_seq_len, head_dim)
        all_k_list = [torch.empty_like(k) for _ in range(cp_world_size)]
        all_v_list = [torch.empty_like(v) for _ in range(cp_world_size)]
        dist.all_gather(all_k_list, k, group=cp_group)
        dist.all_gather(all_v_list, v, group=cp_group)
        all_k = torch.cat(all_k_list, dim=2)
        all_v = torch.cat(all_v_list, dim=2)

        global_seq_len = all_k.shape[2]

        # Now perform sparse attention using global indices
        # topk_indices: (bsz, local_seq_len, topk) - global positions

        # Expand indices for multi-head gathering
        indices_for_k = (
            topk_indices.unsqueeze(1)
            .unsqueeze(-1)
            .expand(bsz, n_heads, local_seq_len, topk, head_dim)
        )
        indices_for_v = (
            topk_indices.unsqueeze(1)
            .unsqueeze(-1)
            .expand(bsz, n_heads, local_seq_len, topk, v_head_dim)
        )

        # Expand K and V for gathering
        # all_k: (bsz, n_heads, global_seq_len, head_dim)
        # -> (bsz, n_heads, local_seq_len, global_seq_len, head_dim)
        k_expanded = all_k.unsqueeze(2).expand(
            bsz, n_heads, local_seq_len, global_seq_len, head_dim
        )
        v_expanded = all_v.unsqueeze(2).expand(
            bsz, n_heads, local_seq_len, global_seq_len, v_head_dim
        )

        # Gather selected K/V
        k_gathered = torch.gather(k_expanded, dim=3, index=indices_for_k)
        v_gathered = torch.gather(v_expanded, dim=3, index=indices_for_v)

        # Compute attention scores
        q_expanded = q.unsqueeze(-2)  # (bsz, n_heads, local_seq_len, 1, head_dim)
        attn_scores = torch.matmul(q_expanded, k_gathered.transpose(-2, -1))
        attn_scores = attn_scores.squeeze(-2) * scale  # (bsz, n_heads, local_seq_len, topk)

        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Optionally combine with indexer scores
        if topk_scores is not None:
            topk_scores_expanded = topk_scores.unsqueeze(1)
            attn_weights = attn_weights * topk_scores_expanded
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-12)

        # Compute output
        attn_weights_expanded = attn_weights.unsqueeze(-1)
        output = (attn_weights_expanded * v_gathered).sum(dim=-2)

        return output


class DSAContextParallelWrapper(nn.Module):
    """
    Complete DSA wrapper with Context Parallelism support.

    This combines the ring indexer and sparse attention into a single module
    that can be used with CP.
    """

    def __init__(
        self,
        indexer: nn.Module,  # DSALightIndexer
        combine_with_indexer_scores: bool = True,
    ) -> None:
        super().__init__()
        self.ring_indexer = DSARingIndexer(indexer)
        self.sparse_attention = DSASparseAttentionCP()
        self.combine_scores = combine_with_indexer_scores

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_compressed: torch.Tensor,
        kv_compressed: torch.Tensor,
        cp_mesh: DeviceMesh,
        scale: float | None = None,
    ) -> torch.Tensor:
        """
        Complete DSA forward pass with CP.

        Args:
            q: Query tensor (bsz, n_heads, local_seq_len, head_dim)
            k: Key tensor (bsz, n_heads, local_seq_len, head_dim)
            v: Value tensor (bsz, n_heads, local_seq_len, v_head_dim)
            q_compressed: Compressed Q for indexer (bsz, local_seq_len, q_dim)
            kv_compressed: Compressed KV for indexer (bsz, local_seq_len, kv_lora_rank)
            cp_mesh: Device mesh for context parallel
            scale: Attention scaling factor

        Returns:
            output: Attention output (bsz, n_heads, local_seq_len, v_head_dim)
        """
        # Step 1: Compute global top-k indices using ring indexer
        topk_indices, topk_scores = self.ring_indexer(
            q_compressed, kv_compressed, cp_mesh
        )

        # Step 2: Compute sparse attention with global indices
        output = self.sparse_attention(
            q,
            k,
            v,
            topk_indices,
            topk_scores if self.combine_scores else None,
            cp_mesh,
            scale,
        )

        return output


class DSASparseAttentionCPWrapper(nn.Module):
    """
    Wrapper for DSA sparse attention that is compatible with _ContextParallel.

    This wrapper follows the same pattern as FlexAttentionWrapper and
    ScaledDotProductAttentionWrapper, with q, k, v as the first three arguments.

    Note: This wrapper requires the cp_mesh to be set before forward pass
    via set_cp_mesh() method, since _ContextParallel doesn't pass mesh info.
    """

    def __init__(self, dsa_cp_wrapper: DSAContextParallelWrapper) -> None:
        super().__init__()
        self.dsa_cp_wrapper = dsa_cp_wrapper
        self._cp_mesh: DeviceMesh | None = None
        self._q_compressed: torch.Tensor | None = None
        self._kv_compressed: torch.Tensor | None = None

    def set_cp_mesh(self, cp_mesh: DeviceMesh) -> None:
        """Set the CP mesh for this attention module."""
        self._cp_mesh = cp_mesh

    def set_compressed_inputs(
        self,
        q_compressed: torch.Tensor,
        kv_compressed: torch.Tensor,
    ) -> None:
        """Set compressed Q/KV for the indexer before calling forward."""
        self._q_compressed = q_compressed
        self._kv_compressed = kv_compressed

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
    ) -> torch.Tensor:
        """
        Forward pass compatible with _ContextParallel signature.

        Note: set_compressed_inputs() must be called before this.
        """
        if self._cp_mesh is None:
            raise RuntimeError("CP mesh not set. Call set_cp_mesh() first.")
        if self._q_compressed is None or self._kv_compressed is None:
            raise RuntimeError(
                "Compressed inputs not set. Call set_compressed_inputs() first."
            )

        output = self.dsa_cp_wrapper(
            q,
            k,
            v,
            self._q_compressed,
            self._kv_compressed,
            self._cp_mesh,
            scale,
        )

        # Clear compressed inputs after use
        self._q_compressed = None
        self._kv_compressed = None

        return output
