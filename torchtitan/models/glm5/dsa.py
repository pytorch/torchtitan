# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLM sparse attention with Attention Gym indexing and PyTorch block masks.

T/Q/K are token/query/key counts; S is selected keys per query; N is heads.
D/H/R/P are model/head/latent-or-rotary/pass-through widths; V is value width.
Document metadata is linear in tokens. Attention delegates to the common
FlexAttention backend; Attention Gym owns the weighted-ReLU top-k selection.
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch_remat as remat
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.distributed.context_parallel import cp_all_gather_tensors

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.base.module import Module
from torchtitan.models.common import ComplexRoPE, LayerNorm, Linear
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.common.attention.cp_mla import AllGatherKVMLAFlexAttention
from torchtitan.models.common.attn_gym import lightning_indexer, require_attn_gym


@dataclass(frozen=True)
class DsaMetadata:
    """Global keys in CP order and rank-local query indices; all tensors are 1D."""

    positions_T: torch.Tensor
    documents_T: torch.Tensor
    valid_T: torch.Tensor
    query_indices_Q: torch.Tensor
    order_T: torch.Tensor

    @classmethod
    def from_positions(cls, positions, padding_mask=None):
        valid = (
            torch.ones_like(positions, dtype=torch.bool)
            if padding_mask is None
            else ~padding_mask
        )
        order = torch.arange(positions.numel(), device=positions.device)
        return cls(positions, (positions == 0).cumsum(0), valid, order, order)


class Glm5CPMLAAttention(AllGatherKVMLAFlexAttention):
    """Reuse upstream compact KV communication with GLM document metadata."""

    @dataclass(kw_only=True, slots=True)
    class Config(AllGatherKVMLAFlexAttention.Config):
        pass

    @staticmethod
    def prepare_cp_metadata(  # pyrefly: ignore[bad-override]
        attention_metadata: DsaMetadata, *, permutation: torch.Tensor | None
    ) -> DsaMetadata:
        metadata = attention_metadata
        group = spmd_mesh_group(MeshAxisName.CP)
        if group is None:
            return metadata
        count = metadata.positions_T.numel()
        if count % group.size():
            raise ValueError("DSA token count must divide evenly across CP ranks")
        order = (
            torch.arange(count, device=metadata.positions_T.device)
            if permutation is None
            else permutation.reshape(-1)
        )
        local = count // group.size()
        queries = torch.arange(
            group.rank() * local, (group.rank() + 1) * local, device=order.device
        )
        return DsaMetadata(
            metadata.positions_T[order],
            metadata.documents_T[order],
            metadata.valid_T[order],
            queries,
            metadata.order_T[order],
        )


class DSAIndexerTopK(Module):
    """Compute local-query top-k indices against a global key sequence.

    Attention Gym processes complete packed documents and returns indices local
    to each document. Compact valid tokens before selection, then map selections
    back to the global key sequence expected by FlexAttention. CP gathers query
    features as well as keys because the indexer API requires complete documents.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        index_topk: int

    def __init__(self, config: Config) -> None:
        super().__init__()
        require_attn_gym(lightning_indexer)
        self.index_topk = config.index_topk

    @torch.no_grad()
    def forward(
        self,
        q_QNH: torch.Tensor,
        k_KH: torch.Tensor,
        weights_QN: torch.Tensor,
        metadata: DsaMetadata,
    ) -> torch.Tensor:
        K = metadata.positions_T.numel()
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        if cp_group is not None:
            # TODO: Keep queries local once Attention Gym supports query chunks.
            q_QNH, weights_QN = cp_all_gather_tensors(
                (q_QNH, weights_QN),
                reduce_dtype=torch.float32,
            )
        if q_QNH.shape[0] != K:
            raise ValueError("DSA indexing requires complete query and key sequences")

        # Keep token order within each document while moving padding to the end.
        # Fixed-capacity offsets use repeated endpoints for empty documents.
        # Restore original document/token order after any CP load-balancer permutation.
        token_order_K = metadata.positions_T.argsort(stable=True)
        token_order_K = token_order_K[
            metadata.documents_T[token_order_K].argsort(stable=True)
        ]
        token_order_K = token_order_K[
            (~metadata.valid_T[token_order_K]).to(torch.int32).argsort(stable=True)
        ]
        valid_K = metadata.valid_T[token_order_K]
        document_starts_K = metadata.documents_T[token_order_K]
        token_indices_K = torch.arange(K, device=q_QNH.device, dtype=torch.int32)
        starts_K = valid_K & (
            (token_indices_K == 0) | (document_starts_K != document_starts_K.roll(1))
        )
        num_valid = valid_K.sum(dtype=torch.int32)
        cu_seqlens = torch.cat(
            (
                torch.where(starts_K, token_indices_K, num_valid).sort().values,
                num_valid[None],
            )
        )
        document_offsets_K = torch.where(starts_K, token_indices_K, 0).cummax(0).values
        # Keep FP32 ranking, including FP32 mixing weights. Attention Gym's
        # fused indexer requires FP16/BF16 inputs and would change this contract.
        indices_KS = lightning_indexer(
            q_QNH[token_order_K].float().unsqueeze(0),
            k_KH[token_order_K].float().unsqueeze(0),
            weights_QN[token_order_K].float().unsqueeze(0),
            min(self.index_topk, K),
            causal=True,
            cu_seqlens=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            impl="reference",
        ).squeeze(0)
        packed_indices_KS = document_offsets_K[:, None] + indices_KS.clamp_min(0)
        global_indices_KS = token_order_K[packed_indices_KS.long()].to(torch.int32)
        global_indices_KS = global_indices_KS.masked_fill(indices_KS < 0, -1)
        indices_KS = torch.empty_like(global_indices_KS).index_copy_(
            0, token_order_K, global_indices_KS
        )
        if cp_group is not None:
            return spmd.redistribute(indices_KS, cp_group, src=spmd.R, dst=spmd.S(0))
        return indices_KS


class Glm5DsaIndexer(Module):
    """Frozen pretrained query-to-key selector used by DSA.

    ``hidden_states_TD`` supplies both the shared key and per-head mixing
    weights. ``q_resid_TR`` reuses MLA's normalized Q LoRA representation so
    the main attention and indexer observe the same query state. RoPE is
    applied only to the configured rotary prefix; the remaining features pass
    through unchanged.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        q_lora_rank: int
        n_heads: int
        head_dim: int
        qk_rope_head_dim: int
        index_topk: int
        wq_b: Linear.Config
        wk: Linear.Config
        k_norm: LayerNorm.Config
        weights_proj: Linear.Config
        rope: ComplexRoPE.Config
        topk: DSAIndexerTopK.Config

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.wq_b = config.wq_b.build()
        self.wk = config.wk.build()
        self.k_norm = config.k_norm.build()
        self.weights_proj = config.weights_proj.build().float()
        self.rope = config.rope.build()
        self.topk = config.topk.build()
        # Released GLM-5 checkpoints contain a pretrained indexer. This model
        # has no auxiliary indexer objective, so LM training must keep it fixed
        # and exclude its parameters from optimizer state.
        self.requires_grad_(False)

    def _apply(self, fn, recurse: bool = True):
        # ``model.to(dtype=...)`` visits frozen modules too. Restore this gate
        # to FP32 after the generic transform because its boundary decisions
        # are more sensitive than the surrounding BF16 attention projections.
        super()._apply(fn, recurse=recurse)
        self.weights_proj.float()
        return self

    @torch.no_grad()
    def forward(
        self,
        hidden_states_TD: torch.Tensor,
        q_resid_TR: torch.Tensor,
        positions_T: torch.Tensor,
        attention_mask_TK: DsaMetadata | None,
    ) -> torch.Tensor:
        T = hidden_states_TD.shape[0]
        # Build query and key features in the same indexer space. The key has a
        # singleton head dimension because every query head searches one shared
        # global key sequence.
        q_TNH = self.wq_b(q_resid_TR)
        remat.recompute_needs_tensor(q_TNH)
        q_TNH = q_TNH.view(T, self.n_heads, self.head_dim)
        q_rot_TNR, q_pass_TNP = torch.split(
            q_TNH,
            [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
            dim=-1,
        )
        k_TH = self.wk(hidden_states_TD)
        remat.recompute_needs_tensor(k_TH)
        k_TH = self.k_norm(k_TH)
        remat.recompute_needs_tensor(k_TH)
        k_T1H = k_TH.unsqueeze(1)
        k_rot_T1R, k_pass_T1P = torch.split(
            k_T1H,
            [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
            dim=-1,
        )
        q_rot_TNR, k_rot_T1R = self.rope(q_rot_TNR, k_rot_T1R, positions_T)
        remat.recompute_needs_tensor(q_rot_TNR, k_rot_T1R)
        q_TNH = torch.cat((q_rot_TNR, q_pass_TNP), dim=-1)
        k_TH = torch.cat((k_rot_T1R, k_pass_T1P), dim=-1).squeeze(1)

        # Context parallelism keeps queries local but every query ranks the
        # global key sequence. Match the KV-all-gather attention contract by
        # gathering indexer keys over CP before the discrete top-k boundary.
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        if cp_group is not None:
            k_TH = spmd.redistribute(
                k_TH,
                cp_group,
                src=spmd.S(0),
                dst=spmd.R,
            )

        # The learned per-token head weights turn N head-specific similarities
        # into the scalar [T,T] ranking consumed by top-k.
        weights_TN = self.weights_proj(
            hidden_states_TD.to(self.weights_proj.weight.dtype)
        )
        remat.recompute_needs_tensor(weights_TN)
        # Attention Gym applies the 1/sqrt(N*H) indexer normalization.
        if attention_mask_TK is None:
            attention_mask_TK = DsaMetadata.from_positions(positions_T)
        return self.topk(q_TNH, k_TH, weights_TN, attention_mask_TK)


def build_dsa_block_mask(
    indices_QS: torch.Tensor,
    metadata: DsaMetadata,
    block_size: int = 128,
) -> BlockMask:
    """Build conservative KV block lists and an exact token-level predicate."""
    Q, S = indices_QS.shape
    K = metadata.positions_T.numel()
    B = block_size
    valid_QS = (indices_QS >= 0) & (indices_QS < K)
    safe_QS = indices_QS.clamp(0, K - 1).long()
    queries_Q = metadata.query_indices_Q
    valid_QS = valid_QS & metadata.valid_T[queries_Q, None] & metadata.valid_T[safe_QS]
    valid_QS = valid_QS & (
        metadata.documents_T[queries_Q, None] == metadata.documents_T[safe_QS]
    )
    valid_QS = valid_QS & (
        metadata.positions_T[queries_Q, None] >= metadata.positions_T[safe_QS]
    )
    num_q_blocks, num_k_blocks = (Q + B - 1) // B, (K + B - 1) // B
    block_ids_QS = (
        torch.arange(Q, device=indices_QS.device)[:, None] // B
    ) * num_k_blocks + safe_QS // B
    occupancy = torch.zeros(
        num_q_blocks * num_k_blocks, device=indices_QS.device, dtype=torch.int32
    )
    occupancy.scatter_add_(
        0, block_ids_QS.flatten(), valid_QS.flatten().to(torch.int32)
    )
    occupied = occupancy.view(1, 1, num_q_blocks, num_k_blocks) > 0
    counts = occupied.sum(-1).to(torch.int32)
    blocks = occupied.to(torch.float32).argsort(dim=-1, descending=True, stable=True)

    def mask_mod(b, h, q, k):
        row = q.clamp(max=Q - 1)
        return (
            (q < Q)
            & (k < K)
            & ((indices_QS[row] == k[..., None]) & valid_QS[row]).any(dim=-1)
        )

    return BlockMask.from_kv_blocks(
        counts,
        blocks.to(torch.int32),
        BLOCK_SIZE=B,
        mask_mod=mask_mod,
        seq_lengths=(Q, K),
    )


class Glm5FlexAttention(FlexInnerAttention):
    """Expanded MLA Q/K/V with GLM token-selected block-sparse attention."""

    @dataclass(kw_only=True, slots=True)
    class Config(FlexInnerAttention.Config):
        block_size: int | tuple[int, int] = 128
        attention_dropout: float = 0.0

        def __post_init__(self):
            if not isinstance(self.block_size, int) or self.block_size <= 0:
                raise ValueError(
                    "GLM DSA requires a positive square integer block_size"
                )
            if self.attention_dropout != 0.0:
                raise ValueError(
                    "GLM FlexAttention currently requires attention_dropout=0"
                )

    def __init__(self, config):
        super().__init__(config)
        assert isinstance(config.block_size, int)
        self.block_size = config.block_size

    def forward(  # pyrefly: ignore[bad-override]
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        attention_metadata: DsaMetadata,
        topk_indices_QS: torch.Tensor,
        scale: float | None = None,
    ) -> torch.Tensor:
        mask = build_dsa_block_mask(
            topk_indices_QS, attention_metadata, self.block_size
        )
        return super().forward(
            q_THK, k_THK, v_THV, attention_metadata=mask, scale=scale
        )
