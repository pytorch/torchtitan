# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""GLM sparse attention using public PyTorch block-mask construction.

T/Q/K are token/query/key counts; S is selected keys per query; N is heads.
D/H/R/P are model/head/latent-or-rotary/pass-through widths; V is value width.
Mask construction retains O(Q*K) storage. Attention delegates to the common
FlexAttention backend rather than a model-owned kernel; the indexer still
computes dense per-head ranking scores.
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn.functional as F
import torch_remat as remat
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.common import ComplexRoPE, LayerNorm, Linear
from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.common.attention.cp_attention import (
    AllGatherKVFlexInnerAttention,
)


def create_dsa_causal_mask(
    positions_T: torch.Tensor,
    *,
    dtype: torch.dtype,
    padding_mask_T: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a packed-document causal additive mask of shape ``[1, T, T]``.

    ``positions_T`` restarts at zero for every packed document. The cumulative
    restart count is therefore a document id. A query can see a key only when
    both tokens belong to the same document and the key is not in the future.
    Disallowed entries use the minimum finite value for indexer ranking.
    The attention mask also excludes them after top-k selection.
    """
    T = positions_T.shape[0]
    token_indices_T = torch.arange(T, device=positions_T.device)
    document_ids_T = (positions_T == 0).cumsum(0)
    allowed_TT = (token_indices_T[:, None] >= token_indices_T[None, :]) & (
        document_ids_T[:, None] == document_ids_T[None, :]
    )
    if padding_mask_T is not None:
        allowed_TT = allowed_TT & padding_mask_T[:, None] & padding_mask_T[None, :]
    return torch.zeros(
        1, T, T, dtype=dtype, device=positions_T.device
    ).masked_fill(~allowed_TT.unsqueeze(0), torch.finfo(dtype).min)


class DSAIndexerTopK(Module):
    """Compute local-query top-k indices against a global key sequence.

    For every indexer head ``n`` and query/key pair, the indexer computes
    ``relu(dot(q[q,n,:], k[k,:]) * scale)``. ``weights_QN`` then reduces the
    head dimension into one score per ``(query, key)``. The causal/document
    mask is added before the final top-k. The output is integer metadata; no
    gradient flows through the discrete selection.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        index_topk: int
        softmax_scale: float


    def __init__(self, config: Config) -> None:
        super().__init__()
        self.index_topk = config.index_topk
        self.softmax_scale = config.softmax_scale

    def forward(
        self,
        q_QNH: torch.Tensor,
        k_KH: torch.Tensor,
        weights_QN: torch.Tensor,
        attention_mask_QK: torch.Tensor,
    ) -> torch.Tensor:
        # [Q,N,H] x [H,K] -> [N,Q,K]. Index scores intentionally use FP32:
        # the K/K+1 boundary can change under a single low-precision ULP.
        scores_NQK = torch.matmul(
            q_QNH.float().transpose(0, 1), k_KH.float().transpose(0, 1)
        ) * self.softmax_scale
        scores_NQK = F.relu(scores_NQK)
        # Each query predicts N mixing weights, producing one global-key score
        # after the weighted reduction over indexer heads.
        index_scores_QK = torch.matmul(
            weights_QN.unsqueeze(1), scores_NQK.transpose(0, 1)
        ).squeeze(1)
        index_scores_QK = index_scores_QK + attention_mask_QK.float()
        topk = min(self.index_topk, index_scores_QK.shape[-1])
        return index_scores_QK.topk(topk, dim=-1).indices.to(torch.int32)


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
        self.softmax_scale = config.head_dim**-0.5
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
        attention_mask_TK: torch.Tensor | None,
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
        weights_TN = weights_TN.float() * (self.n_heads**-0.5)
        if attention_mask_TK is None:
            attention_mask_TK = create_dsa_causal_mask(
                positions_T, dtype=torch.float32
            )[0]
        return self.topk(q_TNH, k_TH, weights_TN, attention_mask_TK)


def selected_mask(
    indices_QS: torch.Tensor, additive_mask_1QK: torch.Tensor
) -> torch.Tensor:
    """Intersect selected positions with the existing causal/document mask.

    Negative entries denote padding. Duplicate indices represent a set, not
    repeated probability mass. Invalid positive indices are excluded as well.
    The supplied mask uses zero for allowed entries, not arbitrary score bias.
    """
    Q, K = additive_mask_1QK.shape[-2:]
    valid_QS = (indices_QS >= 0) & (indices_QS < K)
    safe_QS = indices_QS.clamp(0, K - 1).long()
    counts_QK = torch.zeros((Q, K), dtype=torch.int32, device=indices_QS.device)
    counts_QK.scatter_add_(1, safe_QS, valid_QS.to(torch.int32))
    return (counts_QK > 0) & (additive_mask_1QK[0] == 0)


def build_dsa_block_mask(
    indices_QS: torch.Tensor,
    additive_mask_1QK: torch.Tensor,
    block_size: int = 128,
) -> BlockMask:
    """Build conservative KV block lists and an exact token-level predicate."""
    allowed_QK = selected_mask(indices_QS, additive_mask_1QK)
    Q, K = allowed_QK.shape
    B = block_size
    padded = F.pad(allowed_QK, (0, (-K) % B, 0, (-Q) % B))
    occupied = padded.reshape((Q + B - 1) // B, B, (K + B - 1) // B, B)
    occupied = occupied.any(dim=3).any(dim=1)[None, None]
    counts = occupied.sum(-1).to(torch.int32)
    # Sorting float32 0/1 keys preserves the exact occupancy order while
    # allowing accelerator backends to use their native stable sort path.
    blocks = occupied.to(torch.float32).argsort(
        dim=-1, descending=True, stable=True
    )

    def mask_mod(b, h, q, k):
        # Flex may evaluate padded lanes; clamp reads before applying bounds.
        return (q < Q) & (k < K) & allowed_QK[q.clamp(max=Q - 1), k.clamp(max=K - 1)]

    return BlockMask.from_kv_blocks(
        counts, blocks.to(torch.int32), BLOCK_SIZE=B,
        mask_mod=mask_mod, seq_lengths=(Q, K),
    )


class Glm5FlexAttention(FlexInnerAttention):
    """Expanded MLA Q/K/V with GLM token-selected block-sparse attention."""

    @dataclass(kw_only=True, slots=True)
    class Config(FlexInnerAttention.Config):
        block_size: int = 128
        attention_dropout: float = 0.0

        def build_attention_metadata(self, positions: torch.Tensor, **kwargs):
            # GLM builds a dense legality mask before the learned top-k stage.
            # The resulting sparse BlockMask depends on the selected indices
            # and therefore cannot be built by Decoder's generic metadata path.
            del positions, kwargs
            return None

        def __post_init__(self):
            if self.attention_dropout != 0.0:
                raise ValueError(
                    "GLM FlexAttention currently requires attention_dropout=0"
                )

    def __init__(self, config):
        super().__init__(config)
        self.block_size = config.block_size

    def forward(
        self,
        q_QNH: torch.Tensor,
        k_KNH: torch.Tensor,
        v_KNV: torch.Tensor,
        attention_metadata: torch.Tensor,
        topk_indices_QS: torch.Tensor,
        *,
        scale: float | None = None,
    ) -> torch.Tensor:
        mask = build_dsa_block_mask(
            topk_indices_QS, attention_metadata, self.block_size
        )
        return super().forward(
            q_QNH, k_KNH, v_KNV, attention_metadata=mask, scale=scale
        )


class Glm5AllGatherKVFlexAttention(AllGatherKVFlexInnerAttention):
    """GLM DSA with local queries and official all-gathered CP keys/values."""

    @dataclass(kw_only=True, slots=True)
    class Config(AllGatherKVFlexInnerAttention.Config):
        block_size: int = 128
        attention_dropout: float = 0.0

        def build_attention_metadata(self, positions: torch.Tensor, **kwargs):
            del positions, kwargs
            return None

        def __post_init__(self):
            if self.attention_dropout != 0.0:
                raise ValueError(
                    "GLM FlexAttention currently requires attention_dropout=0"
                )

    def __init__(self, config: Config):
        super().__init__(config)
        self.block_size = config.block_size

    def forward(
        self,
        q_QNH: torch.Tensor,
        k_KNH: torch.Tensor,
        v_KNV: torch.Tensor,
        attention_metadata: torch.Tensor,
        topk_indices_QS: torch.Tensor,
        *,
        scale: float | None = None,
    ) -> torch.Tensor:
        mask = build_dsa_block_mask(
            topk_indices_QS, attention_metadata, self.block_size
        )
        return super().forward(
            q_QNH, k_KNH, v_KNV, attention_metadata=mask, scale=scale
        )
