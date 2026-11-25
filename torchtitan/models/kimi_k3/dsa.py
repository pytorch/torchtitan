# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3.1 DeepSeek Sparse Attention components."""

from dataclasses import dataclass
from typing import Literal

import spmd_types as spmd
import torch
import torch.nn as nn
import torch_remat as remat

from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import (
    Attention,
    create_varlen_metadata_for_document,
    InnerAttention,
    local_head_split,
)
from torchtitan.models.common.attn_gym import AuxRequest, gather_attn
from torchtitan.models.common.linear import Linear, maybe_gather_tp_input
from torchtitan.models.common.nn_modules import LayerNorm

# TODO: move InnerIndexer, SparseIndexerLoss and their metadata to models/common/
# after #5863 lands; Kimi K3.1 is their second user.
from torchtitan.models.deepseek_v4.attention import (
    DeepSeekV4CompressedSparseAttentionMetadata,
)
from torchtitan.models.deepseek_v4.compressor import InnerIndexer, SparseIndexerLoss

from .attention import KimiMLAAttention

__all__ = [
    "AbsorbedKVProjection",
    "KimiDSAAttention",
    "KimiDSAIndexer",
    "KimiSparseAttention",
]

# Shape suffixes:
# T = packed token count (num_tokens)
# S = selected token slots
# D = model dimension (dim)
# C = low-rank query channels (q_lora_rank)
# H = main attention head count (n_heads)
# K = main query/key head dimension (N + R)
# N = per-head non-positional key dimension (qk_nope_head_dim)
# R = head-shared key dimension (qk_rope_head_dim)
# V = main value head dimension
# L = KV latent dimension (kv_lora_rank)
# G = latent and head-shared key channels (L + R)
# P = packed per-head KV up-projection channels (N + V)
# I = indexer head count (num_index_heads)
# E = indexer head dimension (index_head_dim)
# J = flattened projection channels


class AbsorbedKVProjection(Module):
    """Per-head MLA key and value up-projection weights.

    The weight has the layout of the MLA ``wkv_b`` linear. ``forward`` returns
    it split per head, for a caller that absorbs the key part into the query
    and applies the value part after attention.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        kv_lora_rank: int
        num_heads: int
        qk_nope_head_dim: int
        v_head_dim: int

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim
        self.weight = nn.Parameter(
            torch.empty(
                config.num_heads * (config.qk_nope_head_dim + config.v_head_dim),
                config.kv_lora_rank,
            )
        )

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        weight_HPL = self.weight.unflatten(
            0, (-1, self.qk_nope_head_dim + self.v_head_dim)
        )
        w_uk_HNL, w_uv_HVL = weight_HPL.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=1
        )
        return w_uk_HNL, w_uv_HVL


class KimiDSAIndexer(Module):
    """Project normalized low-rank queries and raw inputs into DSA scores."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_index_heads: int
        index_head_dim: int
        wq_b: Linear.Config
        wk: Linear.Config
        k_norm: LayerNorm.Config
        weights_proj: Linear.Config

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.num_index_heads = config.num_index_heads
        self.index_head_dim = config.index_head_dim
        self.wq_b = config.wq_b.build()
        self.wk = config.wk.build()
        self.k_norm = config.k_norm.build()
        self.weights_proj = config.weights_proj.build()

    def forward(
        self, q_TC: torch.Tensor, x_TD: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx_q_TJ = self.wq_b(q_TC)
        remat.recompute_needs_tensor(idx_q_TJ)
        idx_q_TIE = idx_q_TJ.view(
            q_TC.shape[0], self.num_index_heads, self.index_head_dim
        )
        idx_k_TE = self.wk(x_TD)
        remat.recompute_needs_tensor(idx_k_TE)
        idx_k_TE = self.k_norm(idx_k_TE)
        idx_w_TI = self.weights_proj(x_TD)
        remat.recompute_needs_tensor(idx_w_TI)
        idx_w_TI = idx_w_TI * (
            self.index_head_dim**-0.5 * self.num_index_heads**-0.5
        )
        return idx_q_TIE, idx_k_TE, idx_w_TI


class KimiSparseAttention(InnerAttention):
    """Absorbed MLA over indexer-selected tokens, run by Attention Gym.

    Each query head absorbs its key up-projection, so all heads attend the
    shared latent ``kv_c_normed``, which is also the value, and the shared
    RoPE key ``k_shared``. The latent output is up-projected per head.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        wkv_b: AbsorbedKVProjection.Config
        inner_indexer: InnerIndexer.Config
        aux_loss: SparseIndexerLoss.Config | None = None
        impl: Literal["fused", "reference"] = "fused"
        """Attention Gym ``gather_attn`` implementation. ``"fused"`` runs the
        accelerator kernels and raises on inputs they do not support;
        ``"reference"`` is eager PyTorch on any device, for numerics checks."""

        @property
        def is_full_attention(self) -> bool:
            return True

        def build_attention_metadata(
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> DeepSeekV4CompressedSparseAttentionMetadata:
            varlen = create_varlen_metadata_for_document(
                positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )
            empty = torch.empty(0, dtype=torch.int32, device=positions.device)
            return DeepSeekV4CompressedSparseAttentionMetadata(
                cu_seq_q=varlen.cu_seq_q,
                cu_seq_k=varlen.cu_seq_k,
                max_q=varlen.max_q,
                max_k=varlen.max_k,
                window_size=varlen.window_size,
                cu_seq_compressed=varlen.cu_seq_q,
                group_start=empty,
                group_first=torch.empty(0, dtype=torch.bool, device=positions.device),
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.wkv_b = config.wkv_b.build()
        self.inner_indexer = config.inner_indexer.build()
        self.impl = config.impl
        self.aux_loss = config.aux_loss.build() if config.aux_loss is not None else None

    def forward(
        self,
        q_THK: torch.Tensor,
        kv_c_normed_TL: torch.Tensor,
        k_shared_TR: torch.Tensor,
        *,
        idx_q_TIE: torch.Tensor,
        idx_k_TE: torch.Tensor,
        idx_w_TI: torch.Tensor,
        attention_metadata: DeepSeekV4CompressedSparseAttentionMetadata,
        scale: float,
        aux_loss_denominator: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        aux_loss = self.aux_loss if self.training else None
        cu_seqlens = attention_metadata.cu_seq_q
        topk_indices_TS, topk_scores_TS = self.inner_indexer(
            idx_q_TIE,
            idx_k_TE,
            idx_w_TI,
            attention_metadata=attention_metadata,
            return_scores=aux_loss is not None,
        )

        nope_dim = q_THK.shape[-1] - k_shared_TR.shape[-1]
        w_uk_HNL, w_uv_HVL = self.wkv_b()
        q_nope_THN, q_shared_THR = q_THK.split(
            [nope_dim, k_shared_TR.shape[-1]], dim=-1
        )
        q_latent_THL = torch.einsum(
            "thn,hnl->thl",  # codespell:ignore thn
            q_nope_THN,
            w_uk_HNL,
        )
        with spmd.no_typecheck():
            out_1HTL, aux = gather_attn(
                (
                    q_latent_THL.transpose(0, 1).unsqueeze(0),
                    q_shared_THR.transpose(0, 1).unsqueeze(0),
                ),
                None,
                (kv_c_normed_TL[None, None], k_shared_TR[None, None]),
                topk_indices_TS.unsqueeze(0),
                sliding_window_size=0,
                cu_seqlens=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                scale=scale,
                impl=self.impl,
                return_aux=AuxRequest(lse=aux_loss is not None),
            )
            out_THL = out_1HTL.squeeze(0).transpose(0, 1)
            out_THV = torch.einsum("thl,hvl->thv", out_THL, w_uv_HVL)
            if aux_loss is not None:
                assert (
                    topk_scores_TS is not None
                    and aux_loss_denominator is not None
                    and aux.lse is not None
                )
                out_THV = aux_loss(
                    torch.cat((q_latent_THL, q_shared_THR), dim=-1).detach(),
                    torch.cat((kv_c_normed_TL, k_shared_TR), dim=-1).detach(),
                    topk_indices_TS,
                    topk_scores_TS,
                    aux.lse.squeeze(0).transpose(0, 1).detach(),
                    attention_metadata,
                    carrier=out_THV,
                    denominator=aux_loss_denominator,
                    padding_mask_T=padding_mask,
                )
        if spmd.is_type_checking():
            spmd.assert_type(
                out_THV, spmd.get_local_type(q_THK), spmd.get_partition_spec(q_THK)
            )
        return out_THV


class KimiDSAAttention(KimiMLAAttention):
    """Kimi K3 gated MLA with a DeepSeek sparse-attention indexer."""

    @dataclass(kw_only=True, slots=True)
    class Config(KimiMLAAttention.Config):
        inner_attention: KimiSparseAttention.Config  # pyrefly: ignore[bad-override]
        indexer: KimiDSAIndexer.Config

        def __post_init__(self) -> None:
            Attention.Config.__post_init__(self)
            if not isinstance(self.inner_attention, KimiSparseAttention.Config):
                raise ValueError(
                    "KimiDSAAttention requires a KimiSparseAttention.Config, "
                    f"but got {type(self.inner_attention).__qualname__}."
                )
            aux_loss = self.inner_attention.aux_loss
            if aux_loss is not None and (
                aux_loss.softmax_scale != self.softmax_scale
                or aux_loss.num_heads != self.n_heads
            ):
                raise ValueError(
                    "The indexer loss must use the attention's softmax scale "
                    f"({self.softmax_scale}) and head count ({self.n_heads}), but got "
                    f"{aux_loss.softmax_scale} and {aux_loss.num_heads}."
                )

        def flops(self, positions: torch.Tensor) -> int:
            from torchtitan.models.common.flops import quadratic_attention_flops

            # A causal query at document position p has p + 1 keys. The indexer
            # scores all of them; attention reads only the top-k.
            num_keys = positions + 1
            attention_flops = quadratic_attention_flops(
                num_heads=self.n_heads,
                qk_head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
                v_head_dim=self.v_head_dim,
                num_keys=num_keys.clamp(max=self.inner_attention.inner_indexer.topk),
            )

            # TODO: the indexer scores all keys without grad, then recomputes only the
            # top-k with grad for the indexer loss. Counting only that cuts Kimi K3.1
            # Pro's estimate by 0.8% at 32k and 21.7% at 1M (one document).
            # DeepSeek-V4's indexer needs the same fix.
            indexer_flops = (
                6
                * self.indexer.num_index_heads
                * self.indexer.index_head_dim
                * int(num_keys.sum())
            )
            return attention_flops + indexer_flops

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.indexer = config.indexer.build()

    def forward(  # pyrefly: ignore[bad-override]
        self,
        x_TD: torch.Tensor,
        attention_metadata: DeepSeekV4CompressedSparseAttentionMetadata,
        positions: torch.Tensor | None = None,
        *,
        aux_loss_denominator: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions
        x_TD = maybe_gather_tp_input(self, x_TD)
        q_TC = self.wq_a(x_TD)
        remat.recompute_needs_tensor(q_TC)
        q_normed_TC = self.q_norm(q_TC)
        q_TJ = self.wq_b(q_normed_TC)
        remat.recompute_needs_tensor(q_TJ)
        q_THK = local_head_split(q_TJ, self.q_head_dim, cp_shard_dim=0)

        compressed_kv_TG = self.wkv_a(x_TD)
        remat.recompute_needs_tensor(compressed_kv_TG)
        kv_latent_TL, k_shared_TR = torch.split(
            compressed_kv_TG,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv_c_normed_TL = self.kv_norm(kv_latent_TL)
        idx_q_TIE, idx_k_TE, idx_w_TI = self.indexer(
            q_normed_TC.detach(), x_TD.detach()
        )
        out_THV = remat.region(
            self.inner_attention,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(
            q_THK,
            kv_c_normed_TL,
            k_shared_TR,
            attention_metadata=attention_metadata,
            scale=self.scale,
            idx_q_TIE=idx_q_TIE,
            idx_k_TE=idx_k_TE,
            idx_w_TI=idx_w_TI,
            aux_loss_denominator=aux_loss_denominator,
            padding_mask=padding_mask,
        )
        gate_TD = self.gate(x_TD)
        remat.recompute_needs_tensor(out_THV, gate_TD)
        out_TD = out_THV.flatten(-2)
        out_TD = out_TD * torch.sigmoid(gate_TD)
        return self.wo(out_TD)
