# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen Sparse Attention with complete-block selection and indexer distillation."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch_remat as remat
from torch.distributed._functional_collectives import all_reduce, wait_tensor
from torch.nn.attention import sdpa_kernel, SDPBackend

from torchtitan.distributed.spmd_types import spmd_mesh_group, spmd_mesh_size
from torchtitan.models.base.module import Module
from torchtitan.models.common import Linear
from torchtitan.models.common.attention import (
    InnerAttention,
    local_head_split,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.attention.attention import (
    create_varlen_metadata_for_document,
)
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.common.linear import maybe_gather_tp_input
from torchtitan.models.qwen3_5.model import OffsetRMSNorm, Qwen35Attention
from torchtitan.models.qwen3_5.rope import MRoPE

from .embedding import document_starts

# T = packed tokens, U = valid tokens, G = complete blocks, R = block size,
# B = selected blocks, J = selected tokens including the incomplete tail,
# H = query or KV heads, K = head dimension, V = value dimension,
# D = model dimension, Q = query chunk, C = projection channels.


class QSAIndexer(Module):
    """Score average-pooled complete blocks; always keep the incomplete tail."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_heads: int
        head_dim: int
        rotary_dim: int
        token_budget: int
        compress_ratio: int
        index_qk_proj: Linear.Config
        q_layernorm: OffsetRMSNorm.Config
        k_layernorm: OffsetRMSNorm.Config
        rope: MRoPE.Config
        chunk_size: int = 128

        def __post_init__(self):
            if (
                min(self.num_heads, self.head_dim, self.compress_ratio, self.chunk_size)
                <= 0
            ):
                raise ValueError(
                    "Indexer head counts, dimensions and chunk sizes must be positive"
                )
            if (
                self.token_budget < self.compress_ratio
                or self.token_budget % self.compress_ratio
            ):
                raise ValueError(
                    "token_budget must be a positive multiple of compress_ratio"
                )
            if not 0 < self.rotary_dim <= self.head_dim or self.rotary_dim % 2:
                raise ValueError("rotary_dim must be positive, even and <= head_dim")

    def __init__(self, config: Config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        self.compress_ratio = config.compress_ratio
        self.block_topk = config.token_budget // config.compress_ratio
        self.chunk_size = config.chunk_size
        self.index_qk_proj = config.index_qk_proj.build()
        self.q_layernorm = config.q_layernorm.build()
        self.k_layernorm = config.k_layernorm.build()
        self.rope = config.rope.build()

    def _rotate(self, x_THK: torch.Tensor, positions_T: torch.Tensor) -> torch.Tensor:
        rotary_THR = x_THK[..., : self.rotary_dim]
        rotary_THR, _ = self.rope(rotary_THR, rotary_THR, positions_T)
        return torch.cat((rotary_THR, x_THK[..., self.rotary_dim :]), -1)

    def forward(
        self,
        x_TD: torch.Tensor,
        *,
        positions: torch.Tensor,
        document_positions_T: torch.Tensor,
        padding_mask_T: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = x_TD.shape[0]
        qk_TC = self.index_qk_proj(x_TD)
        remat.recompute_needs_tensor(qk_TC)
        query_TC, raw_key_TK = qk_TC.split(
            (self.num_heads * self.head_dim, self.head_dim), -1
        )
        query_THK = self._rotate(
            self.q_layernorm(query_TC.unflatten(-1, (self.num_heads, self.head_dim))),
            positions,
        )
        indices_T = torch.arange(num_tokens, device=x_TD.device)
        starts_T = document_starts(document_positions_T)
        valid_T = (
            torch.ones_like(indices_T, dtype=torch.bool)
            if padding_mask_T is None
            else ~padding_mask_T
        )
        counts_T = valid_T.long().cumsum(0)
        start_counts_T = F.pad(counts_T[:-1], (1, 0))[starts_T]
        local_counts_T = counts_T - start_counts_T
        valid_indices_U = valid_T.nonzero().flatten()
        block_ends_G = (
            (valid_T & (local_counts_T.remainder(self.compress_ratio) == 0))
            .nonzero()
            .flatten()
        )
        block_indices_GR = valid_indices_U[
            counts_T[block_ends_G, None]
            - self.compress_ratio
            + torch.arange(self.compress_ratio, device=x_TD.device)
        ]
        pooled_GK = raw_key_TK[block_indices_GR].float().mean(1).to(raw_key_TK.dtype)
        key_GK = self._rotate(
            self.k_layernorm(pooled_GK).unsqueeze(1), positions[block_indices_GR[:, 0]]
        ).squeeze(1)
        num_selected = min(self.block_topk, block_ends_G.numel())
        tail_count_T = local_counts_T.remainder(self.compress_ratio)
        tail_offsets_TR = torch.arange(
            self.compress_ratio - 1, device=x_TD.device
        ).expand(num_tokens, -1)
        tail_slots_TR = counts_T[:, None] - tail_count_T[:, None] + tail_offsets_TR
        # The extra entry makes all-padding documents a well-defined empty selection.
        tail_indices_TR = F.pad(valid_indices_U, (0, 1))[
            tail_slots_TR.clamp_max(valid_indices_U.numel())
        ]
        tail_indices_TR = tail_indices_TR.masked_fill(
            ~(valid_T[:, None] & (tail_offsets_TR < tail_count_T[:, None])), -1
        )
        selected_indices = []
        selected_scores = []
        for start in range(0, num_tokens, self.chunk_size):
            stop = start + self.chunk_size
            scores_QHG = torch.einsum(
                "qhk,gk->qhg", query_THK[start:stop].float(), key_GK.float()
            )
            scores_QG = scores_QHG.relu().sum(1) * self.head_dim**-0.5
            visible_QG = (
                (block_ends_G[None] <= indices_T[start:stop, None])
                & (starts_T[block_ends_G][None] == starts_T[start:stop, None])
                & valid_T[start:stop, None]
            )
            scores_QB, blocks_QB = scores_QG.masked_fill(~visible_QG, -torch.inf).topk(
                num_selected, -1
            )
            tokens_QBR = block_indices_GR[blocks_QB].masked_fill(
                ~scores_QB.isfinite().unsqueeze(-1), -1
            )
            selected_indices.append(
                torch.cat(
                    (tokens_QBR.flatten(-2), tail_indices_TR[start:stop]), -1
                ).int()
            )
            selected_scores.append(scores_QB)
        return torch.cat(selected_indices), torch.cat(selected_scores)


class QSAIndexerLoss(AuxLoss):
    """KL loss on selected complete blocks from the report's sparse training stage."""

    @dataclass(kw_only=True, slots=True)
    class Config(AuxLoss.Config):
        pass

    @staticmethod
    def block_kl(scores_QB: torch.Tensor, teacher_QB: torch.Tensor) -> torch.Tensor:
        valid_QB = scores_QB.isfinite()
        log_probs_QB = F.log_softmax(
            scores_QB.masked_fill(~valid_QB, torch.finfo(torch.float32).min), -1
        )
        teacher_QB = teacher_QB.detach() * valid_QB
        teacher_QB = teacher_QB / teacher_QB.sum(-1, keepdim=True).clamp_min(1e-30)
        return (torch.xlogy(teacher_QB, teacher_QB) - teacher_QB * log_probs_QB).sum()


class QSAInnerAttention(InnerAttention):
    """Selected-token SDPA with bounded gathers and a detached attention teacher."""

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        compress_ratio: int
        num_heads: int
        indexer_loss: QSAIndexerLoss.Config
        chunk_size: int = 16

        @property
        def is_full_attention(self) -> bool:
            return True

        def build_attention_metadata(
            self, positions, **kwargs
        ) -> VarlenAttentionMetadata:
            return create_varlen_metadata_for_document(positions, **kwargs)

    def __init__(self, config: Config):
        super().__init__()
        self.compress_ratio = config.compress_ratio
        self.num_heads = config.num_heads
        self.chunk_size = config.chunk_size
        self.indexer_loss = config.indexer_loss.build()

    def forward(
        self,
        query_THK: torch.Tensor,
        key_THK: torch.Tensor,
        value_THV: torch.Tensor,
        *,
        selected_indices_TJ: torch.Tensor,
        scores_TB: torch.Tensor,
        scale: float,
        enable_gqa: bool,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        train_indexer = self.training and self.indexer_loss.coeff != 0
        if train_indexer and aux_loss_denominator is None:
            raise ValueError(
                "QSA training requires the step-global auxiliary-loss denominator"
            )
        num_blocks = scores_TB.shape[-1]
        outputs = []
        raw_sum = scores_TB.new_zeros(())
        for start in range(0, query_THK.shape[0], self.chunk_size):
            stop = start + self.chunk_size
            indices_QJ = selected_indices_TJ[start:stop].long()
            valid_QJ = indices_QJ >= 0
            query_QH1K = query_THK[start:stop].unsqueeze(2)
            keys_QHJK = key_THK[indices_QJ.clamp_min(0)].transpose(1, 2)
            values_QHJV = value_THV[indices_QJ.clamp_min(0)].transpose(1, 2)
            with sdpa_kernel(SDPBackend.MATH):
                output_QHV = F.scaled_dot_product_attention(
                    query_QH1K,
                    keys_QHJK,
                    values_QHJV,
                    attn_mask=valid_QJ[:, None, None],
                    scale=scale,
                    enable_gqa=enable_gqa,
                ).squeeze(2)
            outputs.append(output_QHV)
            if train_indexer and num_blocks:
                with torch.no_grad():
                    keys_QHJK = keys_QHJK.repeat_interleave(
                        query_QH1K.shape[1] // keys_QHJK.shape[1], 1
                    )
                    logits_QHJ = (query_QH1K.float() * keys_QHJK.float()).sum(
                        -1
                    ) * scale
                    logits_QHJ = logits_QHJ.masked_fill(
                        ~valid_QJ[:, None], torch.finfo(torch.float32).min
                    )
                    teacher_QJ = logits_QHJ.softmax(-1).sum(1) / self.num_heads
                    teacher_QJ = teacher_QJ * valid_QJ
                    tp_group = spmd_mesh_group("tp")
                    if tp_group is not None:
                        teacher_QJ = wait_tensor(
                            all_reduce(teacher_QJ, "sum", tp_group)
                        )
                    # Sum heads before max pooling; pooling each TP shard first
                    # changes the target distribution.
                    teacher_QB = (
                        teacher_QJ[:, : num_blocks * self.compress_ratio]
                        .unflatten(-1, (num_blocks, self.compress_ratio))
                        .amax(-1)
                    )
                raw_sum = raw_sum + self.indexer_loss.block_kl(
                    scores_TB[start:stop], teacher_QB
                )
        output_THV = torch.cat(outputs)
        if train_indexer:
            # Indexer parameters and scores are TP-replicated. The FSDP bridge
            # sums their local gradients, so contribute the global KL once.
            raw_sum = raw_sum.detach() + (raw_sum - raw_sum.detach()) / spmd_mesh_size(
                "tp"
            )
            output_THV = self.indexer_loss.inject(
                raw_sum,
                carrier=output_THV,
                denominator=aux_loss_denominator,
            )
        return output_THV


class QSAAttention(Qwen35Attention):
    """Qwen3.5's gated QK-normalized attention with the Flash-Next indexer."""

    @dataclass(kw_only=True, slots=True)
    class Config(Qwen35Attention.Config):
        indexer: QSAIndexer.Config

    def __init__(self, config: Config):
        super().__init__(config)
        self.indexer = config.indexer.build()

    def forward(  # pyrefly: ignore [bad-override]
        self,
        x_TD: torch.Tensor,
        attention_metadata: VarlenAttentionMetadata | None,
        positions: torch.Tensor | None = None,
        *,
        document_positions_T: torch.Tensor,
        padding_mask_T: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_metadata
        if positions is None:
            raise ValueError("QSA requires explicit rotary positions")
        x_TD = maybe_gather_tp_input(self, x_TD)
        selected_indices_TJ, scores_TB = self.indexer(
            x_TD,
            positions=positions,
            document_positions_T=document_positions_T,
            padding_mask_T=padding_mask_T,
        )
        query_gate_THC = local_head_split(
            self.wq(x_TD), self.head_dim * 2, cp_shard_dim=0
        )
        query_THK, gate_THV = query_gate_THC.chunk(2, -1)
        key_THK = local_head_split(self.wk(x_TD), self.head_dim, cp_shard_dim=0)
        value_THV = local_head_split(self.wv(x_TD), self.head_dim, cp_shard_dim=0)
        remat.recompute_needs_tensor(query_THK, key_THK)
        query_THK, key_THK = self._partial_rope(
            self.q_norm(query_THK), self.k_norm(key_THK), positions
        )
        output_THV = remat.region(
            self.inner_attention,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(
            query_THK,
            key_THK,
            value_THV,
            selected_indices_TJ=selected_indices_TJ,
            scores_TB=scores_TB,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
            aux_loss_denominator=aux_loss_denominator,
        )
        remat.recompute_needs_tensor(output_THV, gate_THV)
        output_TD = (output_THV * gate_THV.sigmoid()).flatten(-2)
        return self.wo(output_TD)
