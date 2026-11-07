# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen Sparse Attention with complete-block selection and indexer distillation."""

from dataclasses import dataclass
from typing import cast, NamedTuple

import spmd_types as spmd

import torch
import torch.nn.functional as F
import torch_remat as remat
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.checkpoint import checkpoint

from torchtitan.models.base.module import Module
from torchtitan.models.common import Linear
from torchtitan.models.common.attention import (
    AttentionMetadata,
    InnerAttention,
    local_head_split,
)
from torchtitan.models.common.aux_loss import AuxLoss

from torchtitan.models.common.decoder_sharding import token_id_placement
from torchtitan.models.common.linear import maybe_gather_tp_input
from torchtitan.models.qwen3_5.model import OffsetRMSNorm, Qwen35Attention

from .embedding import document_starts
from .rope import MRoPE

# T = packed tokens, U = valid tokens, G = complete blocks, R = block size,
# B = selected blocks, J = selected tokens including the incomplete tail,
# H = query or KV heads, K = head dimension, V = value dimension,
# D = model dimension, Q = query chunk, C = projection channels.


def _select_blocks(
    query_QHK,
    key_GK,
    block_ends_G,
    block_starts_G,
    indices_Q,
    starts_Q,
    valid_Q,
    *,
    num_selected,
    scale,
):
    scores_QHG = torch.einsum("qhk,gk->qhg", query_QHK.float(), key_GK.float())
    scores_QG = scores_QHG.relu().sum(1) * scale
    visible_QG = (
        (block_ends_G[None] <= indices_Q[:, None])
        & (block_starts_G[None] == starts_Q[:, None])
        & valid_Q[:, None]
    )
    ranked_scores_QG = scores_QG.masked_fill(~visible_QG, -torch.inf)
    # ReLU creates exact ties; chronological ordering preserves causality.
    with torch.no_grad():
        blocks_QB = ranked_scores_QG.argsort(dim=-1, descending=True, stable=True)[
            :, :num_selected
        ]
    return blocks_QB, ranked_scores_QG.gather(-1, blocks_QB)


def _selected_attention(
    query_QHK, key_THK, value_THV, indices_QJ, *, scale, enable_gqa
):
    valid_QJ = indices_QJ >= 0
    keys_QHJK = key_THK[indices_QJ.clamp_min(0)].transpose(1, 2)
    values_QHJV = value_THV[indices_QJ.clamp_min(0)].transpose(1, 2)
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(
            query_QHK.unsqueeze(2),
            keys_QHJK,
            values_QHJV,
            attn_mask=valid_QJ[:, None, None],
            scale=scale,
            enable_gqa=enable_gqa,
        ).squeeze(2)


class QSAIndexMetadata(NamedTuple):
    """Document-local complete blocks and tails shared by all QSA layers."""

    starts_T: torch.Tensor  # noqa: N815
    valid_T: torch.Tensor  # noqa: N815
    block_indices_GR: torch.Tensor  # noqa: N815
    tail_indices_TR: torch.Tensor  # noqa: N815

    @classmethod
    def from_inputs(cls, document_positions_T, *, padding_mask_T, compress_ratio):
        num_tokens = document_positions_T.numel()
        indices_T = torch.arange(num_tokens, device=document_positions_T.device)
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
            (valid_T & (local_counts_T.remainder(compress_ratio) == 0))
            .nonzero()
            .flatten()
        )
        block_indices_GR = valid_indices_U[
            counts_T[block_ends_G, None]
            - compress_ratio
            + torch.arange(compress_ratio, device=document_positions_T.device)
        ]
        tail_count_T = local_counts_T.remainder(compress_ratio)
        tail_offsets_TR = torch.arange(
            compress_ratio - 1, device=document_positions_T.device
        ).expand(num_tokens, -1)
        tail_slots_TR = counts_T[:, None] - tail_count_T[:, None] + tail_offsets_TR
        # The extra entry makes all-padding documents a well-defined empty selection.
        tail_indices_TR = F.pad(valid_indices_U, (0, 1))[
            tail_slots_TR.clamp_max(valid_indices_U.numel())
        ]
        tail_indices_TR = tail_indices_TR.masked_fill(
            ~(valid_T[:, None] & (tail_offsets_TR < tail_count_T[:, None])), -1
        )
        return cls(starts_T, valid_T, block_indices_GR, tail_indices_TR)

    def annotate_spmd_types(self) -> None:
        types_T = token_id_placement()
        types_TR = spmd.SpmdType(
            {"dp": spmd.V, "cp": spmd.V, "tp": spmd.R},
            spmd.PartitionSpec(("dp", "cp"), None),
        )
        spmd.assert_type(self.starts_T, types_T)
        spmd.assert_type(self.valid_T, types_T)
        spmd.assert_type(self.block_indices_GR, types_TR)
        spmd.assert_type(self.tail_indices_TR, types_TR)


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
        metadata: QSAIndexMetadata | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = x_TD.shape[0]
        indices_T = torch.arange(num_tokens, device=x_TD.device)
        qk_TC = self.index_qk_proj(x_TD)
        remat.recompute_needs_tensor(qk_TC)
        query_TC, raw_key_TK = qk_TC.split(
            (self.num_heads * self.head_dim, self.head_dim), -1
        )
        query_THK = self._rotate(
            self.q_layernorm(query_TC.unflatten(-1, (self.num_heads, self.head_dim))),
            positions,
        )
        if metadata is None:
            metadata = QSAIndexMetadata.from_inputs(
                document_positions_T,
                padding_mask_T=padding_mask_T,
                compress_ratio=self.compress_ratio,
            )
        starts_T, valid_T, block_indices_GR, tail_indices_TR = metadata
        block_ends_G = block_indices_GR[:, -1]
        block_starts_G = starts_T[block_ends_G]
        pooled_GK = raw_key_TK[block_indices_GR].float().mean(1).to(raw_key_TK.dtype)
        key_GK = self._rotate(
            self.k_layernorm(pooled_GK).unsqueeze(1), positions[block_indices_GR[:, 0]]
        ).squeeze(1)
        num_selected = min(self.block_topk, block_ends_G.numel())
        selected_indices = []
        selected_scores = []
        for start in range(0, num_tokens, self.chunk_size):
            stop = start + self.chunk_size
            # Checkpoint the dense score graph, including the visibility mask.
            # Saving a precomputed Q x G mask would retain quadratic storage.
            blocks_QB, scores_QB = checkpoint(
                _select_blocks,
                query_THK[start:stop],
                key_GK,
                block_ends_G,
                block_starts_G,
                indices_T[start:stop],
                starts_T[start:stop],
                valid_T[start:stop],
                num_selected=num_selected,
                scale=self.head_dim**-0.5,
                use_reentrant=False,
                preserve_rng_state=False,
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
    """Recompute selected-token gathers and SDPA per query chunk for backward."""

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        compress_ratio: int
        indexer_loss: QSAIndexerLoss.Config | None = None
        chunk_size: int = 16

        @property
        def is_full_attention(self) -> bool:
            return True

        def build_attention_metadata(
            self, positions, *, padding_mask=None, **kwargs
        ) -> AttentionMetadata:
            # Decoder dispatches this metadata only to QSA by inner-attention type.
            return cast(
                AttentionMetadata,
                QSAIndexMetadata.from_inputs(
                    positions,
                    padding_mask_T=padding_mask,
                    compress_ratio=self.compress_ratio,
                ),
            )

    def __init__(self, config: Config):
        super().__init__()
        self.compress_ratio = config.compress_ratio
        self.chunk_size = config.chunk_size
        self.indexer_loss = (
            config.indexer_loss.build() if config.indexer_loss is not None else None
        )

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
        train_indexer = self.training and self.indexer_loss is not None
        if train_indexer and aux_loss_denominator is None:
            raise ValueError(
                "QSA training requires the step-global auxiliary-loss denominator"
            )
        num_blocks = scores_TB.shape[-1]
        outputs = []
        # Empty ranks must participate in the indexer's FSDP gradient reduction.
        # An empty slice is finite even when all selected scores are -inf.
        raw_sum = scores_TB[..., :0].sum()
        for start in range(0, query_THK.shape[0], self.chunk_size):
            stop = start + self.chunk_size
            indices_QJ = selected_indices_TJ[start:stop]
            # TODO: use Attention Gym's distinct-K/V grouped backward when
            # available: https://github.com/meta-pytorch/attention-gym/pull/679.
            # Native checkpoints can nest inside a block torch_remat checkpoint.
            # TODO: use torch_remat after nesting and chunk early-stop are supported.
            # Early-stop tracking: https://github.com/meta-pytorch/remat/issues/6.
            output_QHV = checkpoint(
                _selected_attention,
                query_THK[start:stop],
                key_THK,
                value_THV,
                indices_QJ,
                scale=scale,
                enable_gqa=enable_gqa,
                use_reentrant=False,
                preserve_rng_state=False,
            )
            outputs.append(output_QHV)
            if train_indexer and num_blocks:
                assert self.indexer_loss is not None
                with torch.no_grad():
                    valid_QJ = indices_QJ >= 0
                    query_QH1K = query_THK[start:stop].unsqueeze(2)
                    keys_QHJK = key_THK[indices_QJ.clamp_min(0)].transpose(1, 2)
                    keys_QHJK = keys_QHJK.repeat_interleave(
                        query_QH1K.shape[1] // keys_QHJK.shape[1], 1
                    )
                    logits_QHJ = (query_QH1K.float() * keys_QHJK.float()).sum(
                        -1
                    ) * scale
                    logits_QHJ = logits_QHJ.masked_fill(
                        ~valid_QJ[:, None], torch.finfo(torch.float32).min
                    )
                    teacher_QJ = logits_QHJ.softmax(-1).sum(1) / query_QH1K.shape[1]
                    teacher_QJ = teacher_QJ * valid_QJ
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
            assert self.indexer_loss is not None
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
        inner_attention: QSAInnerAttention.Config  # pyrefly: ignore [bad-override]
        indexer: QSAIndexer.Config

        def __post_init__(self):
            Qwen35Attention.Config.__post_init__(self)
            # Decoder caches one metadata entry per inner-attention class.
            # The released block size is shared by every Flash-Next QSA layer.
            if (
                self.indexer.compress_ratio != 4
                or self.inner_attention.compress_ratio != 4
            ):
                raise ValueError("Flash-Next QSA requires compress_ratio=4")

        def flops_per_token(self, seq_len: int) -> int:
            from torchtitan.models.common.flops import (
                quadratic_attention_flops_per_token,
            )

            attended_tokens = min(
                seq_len, self.indexer.token_budget + self.indexer.compress_ratio - 1
            )
            return quadratic_attention_flops_per_token(
                num_heads=self.n_heads,
                qk_head_dim=self.head_dim,
                v_head_dim=self.head_dim,
                seq_len=attended_tokens,
            ) + 6 * self.indexer.num_heads * self.indexer.head_dim * (
                seq_len // self.indexer.compress_ratio
            )

    def __init__(self, config: Config):
        super().__init__(config)
        self.indexer = config.indexer.build()
        if config.inner_attention.indexer_loss is None:
            self.indexer.requires_grad_(False)

    def forward(  # pyrefly: ignore [bad-override]
        self,
        x_TD: torch.Tensor,
        attention_metadata: QSAIndexMetadata | None,
        positions: torch.Tensor | None = None,
        *,
        document_positions_T: torch.Tensor,
        padding_mask_T: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if positions is None:
            raise ValueError("QSA requires explicit rotary positions")
        x_TD = maybe_gather_tp_input(self, x_TD)
        with torch.set_grad_enabled(
            torch.is_grad_enabled()
            and self.training
            and self.inner_attention.indexer_loss is not None
        ):
            selected_indices_TJ, scores_TB = self.indexer(
                # Only the explicit KL objective updates the indexer.
                x_TD.detach(),
                positions=positions,
                document_positions_T=document_positions_T,
                padding_mask_T=padding_mask_T,
                metadata=attention_metadata,
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
