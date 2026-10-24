# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MiniMax Sparse Attention (MSA), https://arxiv.org/abs/2606.13392."""

# Shape suffixes:
#   T = packed tokens, D = model dimension, H = query heads,
#   G = KV heads (one index query head per GQA group), J = query heads per group,
#   K = head dimension, I = index head dimension, N = key blocks in the batch,
#   S = selected blocks per query and group, C = queries of one tile,
#   U = keys gathered for one tile, X = tiles of the flex block mask.

import math
from dataclasses import dataclass
from typing import NamedTuple

import spmd_types as spmd
import torch
import torch.nn.functional as F
import torch_remat as remat
from torch.nn.attention.flex_attention import AuxRequest, BlockMask

from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import FlexInnerAttention, GQAttention
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.norm import OffsetRMSNorm
from torchtitan.models.common.rope import RoPE

__all__ = [
    "MiniMaxSparseAttention",
    "MSAAttentionMetadata",
    "MSAIndexer",
    "MSAIndexerLoss",
    "MSAInnerAttention",
    "build_msa_metadata",
    "msa_block_mask",
    "select_msa_blocks",
]


class MSAAttentionMetadata(NamedTuple):
    """Key blocks of a packed batch, built once per batch from ``positions``.

    A block starts at every document start and every ``block_size`` tokens inside
    a document, so no block crosses a document boundary.
    """

    block_T: torch.Tensor  # noqa: N815
    block_start_N: torch.Tensor  # noqa: N815
    block_len_N: torch.Tensor  # noqa: N815
    doc_start_T: torch.Tensor  # noqa: N815
    tile_size: int
    tile_key_start: list[int]
    tile_block_range: list[tuple[int, int]]
    dense_mask: BlockMask | None


def build_msa_metadata(
    positions_T: torch.Tensor,
    *,
    block_size: int,
    tile_size: int,
    dense_mask: BlockMask | None = None,
) -> MSAAttentionMetadata:
    """Split the packed batch into MSA key blocks and per-tile key ranges."""
    num_tokens = positions_T.shape[0]
    positions_T = positions_T.long()
    token_T = torch.arange(num_tokens, device=positions_T.device)
    starts_T = positions_T % block_size == 0
    starts_T[0] = True
    block_T = torch.cumsum(starts_T, dim=0) - 1
    block_start_N = token_T[starts_T]
    block_len_N = torch.diff(block_start_N, append=token_T.new_full((1,), num_tokens))
    doc_start_T = token_T - positions_T

    tile_first_T = torch.arange(0, num_tokens, tile_size, device=positions_T.device)
    tile_last_T = (tile_first_T + tile_size).clamp(max=num_tokens) - 1
    padded_doc_start = F.pad(
        doc_start_T, (0, (-num_tokens) % tile_size), value=num_tokens
    )
    tile_key_start_X = padded_doc_start.view(-1, tile_size).amin(dim=1)
    tile_block_lo_X = block_T[tile_key_start_X]
    tile_block_hi_X = block_T[tile_last_T]
    tile_key_start, tile_block_lo, tile_block_hi = torch.stack(
        [tile_key_start_X, tile_block_lo_X, tile_block_hi_X]
    ).tolist()
    return MSAAttentionMetadata(
        block_T=block_T,
        block_start_N=block_start_N,
        block_len_N=block_len_N,
        doc_start_T=doc_start_T,
        tile_size=tile_size,
        tile_key_start=tile_key_start,
        tile_block_range=list(zip(tile_block_lo, tile_block_hi)),
        dense_mask=dense_mask,
    )


@torch.no_grad()
def select_msa_blocks(
    index_q_TGI: torch.Tensor,
    index_k_TI: torch.Tensor,
    metadata: MSAAttentionMetadata,
    *,
    topk_blocks: int,
) -> torch.Tensor:
    """Pick each query's top-k key blocks per GQA group, always keeping its own block.

    Block scores are the max over the block's causally visible keys of the index
    dot product. Returns int64 ``[T, G, S]`` global block ids, -1 for unused slots.
    """
    num_tokens, num_groups, _ = index_q_TGI.shape
    tile = metadata.tile_size
    block_indices_TGS = index_q_TGI.new_full(
        (num_tokens, num_groups, topk_blocks), -1, dtype=torch.long
    )
    token_T = torch.arange(num_tokens, device=index_q_TGI.device)
    for tile_id, q0 in enumerate(range(0, num_tokens, tile)):
        q1 = min(q0 + tile, num_tokens)
        k0 = metadata.tile_key_start[tile_id]
        block_lo, block_hi = metadata.tile_block_range[tile_id]
        num_blocks = block_hi - block_lo + 1

        scores_CGU = torch.einsum(
            "cgi,ui->cgu", index_q_TGI[q0:q1].float(), index_k_TI[k0:q1].float()
        )
        key_U = token_T[k0:q1]
        visible_CU = (key_U[None, :] <= token_T[q0:q1, None]) & (
            key_U[None, :] >= metadata.doc_start_T[q0:q1, None]
        )
        scores_CGU = scores_CGU.masked_fill(~visible_CU[:, None, :], -math.inf)

        key_block_U = metadata.block_T[k0:q1] - block_lo
        block_scores_CGN = scores_CGU.new_full(
            (q1 - q0, num_groups, num_blocks), -math.inf
        ).scatter_reduce_(
            2,
            key_block_U.expand(q1 - q0, num_groups, -1),
            scores_CGU,
            reduce="amax",
        )
        own_block_C = metadata.block_T[q0:q1] - block_lo
        block_scores_CGN.scatter_(
            2, own_block_C[:, None, None].expand(-1, num_groups, 1), math.inf
        )

        num_kept = min(topk_blocks, num_blocks)
        top_scores, top_blocks = block_scores_CGN.topk(num_kept, dim=-1)
        block_indices_TGS[q0:q1, :, :num_kept] = torch.where(
            top_scores == -math.inf, -1, top_blocks + block_lo
        )
    return block_indices_TGS


@torch.no_grad()
def msa_block_mask(
    block_indices_TGS: torch.Tensor,
    metadata: MSAAttentionMetadata,
    *,
    num_heads: int,
) -> tuple[BlockMask, torch.Tensor]:
    """Build the flex block mask that attends each query to its selected blocks.

    Returns the mask and the ``[X, G, X]`` tile occupancy it was built from.
    """
    num_tokens, num_groups, num_selected = block_indices_TGS.shape
    tile = metadata.tile_size
    num_tiles = -(-num_tokens // tile)
    heads_per_group = num_heads // num_groups

    valid_TGS = block_indices_TGS >= 0
    block_TGS = block_indices_TGS.clamp(min=0)
    block_start_TGS = metadata.block_start_N[block_TGS]
    block_end_TGS = block_start_TGS + metadata.block_len_N[block_TGS] - 1
    query_tile_TGS = (
        torch.arange(num_tokens, device=block_indices_TGS.device) // tile
    )[:, None, None].expand_as(block_TGS)
    group_TGS = torch.arange(num_groups, device=block_indices_TGS.device)[
        None, :, None
    ].expand_as(block_TGS)
    # One spare column absorbs the unused (-1) slots.
    occupied_XGX = block_indices_TGS.new_zeros(
        (num_tiles, num_groups, num_tiles + 1), dtype=torch.bool
    )
    for key_tile_TGS in (block_start_TGS // tile, block_end_TGS // tile):
        occupied_XGX.index_put_(
            (
                query_tile_TGS,
                group_TGS,
                torch.where(valid_TGS, key_tile_TGS, num_tiles),
            ),
            torch.ones_like(valid_TGS),
        )
    occupied_XGX = occupied_XGX[..., :num_tiles]

    kv_num_blocks_1HX = (
        occupied_XGX.sum(dim=-1, dtype=torch.int32)
        .transpose(0, 1)
        .repeat_interleave(heads_per_group, dim=0)[None]
    )
    # Sorting float32 0/1 keys keeps the occupied tiles first in ascending order.
    kv_indices_1HXX = (
        occupied_XGX.to(torch.float32)
        .argsort(dim=-1, descending=True, stable=True)
        .to(torch.int32)
        .transpose(0, 1)
        .repeat_interleave(heads_per_group, dim=0)[None]
    )

    # Both bounds share one captured tensor; keep the mask_mod to a single capture.
    key_range_TGS2 = torch.stack(
        (
            torch.where(valid_TGS, block_start_TGS, num_tokens),
            torch.where(valid_TGS, block_end_TGS, -1),
        ),
        dim=-1,
    ).to(torch.int32)
    last_token = num_tokens - 1

    def mask_mod(b, h, q_idx, kv_idx):
        # Flex may evaluate padded query lanes; clamp the read before applying bounds.
        q_safe = q_idx.clamp(max=last_token)
        group = h // heads_per_group

        def in_slot(slot: int) -> torch.Tensor:
            start = key_range_TGS2[q_safe, group, slot, 0]
            end = key_range_TGS2[q_safe, group, slot, 1]
            return (start <= kv_idx) & (kv_idx <= end)

        selected = in_slot(0)
        for slot in range(1, num_selected):
            selected = selected | in_slot(slot)
        return selected & (kv_idx <= q_idx) & (q_idx < num_tokens)

    block_mask = BlockMask.from_kv_blocks(
        kv_num_blocks_1HX,
        kv_indices_1HXX,
        BLOCK_SIZE=tile,
        mask_mod=mask_mod,
        seq_lengths=(num_tokens, num_tokens),
    )
    return block_mask, occupied_XGX


class MSAIndexerLoss(AuxLoss):
    """KL from the group-averaged main attention to the index-score softmax."""

    @dataclass(kw_only=True, slots=True)
    class Config(AuxLoss.Config):
        softmax_scale: float
        index_softmax_scale: float

    def __init__(self, config: Config):
        super().__init__(config)
        self.softmax_scale = config.softmax_scale
        self.index_softmax_scale = config.index_softmax_scale

    def forward(
        self,
        q_THK: torch.Tensor,
        k_TGK: torch.Tensor,
        index_q_TGI: torch.Tensor,
        index_k_TI: torch.Tensor,
        lse_TH: torch.Tensor,
        metadata: MSAAttentionMetadata,
        *,
        block_indices_TGS: torch.Tensor | None,
        occupied_XGX: torch.Tensor | None,
        carrier: torch.Tensor,
        denominator: torch.Tensor,
        padding_mask_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the KL over every query and inject its gradient on ``carrier``.

        ``block_indices_TGS`` lists the attended blocks; ``None`` means every
        causally visible key of the query's document (dense attention).
        """
        num_tokens, num_heads, _ = q_THK.shape
        num_groups = k_TGK.shape[1]
        heads_per_group = num_heads // num_groups
        tile = metadata.tile_size
        token_T = torch.arange(num_tokens, device=q_THK.device)
        key_tiles: list[list[int]] | None = None
        if occupied_XGX is not None:
            key_tiles = [[] for _ in range(occupied_XGX.shape[0])]
            for query_tile, key_tile in occupied_XGX.any(dim=1).nonzero().tolist():
                key_tiles[query_tile].append(key_tile)

        raw_sum = q_THK.new_zeros((), dtype=torch.float32)
        for tile_id, q0 in enumerate(range(0, num_tokens, tile)):
            q1 = min(q0 + tile, num_tokens)
            if key_tiles is None:
                key_U = token_T[metadata.tile_key_start[tile_id] : q1]
            else:
                tiles = torch.tensor(key_tiles[tile_id], device=q_THK.device)
                key_U = tiles[:, None] * tile + torch.arange(tile, device=q_THK.device)
                key_U = key_U.flatten()
                key_U = key_U[key_U < num_tokens]
            query_C = token_T[q0:q1]
            visible_CU = (key_U[None, :] <= query_C[:, None]) & (
                key_U[None, :] >= metadata.doc_start_T[q0:q1, None]
            )
            if block_indices_TGS is None:
                attended_CGU = visible_CU[:, None, :].expand(-1, num_groups, -1)
            else:
                attended_CGU = (
                    block_indices_TGS[q0:q1, :, :, None]
                    == metadata.block_T[key_U][None, None, None, :]
                ).any(dim=2) & visible_CU[:, None, :]

            with torch.no_grad():
                logits_CGJU = torch.einsum(
                    "cgjk,ugk->cgju",
                    q_THK[q0:q1].float().unflatten(1, (num_groups, heads_per_group)),
                    k_TGK[key_U].float(),
                )
                logits_CGJU = logits_CGJU * self.softmax_scale - lse_TH[
                    q0:q1
                ].float().view(q1 - q0, num_groups, heads_per_group, 1)
                probs_CGJU = logits_CGJU.exp().masked_fill(
                    ~attended_CGU[:, :, None, :], 0.0
                )
                target_CGU = probs_CGJU.mean(dim=2)

            index_logits_CGU = (
                torch.einsum(
                    "cgi,ui->cgu", index_q_TGI[q0:q1].float(), index_k_TI[key_U].float()
                )
                * self.index_softmax_scale
            )
            log_student_CGU = F.log_softmax(
                index_logits_CGU.masked_fill(~attended_CGU, -math.inf), dim=-1
            ).masked_fill(~attended_CGU, 0.0)
            kl_CG = (
                torch.special.xlogy(target_CGU, target_CGU)
                - target_CGU * log_student_CGU
            ).sum(dim=-1)
            if padding_mask_T is not None:
                kl_CG = kl_CG.masked_fill(padding_mask_T[q0:q1, None], 0.0)
            raw_sum = raw_sum + kl_CG.sum()
        return self.inject(
            raw_sum / num_groups, carrier=carrier, denominator=denominator
        )


class MSAInnerAttention(FlexInnerAttention):
    """Flex attention over the key blocks the MSA indexer selects for each query."""

    @dataclass(kw_only=True, slots=True)
    class Config(FlexInnerAttention.Config):
        index_block_size: int = 128
        topk_blocks: int = 16
        sparse: bool = True
        """False attends every causal key of the document; the indexer still trains."""
        aux_loss: MSAIndexerLoss.Config | None = None

        @property
        def is_full_attention(self) -> bool:
            return not self.sparse

        def __post_init__(self) -> None:
            if not isinstance(self.block_size, int):
                raise ValueError("MSA requires a square flex block size.")
            if not 0 < self.index_block_size <= self.block_size:
                raise ValueError(
                    f"index_block_size ({self.index_block_size}) must be in "
                    f"(0, block_size ({self.block_size})]."
                )

        def build_attention_metadata(  # pyrefly: ignore[bad-override]
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> MSAAttentionMetadata:
            """Build the batch's key blocks, plus the causal document mask in dense mode."""
            dense_mask = None
            if not self.sparse:
                dense_mask = FlexInnerAttention.Config.build_attention_metadata(
                    self,
                    positions,
                    padding_mask=padding_mask,
                    max_num_documents=max_num_documents,
                    max_context_length=max_context_length,
                )
            assert isinstance(self.block_size, int)
            return build_msa_metadata(
                positions,
                block_size=self.index_block_size,
                tile_size=self.block_size,
                dense_mask=dense_mask,
            )

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.topk_blocks = config.topk_blocks
        self.sparse = config.sparse
        self.aux_loss = config.aux_loss.build() if config.aux_loss is not None else None

    def forward(  # pyrefly: ignore[bad-override]
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        attention_metadata: MSAAttentionMetadata,
        index_q_TGI: torch.Tensor,
        index_k_TI: torch.Tensor,
        scale: float | None = None,
        enable_gqa: bool = False,
        padding_mask_T: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        num_heads = q_THK.shape[1]
        aux_loss = self.aux_loss if self.training else None
        with spmd.no_typecheck():
            block_indices_TGS = None
            occupied_XGX = None
            if self.sparse:
                block_indices_TGS = select_msa_blocks(
                    index_q_TGI,
                    index_k_TI,
                    attention_metadata,
                    topk_blocks=self.topk_blocks,
                )
                block_mask, occupied_XGX = msa_block_mask(
                    block_indices_TGS, attention_metadata, num_heads=num_heads
                )
            else:
                assert attention_metadata.dense_mask is not None
                block_mask = attention_metadata.dense_mask

            out_1HTV, aux = FlexInnerAttention.compiled_flex_attn(
                q_THK.transpose(0, 1).unsqueeze(0),
                k_THK.transpose(0, 1).unsqueeze(0),
                v_THV.transpose(0, 1).unsqueeze(0),
                score_mod=None,
                block_mask=block_mask,
                scale=scale,
                enable_gqa=enable_gqa,
                return_aux=AuxRequest(lse=aux_loss is not None),
                kernel_options=self.kernel_options,
            )
            out_THV = out_1HTV.squeeze(0).transpose(0, 1)
            if aux_loss is not None:
                if aux_loss_denominator is None:
                    raise RuntimeError("MSAIndexerLoss requires aux_loss_denominator.")
                out_THV = aux_loss(
                    q_THK.detach(),
                    k_THK.detach(),
                    index_q_TGI,
                    index_k_TI,
                    aux.lse.squeeze(0).transpose(0, 1).detach(),
                    attention_metadata,
                    block_indices_TGS=block_indices_TGS,
                    occupied_XGX=occupied_XGX,
                    carrier=out_THV,
                    denominator=aux_loss_denominator,
                    padding_mask_T=padding_mask_T,
                )
        if spmd.is_type_checking():
            spmd.assert_type(
                out_THV, spmd.get_local_type(q_THK), spmd.get_partition_spec(q_THK)
            )
        return out_THV


class MSAIndexer(Module):
    """Index branch of MSA: one query head per GQA group and one shared key head."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        head_dim: int
        wq: Linear.Config
        wk: Linear.Config
        q_norm: OffsetRMSNorm.Config
        k_norm: OffsetRMSNorm.Config

    def __init__(self, config: Config):
        super().__init__()
        self.head_dim = config.head_dim
        self.wq = config.wq.build()
        self.wk = config.wk.build()
        self.q_norm = config.q_norm.build()
        self.k_norm = config.k_norm.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        positions: torch.Tensor | None,
        *,
        rope: RoPE,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the normalized, rotated index queries ``[T, G, I]`` and keys ``[T, I]``."""
        index_q_TGI = self.wq(x_TD).unflatten(-1, (-1, self.head_dim))
        index_k_T1I = self.wk(x_TD).unflatten(-1, (1, self.head_dim))
        remat.recompute_needs_tensor(index_q_TGI, index_k_T1I)
        index_q_TGI = self.q_norm(index_q_TGI)
        index_k_T1I = self.k_norm(index_k_T1I)
        index_q_TGI, index_k_T1I = rope(index_q_TGI, index_k_T1I, positions)
        return index_q_TGI, index_k_T1I.squeeze(1)


class MiniMaxSparseAttention(GQAttention):
    """GQA attention whose keys are restricted to the blocks MSA's indexer selects.

    The indexer reads the detached layer input, so its KL loss trains only the
    index query and key projections and their norms.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GQAttention.Config):
        indexer: MSAIndexer.Config

    def __init__(self, config: Config):
        super().__init__(config)
        self.indexer = config.indexer.build()

    def forward(  # pyrefly: ignore[bad-override]
        self,
        x_TD: torch.Tensor,
        attention_metadata: MSAAttentionMetadata,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.q_norm is not None and self.k_norm is not None
        assert self.rope is not None
        xq_THK, xk_THK, xv_THV = self.qkv_linear(x_TD)
        remat.recompute_needs_tensor(xq_THK)
        xq_THK = self.q_norm(xq_THK)
        remat.recompute_needs_tensor(xk_THK)
        xk_THK = self.k_norm(xk_THK)
        remat.recompute_needs_tensor(xq_THK, xk_THK)
        xq_THK, xk_THK = self.rope(xq_THK, xk_THK, positions)

        index_q_TGI, index_k_TI = self.indexer(x_TD.detach(), positions, rope=self.rope)

        out_THV = remat.region(
            self.inner_attention,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(
            xq_THK,
            xk_THK,
            xv_THV,
            attention_metadata=attention_metadata,
            index_q_TGI=index_q_TGI,
            index_k_TI=index_k_TI,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
            padding_mask_T=padding_mask,
            aux_loss_denominator=aux_loss_denominator,
        )
        remat.recompute_needs_tensor(out_THV)
        out_THV = out_THV.contiguous()
        return self.wo(out_THV.view(out_THV.shape[0], -1))
