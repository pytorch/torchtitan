# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flash-Next projections with paged QSA selection and attention."""

from dataclasses import dataclass, replace

import torch
from vllm._custom_ops import concat_and_cache_mla
from vllm.config import get_current_vllm_config
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode
from torchtitan.models.base.module import Module
from torchtitan.models.common.attn_gym import paged_lightning_indexer, require_attn_gym
from torchtitan.models.qwen3_8_flash_next.attention import QSAIndexer, QSAInnerAttention
from torchtitan.rl.generator.vllm.attention.qsa import PagedCache, PagedRingCache

# T = packed tokens, D = model dimension, H = query or KV heads,
# K = head dimension, V = value head dimension, C = projection channels,
# B = selected blocks, P = pages, N = states per page,
# J = selected indices.


class _QSACandidateCache(PagedCache):
    def __init__(self, *, compress_ratio: int, **kwargs):
        super().__init__(**kwargs)
        self.compress_ratio = compress_ratio

    def get_kv_cache_spec(self, vllm_config):
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            dtype=self.dtype,
            tokens_per_state=self.compress_ratio,
        )


class _QSARawKeyRing(PagedRingCache):
    def get_kv_cache_spec(self, vllm_config):
        return replace(super().get_kv_cache_spec(vllm_config), head_size_v=0)


def _store_rows(cache_PNHK, rows_TK, slots_T):
    concat_and_cache_mla(
        rows_TK,
        rows_TK[:, :0],
        cache_PNHK[:, :, 0],
        slots_T,
        "auto",
        rows_TK.new_ones((), dtype=torch.float32),
    )


class VLLMQSAIndexer(QSAIndexer):
    @dataclass(kw_only=True, slots=True)
    class Config(QSAIndexer.Config):
        layer_idx: int

    def __init__(self, config: Config):
        super().__init__(config)
        # TODO: bump the attn-gym pin
        require_attn_gym(paged_lightning_indexer)
        self.cache = _QSACandidateCache(
            prefix=f"model.layers.{config.layer_idx}.attention.indexer.cache",
            num_kv_heads=1,
            head_dim=config.head_dim,
            compress_ratio=self.compress_ratio,
        )
        self.raw_cache = _QSARawKeyRing(
            prefix=f"model.layers.{config.layer_idx}.attention.indexer.raw_cache",
            num_kv_heads=1,
            head_dim=config.head_dim,
        )

    def forward(
        self,
        x_TD,
        *,
        positions,
        document_positions_T,
        padding_mask_T=None,
        metadata=None,
    ):
        metadata = self.cache.metadata()
        if metadata is None or metadata.seq_lens.numel() == 0:
            blocks_TB = torch.full(
                (x_TD.shape[0], self.block_topk),
                -1,
                device=x_TD.device,
                dtype=torch.int32,
            )
            return blocks_TB, x_TD.new_empty((x_TD.shape[0], 0))
        qk_TC = self.index_qk_proj(x_TD)
        query_TC, raw_key_TK = qk_TC.split(
            (self.num_heads * self.head_dim, self.head_dim), -1
        )
        query_THK = self._rotate(
            self.q_layernorm(query_TC.unflatten(-1, (self.num_heads, self.head_dim))),
            positions,
        )
        request_T, position_T, valid_T = metadata.token_coordinates(x_TD.shape[0])
        raw_metadata = self.raw_cache.metadata()
        rows_T = torch.arange(x_TD.shape[0], device=x_TD.device)
        starts_T = metadata.query_start_loc[request_T]
        (raw_cache_PNHK,) = self.raw_cache.views()
        pooled_TK = (
            torch.stack(
                [
                    torch.where(
                        (rows_T - shift >= starts_T)[:, None],
                        raw_key_TK[(rows_T - shift).clamp_min(0)],
                        self.raw_cache.gather(
                            raw_cache_PNHK, request_T, position_T - shift, raw_metadata
                        )[:, 0],
                    )
                    for shift in range(self.compress_ratio - 1, -1, -1)
                ],
                dim=1,
            )
            .float()
            .mean(1)
            .to(raw_key_TK.dtype)
        )
        pooled_TK = self._rotate(
            self.k_layernorm(pooled_TK)[:, None],
            (position_T - self.compress_ratio + 1).clamp_min(0),
        )[:, 0]
        (pooled_cache_PNHK,) = self.cache.views()
        candidate_page_size = pooled_cache_PNHK.shape[1]
        candidate_T = position_T // self.compress_ratio
        slots_T = (
            metadata.block_table[request_T, candidate_T // candidate_page_size].long()
            * candidate_page_size
            + candidate_T % candidate_page_size
        ).masked_fill(~valid_T | ((position_T + 1) % self.compress_ratio != 0), -1)
        _store_rows(pooled_cache_PNHK, pooled_TK, slots_T)
        # Only the unfinished block survives the step. Completed blocks are
        # represented solely by their transformed means in the compressed cache.
        ring_capacity = raw_cache_PNHK.shape[1]
        raw_slots_T = (
            raw_metadata.block_table[request_T, 0].long() * ring_capacity
            + position_T % ring_capacity
        ).masked_fill(
            ~valid_T
            | (
                position_T
                < metadata.seq_lens[request_T]
                // self.compress_ratio
                * self.compress_ratio
            ),
            -1,
        )
        _store_rows(raw_cache_PNHK, raw_key_TK, raw_slots_T)
        blocks_TB = paged_lightning_indexer(
            query_THK,
            pooled_cache_PNHK[:, :, 0],
            query_THK.new_ones(query_THK.shape[:2]),
            metadata.block_table,
            metadata.query_start_loc,
            metadata.seq_lens,
            topk=self.block_topk,
            compress_ratio=self.compress_ratio,
        )
        if is_in_batch_invariant_mode():
            # Match the trainer's block order before attention accumulation.
            blocks_TB = blocks_TB.masked_fill(
                blocks_TB < 0, torch.iinfo(torch.int32).max
            )
            blocks_TB = blocks_TB.sort(-1).values
            blocks_TB = blocks_TB.masked_fill(
                (blocks_TB == torch.iinfo(torch.int32).max) | ~valid_T[:, None], -1
            )
        return blocks_TB, x_TD.new_empty((x_TD.shape[0], 0))


class VLLMQSAInnerAttention(QSAInnerAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(QSAInnerAttention.Config):
        layer_idx: int
        num_kv_heads: int
        head_dim: int

    def __init__(self, config: Config):
        Module.__init__(self)
        self.indexer_loss = None
        tp_degree = get_current_vllm_config().parallel_config.tensor_parallel_size
        if config.num_kv_heads % tp_degree:
            raise ValueError("QSA KV heads must be divisible by generator TP")
        self.cache = PagedCache(
            prefix=f"model.layers.{config.layer_idx}.attention.inner_attention.cache",
            num_kv_heads=config.num_kv_heads // tp_degree,
            head_dim=config.head_dim,
        )

    def forward(
        self,
        query_THK,
        key_THK,
        value_THV,
        *,
        selected_indices_TJ,
        scores_TB,
        scale,
        aux_loss_denominator=None,
    ):
        # VLLMQSAIndexer supplies logical block IDs through the shared signature.
        return self.cache.attend(
            query_THK,
            key_THK,
            value_THV,
            selected_indices_TJ,
            scale=scale,
        )
