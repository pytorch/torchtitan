# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flash-Next paged QSA generation: vLLM-owned pages, metadata, and the indexer and
attention modules that read them."""

from dataclasses import dataclass

import torch
from vllm.config import get_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    MLAAttentionSpec,
)

from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode
from torchtitan.models.base.module import Module
from torchtitan.models.common.attn_gym import (
    paged_lightning_indexer,
    paged_sparse_decode,
    PagedKV,
    require_attn_gym,
)
from torchtitan.models.qwen3_8_flash_next.attention import QSAIndexer, QSAInnerAttention
from torchtitan.rl.generator.local_compile import inference_local_compile
from torchtitan.rl.generator.vllm.attention.query_positions import vllm_query_positions

# T = packed tokens, S = requests, P = pages, N = tokens or states per page,
# Q = query heads, H = KV heads (query heads in the indexer), K = head dimension,
# V = value head dimension, D = model dimension, C = projection channels,
# B = selected blocks, J = selected indices.


@dataclass
class QSAPagedMetadata(AttentionMetadata):
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    max_query_len: int
    _query_positions: tuple[
        int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] | None = None

    def query_positions(
        self, num_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return request indices, absolute positions and validity for packed rows,
        computed once per step for each row count."""
        if self._query_positions is None or self._query_positions[0] != num_tokens:
            self._query_positions = (
                num_tokens,
                vllm_query_positions(self.query_start_loc, self.seq_lens, num_tokens),
            )
        return self._query_positions[1]


@inference_local_compile(batch_invariant=True)
def _zero_padding_rows(output_TQK, query_start_loc):
    valid_T = (
        torch.arange(output_TQK.shape[0], device=output_TQK.device)
        < query_start_loc[-1]
    )
    return torch.where(valid_T[:, None, None], output_TQK, 0)


class QSAPagedMetadataBuilder(AttentionMetadataBuilder[QSAPagedMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    reorder_batch_threshold = 1

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.max_num_pages = kv_cache_spec.max_num_blocks_per_req(
            vllm_config, vllm_config.model_config.max_model_len
        )

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = common_attn_metadata
        # vLLM aligns table widths for every backend. Unused columns would
        # enlarge the indexer's scoring workspace and launch grid.
        return QSAPagedMetadata(
            block_table=metadata.block_table_tensor[
                : metadata.num_reqs, : self.max_num_pages
            ],
            slot_mapping=metadata.slot_mapping,
            query_start_loc=metadata.query_start_loc[: metadata.num_reqs + 1],
            seq_lens=metadata.seq_lens[: metadata.num_reqs],
            max_query_len=metadata.max_query_len,
        )


class QSAPagedBackend(AttentionBackend):
    @staticmethod
    def get_name():
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        # Computation is owned by the TorchTitan modules using PagedCache.
        raise NotImplementedError

    @staticmethod
    def get_builder_cls():
        return QSAPagedMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [MultipleOf(16)]


class PagedCache(Module, AttentionLayerBase):
    """Store rank-local tensors in pages allocated and reclaimed by vLLM."""

    supports_dcp = False

    def __init__(self, *, prefix: str, num_kv_heads: int, head_dim: int):
        super().__init__()
        # TODO: bump the attn-gym pin once paged_sparse_decode's causal_topk lands:
        # https://github.com/meta-pytorch/attention-gym/pull/979
        require_attn_gym(paged_sparse_decode, PagedKV)
        config = get_current_vllm_config()
        if config.model_config.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("Flash-Next paged caches require bf16 or fp16")
        if config.cache_config.cache_dtype != "auto":
            raise ValueError("Flash-Next generation requires an unquantized KV cache")
        if (
            not config.model_config.enforce_eager
            and config.compilation_config.cudagraph_mode
            not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY)
        ):
            raise ValueError(
                "Flash-Next generation supports NONE or FULL_DECODE_ONLY CUDA graphs"
            )
        if config.speculative_config is not None:
            raise ValueError(
                "Flash-Next generation does not support speculative decoding"
            )
        if config.parallel_config.decode_context_parallel_size != 1:
            raise ValueError("Flash-Next generation does not support decode CP")
        self.prefix = prefix
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = config.model_config.dtype
        self.kv_cache = torch.empty(0)
        self._unit_scale: torch.Tensor | None = None
        context = config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate Flash-Next cache name: {prefix}")
        context[prefix] = self

    def get_attn_backend(self):
        return QSAPagedBackend

    def get_kv_cache_spec(self, vllm_config):
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            dtype=self.dtype,
        )

    def metadata(self) -> QSAPagedMetadata | None:
        metadata = get_forward_context().attn_metadata
        return None if metadata is None else metadata[self.prefix]

    def views(self):
        return self.kv_cache.transpose(1, 2).split(self.head_dim, dim=-1)

    def write(self, key_THK, value_THK, metadata: QSAPagedMetadata):
        key_PNHK, value_PNHK = self.views()
        # The unquantized writer ignores these scales; no persistent model state.
        if self._unit_scale is None or self._unit_scale.device != key_THK.device:
            self._unit_scale = key_THK.new_ones((), dtype=torch.float32)
        scale = self._unit_scale
        reshape_and_cache_flash(
            key_THK.contiguous(),
            value_THK.contiguous(),
            key_PNHK,
            value_PNHK,
            metadata.slot_mapping,
            "auto",
            scale,
            scale,
        )

    def attend(self, query_TQK, key_THK, value_THK, blocks_TB, *, scale):
        metadata = self.metadata()
        if metadata is None or metadata.seq_lens.numel() == 0:
            return torch.zeros_like(query_TQK)
        self.write(key_THK, value_THK, metadata)
        key_PNHK, value_PNHK = self.views()
        packed = metadata.max_query_len > 1
        output_TQK = paged_sparse_decode(
            query_TQK,
            PagedKV(key_PNHK, value_PNHK, metadata.block_table),
            blocks_TB,
            metadata.query_start_loc,
            metadata.seq_lens,
            max_seqlen_q=None if packed else 1,
            # QSAAttention.Config in qwen3_8_flash_next/attention.py requires
            # compress_ratio=4 for both the indexer and inner attention.
            index_block_size=4,
            window_size=0,
            scale=scale,
            causal_topk=packed,
        )
        return _zero_padding_rows(output_TQK, metadata.query_start_loc)


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


class _QSARawKeyRing(PagedCache):
    """One key-only state ring per request, with at least 16 slots, managed by vLLM."""

    def get_kv_cache_spec(self, vllm_config):
        return CircularBufferSpec(
            block_size=16,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=0,
            dtype=self.dtype,
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

    @inference_local_compile(batch_invariant=False)
    def _decode_prep(
        self,
        qk_TC,
        positions,
        request_T,
        position_T,
        valid_T,
        query_start_loc,
        seq_lens,
        candidate_table,
        raw_table,
        raw_cache_PNHK,
        candidate_page_size: int,
    ):
        num_tokens = qk_TC.shape[0]
        query_TC, raw_key_TK = qk_TC.split(
            (self.num_heads * self.head_dim, self.head_dim), -1
        )
        query_THK = self._rotate(
            self.q_layernorm(query_TC.unflatten(-1, (self.num_heads, self.head_dim))),
            positions,
        )
        rows_T = torch.arange(num_tokens, device=qk_TC.device)
        starts_T = query_start_loc[request_T]
        ring_capacity = raw_cache_PNHK.shape[1]
        ring_pages_T = raw_table[request_T, 0].long()
        pooled_TK = (
            torch.stack(
                [
                    torch.where(
                        (rows_T - shift >= starts_T)[:, None],
                        raw_key_TK[(rows_T - shift).clamp_min(0)],
                        raw_cache_PNHK[
                            ring_pages_T, (position_T - shift) % ring_capacity
                        ][:, 0],
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
        candidate_T = position_T // self.compress_ratio
        skip_candidate_T = ~valid_T | ((position_T + 1) % self.compress_ratio != 0)
        candidate_pages_T = (
            candidate_table[request_T, candidate_T // candidate_page_size]
            .long()
            .masked_fill(skip_candidate_T, 0)
        )
        candidate_offsets_T = (candidate_T % candidate_page_size).masked_fill(
            skip_candidate_T, 0
        )
        # Only the unfinished block survives the step. Completed blocks are
        # represented solely by their transformed means in the compressed cache.
        skip_ring_T = ~valid_T | (
            position_T
            < seq_lens[request_T] // self.compress_ratio * self.compress_ratio
        )
        return (
            query_THK,
            raw_key_TK,
            pooled_TK,
            candidate_pages_T,
            candidate_offsets_T,
            ring_pages_T.masked_fill(skip_ring_T, 0),
            (position_T % ring_capacity).masked_fill(skip_ring_T, 0),
        )

    def forward(self, x_TD, *, positions, metadata=None):
        step_metadata = self.cache.metadata()
        if step_metadata is None or step_metadata.seq_lens.numel() == 0:
            blocks_TB = torch.full(
                (x_TD.shape[0], self.block_topk),
                -1,
                device=x_TD.device,
                dtype=torch.int32,
            )
            return blocks_TB, x_TD.new_empty((x_TD.shape[0], 0))
        num_tokens = x_TD.shape[0]
        qk_TC = self.index_qk_proj(x_TD)
        raw_metadata = self.raw_cache.metadata()
        (raw_cache_PNHK,) = self.raw_cache.views()
        (pooled_cache_PNHK,) = self.cache.views()
        request_T, position_T, valid_T = step_metadata.query_positions(num_tokens)
        (
            query_THK,
            raw_key_TK,
            pooled_TK,
            candidate_pages_T,
            candidate_offsets_T,
            ring_pages_T,
            ring_offsets_T,
        ) = self._decode_prep(
            qk_TC,
            positions,
            request_T,
            position_T,
            valid_T,
            step_metadata.query_start_loc,
            step_metadata.seq_lens,
            step_metadata.block_table,
            raw_metadata.block_table,
            raw_cache_PNHK,
            pooled_cache_PNHK.shape[1],
        )
        pooled_cache_PNHK[candidate_pages_T, candidate_offsets_T] = pooled_TK[:, None]
        raw_cache_PNHK[ring_pages_T, ring_offsets_T] = raw_key_TK[:, None]
        blocks_TB = paged_lightning_indexer(
            query_THK,
            pooled_cache_PNHK[:, :, 0],
            query_THK.new_ones((num_tokens, self.num_heads)),
            step_metadata.block_table,
            step_metadata.query_start_loc,
            step_metadata.seq_lens,
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
