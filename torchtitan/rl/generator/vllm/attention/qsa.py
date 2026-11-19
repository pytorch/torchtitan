# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM-owned pages for Flash-Next sparse attention and its request history."""

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
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import CircularBufferSpec, FullAttentionSpec

from torchtitan.distributed.local_compile import local_compile
from torchtitan.models.common.attn_gym import gather_attn, paged_sparse_decode, PagedKV

# T = packed tokens, S = requests, P = pages, N = tokens per page,
# Q = query heads, H = KV heads, K = head dimension, B = selected blocks.


@dataclass
class QSAPagedMetadata(AttentionMetadata):
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    max_query_len: int
    num_decodes: int = 0
    num_decode_tokens: int = 0
    num_prefill_context_tokens: int = 0
    _coordinates: tuple[
        int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] | None = None

    def token_coordinates(self, num_tokens: int):
        """Return request indices, absolute positions and validity for packed rows."""
        return token_coordinates(self.query_start_loc, self.seq_lens, num_tokens)


@local_compile("qsa_token_coordinates", batch_invariant=True)
def token_coordinates(
    query_start_loc: torch.Tensor, seq_lens: torch.Tensor, num_tokens: int
):
    rows_T = torch.arange(num_tokens, device=query_start_loc.device)
    request_T = torch.searchsorted(query_start_loc[1:], rows_T, right=True).clamp_max(
        seq_lens.numel() - 1
    )
    query_lens_S = query_start_loc.diff()
    positions_T = (
        seq_lens[request_T]
        - query_lens_S[request_T]
        + rows_T
        - query_start_loc[request_T]
    )
    valid_T = rows_T < query_start_loc[-1]
    return request_T, positions_T.masked_fill(~valid_T, 0), valid_T


def step_coordinates(metadata: QSAPagedMetadata, num_tokens: int):
    cached = metadata._coordinates
    if cached is None or cached[0] != num_tokens:
        cached = (num_tokens, metadata.token_coordinates(num_tokens))
        metadata._coordinates = cached
    return cached[1]


@local_compile("qsa_prefill_plan", batch_invariant=True)
def _prefill_plan(
    key_PNHK,
    value_PNHK,
    block_table,
    query_start_S,
    context_S,
    blocks_TB,
    position_T,
    num_context: int,
    block_size: int,
):
    cu_seqlens_q = (query_start_S - query_start_S[0]).int()
    cu_seqlens_k = torch.nn.functional.pad(context_S.cumsum(0), (1, 0))
    rows_C = torch.arange(num_context, device=context_S.device)
    request_C = torch.searchsorted(cu_seqlens_k[1:], rows_C, right=True)
    position_C = rows_C - cu_seqlens_k[request_C]
    page_size = key_PNHK.shape[1]
    pages_C = block_table[request_C, position_C // page_size].long()
    offsets_C = position_C % page_size
    blocks_TB = blocks_TB.long()
    tokens_TJ = (
        (
            blocks_TB[..., None] * block_size
            + torch.arange(block_size, device=blocks_TB.device)
        )
        .masked_fill((blocks_TB < 0)[..., None], -1)
        .flatten(1)
    )
    tail_TR = ((position_T + 1) // block_size * block_size)[:, None] + torch.arange(
        block_size - 1, device=blocks_TB.device
    )
    tail_TR = tail_TR.masked_fill(tail_TR > position_T[:, None], -1)
    return (
        key_PNHK[pages_C, offsets_C],
        value_PNHK[pages_C, offsets_C],
        torch.cat((tokens_TJ, tail_TR), -1).int(),
        cu_seqlens_q,
        cu_seqlens_k.int(),
    )


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
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            metadata, decode_threshold=1
        )
        num_prefill_context_tokens = 0
        if num_prefills:
            seq_lens_cpu = metadata.seq_lens_cpu_upper_bound
            if seq_lens_cpu is None:
                seq_lens_cpu = metadata.seq_lens[: metadata.num_reqs].cpu()
            num_prefill_context_tokens = int(
                seq_lens_cpu[num_decodes : metadata.num_reqs].sum()
            )
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
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefill_context_tokens=num_prefill_context_tokens,
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


class PagedCache(torch.nn.Module, AttentionLayerBase):
    """Store rank-local tensors in pages allocated and reclaimed by vLLM."""

    supports_dcp = False

    def __init__(self, *, prefix: str, num_kv_heads: int, head_dim: int):
        super().__init__()
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
        context = config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate Flash-Next cache name: {prefix}")
        context[prefix] = self

    def get_attn_backend(self):
        return QSAPagedBackend

    def get_kv_cache_spec(self, vllm_config):
        kwargs = dict(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            dtype=self.dtype,
        )
        return FullAttentionSpec(**kwargs)

    def metadata(self) -> QSAPagedMetadata | None:
        metadata = get_forward_context().attn_metadata
        return None if metadata is None else metadata[self.prefix]

    def views(self):
        return self.kv_cache.transpose(1, 2).split(self.head_dim, dim=-1)

    def write(self, key_THK, value_THK, metadata: QSAPagedMetadata):
        key_PNHK, value_PNHK = self.views()
        # The unquantized writer ignores these scales; no persistent model state.
        scale = key_THK.new_ones((), dtype=torch.float32)
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
        if metadata.num_prefill_context_tokens == 0:
            output_TQK = paged_sparse_decode(
                query_TQK,
                PagedKV(key_PNHK, value_PNHK, metadata.block_table),
                blocks_TB,
                metadata.query_start_loc,
                metadata.seq_lens,
                max_seqlen_q=metadata.max_query_len,
                # QSAAttention.Config in qwen3_8_flash_next/attention.py requires
                # compress_ratio=4 for both the indexer and inner attention.
                index_block_size=4,
                window_size=0,
                scale=scale,
            )
        else:
            output_TQK = self._attend_mixed(
                query_TQK, key_PNHK, value_PNHK, blocks_TB, metadata, scale=scale
            )
        valid_T = (
            torch.arange(query_TQK.shape[0], device=query_TQK.device)
            < metadata.query_start_loc[-1]
        )
        return output_TQK.masked_fill(~valid_T[:, None, None], 0)

    def _attend_mixed(
        self, query_TQK, key_PNHK, value_PNHK, blocks_TB, metadata, *, scale
    ):
        num_decodes = metadata.num_decodes
        num_decode_tokens = metadata.num_decode_tokens
        block_size = 4
        output_TQK = torch.empty_like(query_TQK)
        if num_decodes:
            output_TQK[:num_decode_tokens] = paged_sparse_decode(
                query_TQK[:num_decode_tokens],
                PagedKV(key_PNHK, value_PNHK, metadata.block_table[:num_decodes]),
                blocks_TB[:num_decode_tokens],
                metadata.query_start_loc[: num_decodes + 1],
                metadata.seq_lens[:num_decodes],
                max_seqlen_q=1,
                index_block_size=block_size,
                window_size=0,
                scale=scale,
            )
        _, position_T, _ = step_coordinates(metadata, query_TQK.shape[0])
        key_CHK, value_CHK, indices_TJ, cu_seqlens_q, cu_seqlens_k = _prefill_plan(
            key_PNHK,
            value_PNHK,
            metadata.block_table[num_decodes:],
            metadata.query_start_loc[num_decodes:],
            metadata.seq_lens[num_decodes:],
            blocks_TB[num_decode_tokens:],
            position_T[num_decode_tokens:],
            metadata.num_prefill_context_tokens,
            block_size,
        )
        prefill_1QTK = gather_attn(
            query_TQK[num_decode_tokens:].transpose(0, 1).unsqueeze(0),
            None,
            key_CHK.transpose(0, 1).unsqueeze(0),
            indices_TJ.unsqueeze(0),
            sliding_window_size=0,
            cu_seqlens=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            scale=scale,
            impl="fused",
            sparse_v=value_CHK.transpose(0, 1).unsqueeze(0),
        )
        output_TQK[num_decode_tokens:] = prefill_1QTK.squeeze(0).transpose(0, 1)
        return output_TQK


class PagedRingCache(PagedCache):
    """One state ring per request, with at least 16 slots, managed by vLLM."""

    def get_kv_cache_spec(self, vllm_config):
        return CircularBufferSpec(
            block_size=16,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            dtype=self.dtype,
        )

    def gather(self, cache_PNHK, request_T, positions_T, metadata):
        return cache_PNHK[
            metadata.block_table[request_T, 0].long(),
            positions_T % cache_PNHK.shape[1],
        ]
