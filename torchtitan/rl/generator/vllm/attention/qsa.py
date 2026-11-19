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
from vllm.v1.kv_cache_interface import FullAttentionSpec

from torchtitan.models.base.module import Module
from torchtitan.models.common.attn_gym import (
    paged_sparse_decode,
    PagedKV,
    require_attn_gym,
)

from torchtitan.rl.generator.local_compile import inference_local_compile

# T = packed tokens, S = requests, P = pages, N = tokens per page,
# Q = query heads, H = KV heads, K = head dimension, B = selected blocks.


@dataclass
class QSAPagedMetadata(AttentionMetadata):
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    max_query_len: int


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
