# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Attention Gym's paged GDN metadata contract for the native vLLM runner."""

import torch
from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import (
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec


class TorchTitanGDNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "TORCHTITAN_GDN"

    @staticmethod
    # The upstream abstract method has no return annotation.
    # pyrefly: ignore[bad-override]
    def get_builder_cls() -> type["TorchTitanGDNAttentionMetadataBuilder"]:
        return TorchTitanGDNAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


class TorchTitanGDNAttentionMetadataBuilder(
    AttentionMetadataBuilder[GDNAttentionMetadata]
):
    """Stage shared request buffers before eager execution or native graph replay."""

    reorder_batch_threshold = 1

    @classmethod
    def get_cudagraph_support(
        cls, vllm_config: VllmConfig, kv_cache_spec: KVCacheSpec
    ) -> AttentionCGSupport:
        parallel = vllm_config.parallel_config
        if (
            parallel.data_parallel_size != 1
            or parallel.use_ubatching
            or parallel.enable_dbo
        ):
            return AttentionCGSupport.UNIFORM_BATCH
        return AttentionCGSupport.ALWAYS

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        # The abstract base initializer has a concrete field-initialization body.
        # pyrefly: ignore[missing-attribute]
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.kv_cache_spec = kv_cache_spec
        compilation = vllm_config.compilation_config
        if (
            compilation.cudagraph_mode.has_full_cudagraphs()
            and compilation.cudagraph_num_of_warmups < 1
        ):
            raise ValueError("GDN FULL capture requires at least one kernel warmup")
        if vllm_config.cache_config.mamba_cache_mode == "all":
            raise ValueError(
                "Attention Gym GDN does not support intermediate prefix-cache checkpoints; use 'none' or 'align'"
            )
        self.num_reqs_capacity = vllm_config.scheduler_config.max_num_seqs
        # max_num_seqs + 1 request slots need max_num_seqs + 2 offsets. The
        # extra local null request owns [real_tokens, padded_capacity), even
        # with all real slots full; it is not an extra scheduler request.
        # Common metadata remains untouched.
        self.query_start_loc = torch.zeros(
            self.num_reqs_capacity + 2, device=device, dtype=torch.int32
        )
        self.state_indices = torch.zeros(
            self.num_reqs_capacity + 1, device=device, dtype=torch.int32
        )
        self.has_initial_state = torch.zeros(
            self.num_reqs_capacity + 1, device=device, dtype=torch.bool
        )
        # Decode has one sequence per physical row, including null padding.
        self.decode_query_start_loc = torch.arange(
            self.num_reqs_capacity + 1, device=device, dtype=torch.int32
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        m = common_attn_metadata
        capacity = min(self.num_reqs_capacity, m.num_actual_tokens)
        if not 0 <= m.num_reqs <= capacity:
            raise ValueError("GDN request count exceeds the batch's metadata capacity")
        slots = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )[:, 0]
        self.query_start_loc[: m.num_reqs + 1].copy_(
            m.query_start_loc[: m.num_reqs + 1]
        )
        # Under FULL, num_actual_tokens is the padded capacity, while the last
        # query offset is the real token count. The appended interval is null.
        self.query_start_loc[m.num_reqs + 1 :].fill_(m.num_actual_tokens)
        self.state_indices[: m.num_reqs].copy_(slots[: m.num_reqs])
        self.state_indices[m.num_reqs :].zero_()
        self.has_initial_state[: m.num_reqs].copy_(
            m.compute_num_computed_tokens()[: m.num_reqs] > 0
        )
        self.has_initial_state[m.num_reqs :].zero_()
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(m, decode_threshold=1)
            if m.num_reqs
            else (0, 0, 0, 0)
        )
        # Retain native split counts, even when a general capture dummy happens
        # to contain only one-token requests. Decode needs no initialization mask;
        # packed execution always keeps one, including for those capture dummies.
        decode = m.max_query_len == 1
        return GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            # The GDN layer constructor rejects speculative decoding.
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=m.num_actual_tokens,
            non_spec_query_start_loc=(
                self.decode_query_start_loc[: capacity + 1]
                if decode
                else self.query_start_loc[: capacity + 2]
            ),
            non_spec_state_indices_tensor=(
                self.state_indices[:capacity]
                if decode
                else self.state_indices[: capacity + 1]
            ),
            has_initial_state=(
                None if decode else self.has_initial_state[: capacity + 1]
            ),
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> GDNAttentionMetadata:
        metadata = self.build(0, common_attn_metadata)
        self.state_indices.zero_()
        self.has_initial_state.zero_()
        return metadata
