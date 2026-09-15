# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM metadata staging for TorchTitan's breakable CUDA graph GDN path."""

import torch
from torchtitan.experiments.rl.models.gdn_metadata import GDNGraphMetadata
from vllm.config import VllmConfig
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec


class TorchTitanGDNAttentionBackend(GDNAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "TORCHTITAN_GDN"

    @staticmethod
    def get_builder_cls() -> type["TorchTitanGDNAttentionMetadataBuilder"]:
        return TorchTitanGDNAttentionMetadataBuilder


class TorchTitanGDNAttentionMetadataBuilder(GDNAttentionMetadataBuilder):
    """Own packed buffers while retaining native FULL-decode metadata building."""

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.graph_metadata = GDNGraphMetadata.allocate(
            vllm_config.scheduler_config.max_num_seqs, device=device
        )

    def stage_packed(
        self, metadata: GDNAttentionMetadata, *, token_capacity: int
    ) -> GDNGraphMetadata:
        assert metadata.spec_sequence_masks is None
        assert metadata.num_spec_decodes == 0
        assert metadata.non_spec_query_start_loc is not None
        assert metadata.non_spec_state_indices_tensor is not None
        self.graph_metadata.update(
            metadata.non_spec_query_start_loc,
            metadata.non_spec_state_indices_tensor,
            metadata.has_initial_state,
            num_reqs=metadata.num_decodes + metadata.num_prefills,
            num_tokens=metadata.num_actual_tokens,
            token_capacity=token_capacity,
        )
        return self.graph_metadata.for_capacity(token_capacity)

    def stage_dummy(self, *, token_capacity: int) -> GDNGraphMetadata:
        self.graph_metadata.clear(token_capacity=token_capacity)
        return self.graph_metadata.for_capacity(token_capacity)
