# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Shape suffixes: T = packed tokens.

import spmd_types as spmd
import torch
from spmd_types import SpmdType

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_size
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.attention import (
    create_varlen_metadata_for_document,
    VarlenAttentionMetadata,
)

DP = MeshAxisName.DP
CP = MeshAxisName.CP


def flux_activation_placement(
    *,
    cp: spmd.PerMeshAxisSpmdType,
) -> SpmdType:
    return SpmdType(
        {
            DP: spmd.S(0),
            CP: cp,
        }
    )


def flux_input_sharding() -> dict[str, SpmdType]:
    """Input sharding for Flux training and validation."""
    return {
        name: flux_activation_placement(cp=spmd.S(1))
        for name in ("img", "img_ids", "txt", "txt_ids", "target")
    }


def flux_inner_attention_sharding_config() -> ShardingConfig:
    """Packed attention with local queries and already gathered keys/values."""
    q_layout = SpmdType(
        {DP: spmd.V, CP: spmd.V},
        partition_spec=spmd.PartitionSpec((DP, CP), None, None),
    )
    kv_layout = SpmdType(
        {DP: spmd.V, CP: spmd.R},
        partition_spec=spmd.PartitionSpec(DP, None, None),
    )
    return ShardingConfig(
        in_src_shardings={
            "q_THK": q_layout,
            "k_THK": kv_layout,
            "v_THV": kv_layout,
        },
        out_src_shardings=q_layout,
        local_spmd=True,
    )


def create_flux_attention_metadata(
    batch_size: int, seq_len: int, *, device: torch.device
) -> VarlenAttentionMetadata:
    """Boundaries for each sample's local queries and gathered keys/values."""
    positions_T = torch.arange(seq_len, device=device, dtype=torch.int32).repeat(
        batch_size
    )
    metadata = create_varlen_metadata_for_document(
        positions_T,
        max_num_documents=batch_size,
        max_context_length=seq_len,
    )
    cp_size = spmd_mesh_size(CP)
    return metadata._replace(
        cu_seq_k=metadata.cu_seq_k * cp_size,
        max_k=metadata.max_k * cp_size,
    )


def annotate_flux_forward_inputs(
    *,
    latents: torch.Tensor,
    latent_pos_enc: torch.Tensor,
    t5_encodings: torch.Tensor,
    text_pos_enc: torch.Tensor,
    target: torch.Tensor | None,
    clip_encodings: torch.Tensor,
    timesteps: torch.Tensor,
) -> None:
    sequence_type = {
        DP: spmd.S(0),
        CP: spmd.S(1),
    }
    batch_type = {
        DP: spmd.S(0),
        CP: spmd.R,
    }

    for tensor in (latents, latent_pos_enc, t5_encodings, text_pos_enc):
        spmd.assert_type(tensor, sequence_type)
    if target is not None:
        spmd.assert_type(target, sequence_type)
    for tensor in (clip_encodings, timesteps):
        spmd.assert_type(tensor, batch_type)
