# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Shape suffixes: B = batch, L = sequence, T = packed tokens, H = heads,
# K = query/key head dimension, V = value head dimension, D = generic head dimension,
# N = sequence boundaries.

import spmd_types as spmd
import torch
from spmd_types import SpmdType

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.attention import VarlenAttentionMetadata

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
    cp_group = spmd_mesh_group(CP)
    cp_size = cp_group.size() if cp_group is not None else 1
    offsets_N = torch.arange(batch_size + 1, device=device, dtype=torch.int32)
    return VarlenAttentionMetadata(
        cu_seq_q=offsets_N * seq_len,
        cu_seq_k=offsets_N * (seq_len * cp_size),
        max_q=seq_len,
        max_k=seq_len * cp_size,
    )


@spmd.local_map(
    in_types=(
        spmd.PartitionSpec(DP, CP, None, None),
        spmd.PartitionSpec(DP, None, None, None),
        spmd.PartitionSpec(DP, None, None, None),
    ),
    out_types=(
        spmd.PartitionSpec((DP, CP), None, None),
        spmd.PartitionSpec(DP, None, None),
        spmd.PartitionSpec(DP, None, None),
    ),
)
def _pack_attention_inputs(
    q_BLHK: torch.Tensor, k_BLHK: torch.Tensor, v_BLHV: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return q_BLHK.flatten(0, 1), k_BLHK.flatten(0, 1), v_BLHV.flatten(0, 1)


def prepare_flux_attention_inputs(
    q_BLHK: torch.Tensor, k_BLHK: torch.Tensor, v_BLHV: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather K/V along sequence before packing to preserve sample boundaries."""
    cp_group = spmd_mesh_group(CP)
    if cp_group is not None:
        k_BLHK, v_BLHV = (
            spmd.redistribute(
                x_BLHD,
                cp_group,
                src=spmd.S(1),
                dst=spmd.R,
                backward_options={"op_dtype": x_BLHD.dtype},
            )
            for x_BLHD in (k_BLHK, v_BLHV)
        )
    return _pack_attention_inputs(q_BLHK, k_BLHK, v_BLHV)


@spmd.local_map(
    in_types=(spmd.PartitionSpec((DP, CP), None, None), None),
    out_types=spmd.PartitionSpec(DP, CP, None, None),
)
def unpack_flux_attention_output(
    out_THV: torch.Tensor, batch_size: int
) -> torch.Tensor:
    return out_THV.unflatten(0, (batch_size, -1))


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
