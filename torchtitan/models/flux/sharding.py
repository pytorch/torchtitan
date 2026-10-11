# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import spmd_types as spmd
import torch
from spmd_types import SpmdType

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.base.sharding import ShardingConfig

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


def flux_inner_attention_config() -> ShardingConfig:
    """Local-SPMD attention kernel that gathers keys and values across CP."""
    q_layout = flux_activation_placement(cp=spmd.S(1))
    kv_src_layout = flux_activation_placement(cp=spmd.S(1))
    kv_dst_layout = flux_activation_placement(cp=spmd.R)
    return ShardingConfig(
        in_src_shardings={
            "q_BLHK": q_layout,
            "k_BLHK": kv_src_layout,
            "v_BLHV": kv_src_layout,
        },
        in_dst_shardings={
            "q_BLHK": q_layout,
            "k_BLHK": kv_dst_layout,
            "v_BLHV": kv_dst_layout,
        },
        out_src_shardings=q_layout,
        local_spmd=True,
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
