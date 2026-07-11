# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

import spmd_types as spmd
import torch
from torch import nn

from torchtitan.distributed import ParallelDims
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import set_current_spmd_mesh
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.protocols.sharding import (
    LocalMapConfig,
    RedistributionSpec,
    ShardingConfig,
    SpmdLayout,
)

if TYPE_CHECKING:
    from torchtitan.models.flux.model.model import FluxModel


DP = MeshAxisName.DP
CP = MeshAxisName.CP


def flux_activation_placement(
    *,
    cp: spmd.PerMeshAxisSpmdType,
) -> SpmdLayout:
    return SpmdLayout(
        {
            DP: spmd.S(0),
            CP: cp,
        }
    )


def set_flux_inner_attention_local_map(inner_attention_cfg) -> None:
    q_layout = flux_activation_placement(cp=spmd.S(1))
    kv_src_layout = flux_activation_placement(cp=spmd.S(1))
    kv_grad_layout = flux_activation_placement(cp=spmd.P)

    inner_attention_cfg.sharding_config = ShardingConfig(
        in_src_shardings={
            "q_BLNH": q_layout,
            "k_BLNH": kv_src_layout,
            "v_BLNH": kv_src_layout,
        },
        in_redist={
            "k_BLNH": RedistributionSpec.Config(
                axis=CP,
                src=spmd.S(1),
                dst=spmd.R,
            ),
            "v_BLNH": RedistributionSpec.Config(
                axis=CP,
                src=spmd.S(1),
                dst=spmd.R,
            ),
        },
        out_src_shardings=q_layout,
        local_map=LocalMapConfig(
            in_grad_placements=(q_layout, kv_grad_layout, kv_grad_layout)
        ),
    )


def set_flux_sharding_config(config: "FluxModel.Config") -> None:
    for block_cfg in config.double_blocks:
        set_flux_inner_attention_local_map(block_cfg.img_attn.inner_attention)
        set_flux_inner_attention_local_map(block_cfg.txt_attn.inner_attention)
        set_flux_inner_attention_local_map(block_cfg.inner_attention)

    for block_cfg in config.single_blocks:
        set_flux_inner_attention_local_map(block_cfg.inner_attention)


def annotate_dp_cp_params_as_r(model: nn.Module, parallel_dims: ParallelDims) -> None:
    # TODO(pianpwk): Infer these from the active SPMD mesh instead.
    with set_current_spmd_mesh(parallel_dims.spmd_dense_mesh()):
        for param in model.parameters():
            spmd.assert_type(param, spmd.R)


def annotate_flux_forward_inputs(
    *,
    latents: torch.Tensor,
    latent_pos_enc: torch.Tensor,
    t5_encodings: torch.Tensor,
    text_pos_enc: torch.Tensor,
    target: torch.Tensor,
    clip_encodings: torch.Tensor,
    timesteps: torch.Tensor,
) -> None:
    if get_spmd_backend() != "spmd_types":
        return

    sequence_type = {
        DP: spmd.S(0),
        CP: spmd.S(1),
    }
    batch_type = {
        DP: spmd.S(0),
        CP: spmd.R,
    }

    for tensor in (latents, latent_pos_enc, t5_encodings, text_pos_enc, target):
        spmd.assert_type(tensor, sequence_type)
    for tensor in (clip_encodings, timesteps):
        spmd.assert_type(tensor, batch_type)
