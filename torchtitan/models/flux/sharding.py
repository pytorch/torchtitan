# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

import spmd_types as spmd

from torchtitan.models.common.decoder_sharding import (
    inner_attention_spmd_sharding_config,
)
from torchtitan.protocols.sharding import (
    LocalSpmdConfig,
    NamedPlacement,
    ShardingConfig,
)
from torchtitan.protocols.types import MeshAxisName

if TYPE_CHECKING:
    from torchtitan.models.common.linear import Linear
    from torchtitan.models.flux.model.model import FluxModel

DP = MeshAxisName.DP
CP = MeshAxisName.CP


def flux_sequence_placement() -> NamedPlacement:
    return {DP: spmd.S(0), CP: spmd.S(1)}


def flux_batch_placement() -> NamedPlacement:
    return {DP: spmd.S(0), CP: spmd.R}


def flux_model_config() -> ShardingConfig:
    return ShardingConfig(
        in_dst_shardings={
            "img": flux_sequence_placement(),
            "img_ids": flux_sequence_placement(),
            "txt": flux_sequence_placement(),
            "txt_ids": flux_sequence_placement(),
            "timesteps": flux_batch_placement(),
            "y": flux_batch_placement(),
        },
        out_src_shardings=flux_sequence_placement(),
        local_spmd=LocalSpmdConfig(),
    )


def dense_param_config(*, has_bias: bool = False) -> ShardingConfig:
    state = {"weight": {DP: spmd.R, CP: spmd.R}}
    if has_bias:
        state["bias"] = {DP: spmd.R, CP: spmd.R}
    return ShardingConfig(state_shardings=state)


def set_linear_sharding(linear_cfg: "Linear.Config") -> None:
    linear_cfg.sharding_config = dense_param_config(has_bias=linear_cfg.bias)


def set_flux_sharding_config(config: "FluxModel.Config") -> None:
    config.sharding_config = flux_model_config()
    for linear_cfg in (
        config.img_in,
        config.txt_in,
        config.time_in_config.in_layer,
        config.time_in_config.out_layer,
        config.vector_in_config.in_layer,
        config.vector_in_config.out_layer,
    ):
        set_linear_sharding(linear_cfg)

    for block in config.double_blocks:
        for linear_cfg in (
            block.img_mod.lin,
            block.txt_mod.lin,
            block.img_attn.qkv,
            block.img_attn.proj,
            block.txt_attn.qkv,
            block.txt_attn.proj,
            block.img_mlp_in,
            block.img_mlp_out,
            block.txt_mlp_in,
            block.txt_mlp_out,
        ):
            set_linear_sharding(linear_cfg)

        for attention in (block.img_attn, block.txt_attn):
            attention.norm.query_norm.sharding_config = dense_param_config()
            attention.norm.key_norm.sharding_config = dense_param_config()
            attention.inner_attention.sharding_config = (
                inner_attention_spmd_sharding_config()
            )
        block.inner_attention.sharding_config = inner_attention_spmd_sharding_config()

    for block in config.single_blocks:
        for linear_cfg in (
            block.linear1,
            block.linear2,
            block.modulation.lin,
        ):
            set_linear_sharding(linear_cfg)
        block.norm.query_norm.sharding_config = dense_param_config()
        block.norm.key_norm.sharding_config = dense_param_config()
        block.inner_attention.sharding_config = inner_attention_spmd_sharding_config()

    for linear_cfg in (
        config.final_layer_config.linear,
        config.final_layer_config.adaln_linear,
    ):
        set_linear_sharding(linear_cfg)
