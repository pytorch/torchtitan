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
    NamedPlacement,
    ShardingConfig,
    SpmdInputConfig,
)
from torchtitan.protocols.types import MeshAxisName

if TYPE_CHECKING:
    from torchtitan.models.common.linear import Linear
    from torchtitan.models.flux.model.layers import (
        DoubleStreamBlock,
        LastLayer,
        MLPEmbedder,
        Modulation,
        QKNorm,
        SelfAttention,
        SingleStreamBlock,
    )
    from torchtitan.models.flux.model.model import FluxModel

DP = MeshAxisName.DP
CP = MeshAxisName.CP


def flux_sequence_placement() -> NamedPlacement:
    return {DP: spmd.S(0), CP: spmd.S(1)}


def flux_batch_placement() -> NamedPlacement:
    return {DP: spmd.S(0), CP: spmd.R}


def flux_spmd_input_config() -> SpmdInputConfig:
    return SpmdInputConfig(
        labels=flux_sequence_placement(),
        extra_inputs={
            "img": flux_sequence_placement(),
            "img_ids": flux_sequence_placement(),
            "txt": flux_sequence_placement(),
            "txt_ids": flux_sequence_placement(),
            "timesteps": flux_batch_placement(),
            "y": flux_batch_placement(),
        },
    )


def flux_state_config(*, has_bias: bool = False) -> ShardingConfig:
    state = {"weight": {DP: spmd.R, CP: spmd.R}}
    if has_bias:
        state["bias"] = {DP: spmd.R, CP: spmd.R}
    return ShardingConfig(state_shardings=state)


def set_flux_linear_sharding(linear_cfg: "Linear.Config") -> None:
    linear_cfg.sharding_config = flux_state_config(has_bias=linear_cfg.bias)


def set_flux_qk_norm_sharding(norm_cfg: "QKNorm.Config") -> None:
    norm_cfg.query_norm.sharding_config = flux_state_config()
    norm_cfg.key_norm.sharding_config = flux_state_config()


def set_flux_mlp_embedder_sharding(config: "MLPEmbedder.Config") -> None:
    set_flux_linear_sharding(config.in_layer)
    set_flux_linear_sharding(config.out_layer)


def set_flux_modulation_sharding(config: "Modulation.Config") -> None:
    set_flux_linear_sharding(config.lin)


def set_flux_self_attention_sharding(config: "SelfAttention.Config") -> None:
    set_flux_linear_sharding(config.qkv)
    set_flux_linear_sharding(config.proj)
    set_flux_qk_norm_sharding(config.norm)
    config.inner_attention.sharding_config = inner_attention_spmd_sharding_config()


def set_flux_double_block_sharding(config: "DoubleStreamBlock.Config") -> None:
    set_flux_modulation_sharding(config.img_mod)
    set_flux_modulation_sharding(config.txt_mod)
    set_flux_self_attention_sharding(config.img_attn)
    set_flux_self_attention_sharding(config.txt_attn)
    set_flux_linear_sharding(config.img_mlp_in)
    set_flux_linear_sharding(config.img_mlp_out)
    set_flux_linear_sharding(config.txt_mlp_in)
    set_flux_linear_sharding(config.txt_mlp_out)
    config.inner_attention.sharding_config = inner_attention_spmd_sharding_config()


def set_flux_single_block_sharding(config: "SingleStreamBlock.Config") -> None:
    set_flux_linear_sharding(config.linear1)
    set_flux_linear_sharding(config.linear2)
    set_flux_modulation_sharding(config.modulation)
    set_flux_qk_norm_sharding(config.norm)
    config.inner_attention.sharding_config = inner_attention_spmd_sharding_config()


def set_flux_last_layer_sharding(config: "LastLayer.Config") -> None:
    set_flux_linear_sharding(config.linear)
    set_flux_linear_sharding(config.adaln_linear)


def set_flux_sharding_config(config: "FluxModel.Config") -> None:
    config.spmd_input_config = flux_spmd_input_config()
    set_flux_linear_sharding(config.img_in)
    set_flux_linear_sharding(config.txt_in)
    set_flux_mlp_embedder_sharding(config.time_in_config)
    set_flux_mlp_embedder_sharding(config.vector_in_config)
    for block in config.double_blocks:
        set_flux_double_block_sharding(block)
    for block in config.single_blocks:
        set_flux_single_block_sharding(block)
    set_flux_last_layer_sharding(config.final_layer_config)
