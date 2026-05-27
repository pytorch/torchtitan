# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

import spmd_types as spmd

from torchtitan.models.common.decoder import _decoder_spmd_input_config
from torchtitan.models.common.decoder_sharding import colwise_config, rowwise_config
from torchtitan.models.qwen3.sharding import set_qwen3_sharding_config
from torchtitan.protocols.sharding import (
    LocalSpmdConfig,
    NamedPlacement,
    ShardingConfig,
)
from torchtitan.protocols.types import MeshAxisName

if TYPE_CHECKING:
    from torchtitan.models.qwen3_vl.model import Qwen3VLModel

DP = MeshAxisName.DP
TP = MeshAxisName.TP


# ---------------------------------------------------------------------------
# Model input placements
# ---------------------------------------------------------------------------


def qwen3_vl_spmd_input_config():
    config = _decoder_spmd_input_config()
    multimodal_input: NamedPlacement = {DP: spmd.V, TP: spmd.I}
    config.extra_inputs.update(
        {
            "pixel_values": multimodal_input,
            "pixel_values_videos": multimodal_input,
            "grid_thw": multimodal_input,
            "grid_thw_videos": multimodal_input,
        }
    )
    return config


# ---------------------------------------------------------------------------
# Vision placement helpers
# ---------------------------------------------------------------------------


def activation_placement(*, tp: spmd.PerMeshAxisSpmdType = spmd.I) -> NamedPlacement:
    return {DP: spmd.V, TP: tp}


def param_placement() -> NamedPlacement:
    return {DP: spmd.R, TP: spmd.I}


def buffer_placement() -> NamedPlacement:
    return {DP: spmd.R, TP: spmd.R}


def vision_state_config(*, has_bias: bool = True) -> ShardingConfig:
    state = {"weight": param_placement()}
    if has_bias:
        state["bias"] = param_placement()
    return ShardingConfig(state_shardings=state)


# ---------------------------------------------------------------------------
# Vision TP configs
# ---------------------------------------------------------------------------


def set_qwen3_vl_vision_sharding_config(
    config: "Qwen3VLModel.Config",
    *,
    enable_tp: bool,
) -> None:
    vision = config.vision_encoder
    block = vision.block
    attn = block.attn
    mlp = block.mlp
    merger_configs = (vision.merger, vision.deepstack_merger)

    vision.sharding_config = ShardingConfig(
        state_shardings={"pos_embed": param_placement()},
    )
    norm = vision_state_config()
    block.norm_sharding_config = norm
    for merger in merger_configs:
        merger.norm_sharding_config = norm
    vision.rotary_pos_emb.sharding_config = ShardingConfig(
        state_shardings={"inv_freq": buffer_placement()},
    )

    if enable_tp:
        vision_tp_in = {"hidden_states": activation_placement()}
        vision_tp_dst = {"hidden_states": activation_placement(tp=spmd.R)}
        attn.sharding_config = ShardingConfig(
            in_src_shardings=vision_tp_in,
            in_dst_shardings=vision_tp_dst,
        )
        mlp.sharding_config = ShardingConfig(
            in_src_shardings={"hidden_state": activation_placement()},
            in_dst_shardings={"hidden_state": activation_placement(tp=spmd.R)},
        )
        attn.qkv.sharding_config = colwise_config()
        mlp.fc1.sharding_config = colwise_config()
        for linear_cfg in (attn.proj, mlp.fc2):
            config = rowwise_config()
            if linear_cfg.bias:
                config.state_tp_ir = {"bias"}
            linear_cfg.sharding_config = config

        for merger in merger_configs:
            # Merger fc1 is colwise, but its input comes from local vision
            # activations rather than the decoder's dense activation layout.
            fc1_config = colwise_config()
            fc1_config.in_src_shardings = {"input": activation_placement()}
            fc1_config.in_dst_shardings = {"input": activation_placement(tp=spmd.R)}
            merger.fc1.sharding_config = fc1_config

            fc2_config = rowwise_config()
            if merger.fc2.bias:
                fc2_config.state_tp_ir = {"bias"}
            merger.fc2.sharding_config = fc2_config
    else:
        for linear_cfg in (
            attn.qkv,
            attn.proj,
            mlp.fc1,
            mlp.fc2,
            vision.merger.fc1,
            vision.merger.fc2,
            vision.deepstack_merger.fc1,
            vision.deepstack_merger.fc2,
        ):
            linear_cfg.sharding_config = vision_state_config(
                has_bias=linear_cfg.bias,
            )

    vision.patch_embed.proj.sharding_config = ShardingConfig(
        state_shardings={
            "weight": param_placement(),
            "bias": param_placement(),
        },
        in_dst_shardings={"input": activation_placement()},
        out_src_shardings=activation_placement(),
        local_spmd=LocalSpmdConfig(),
    )


def set_qwen3_vl_sharding_config(
    config: "Qwen3VLModel.Config",
    *,
    loss_parallel: bool,
    enable_tp: bool,
    enable_ep: bool,
    chunked_loss: bool,
) -> None:
    """Fill ``sharding_config`` on all Qwen3-VL sub-configs.

    Delegates to ``set_qwen3_sharding_config`` with ``enable_sp=False``
    because Qwen3-VL keeps hidden states as ``Replicate`` (not
    ``Shard(1)``) — no SequenceParallel due to vision scatter and
    DeepStack needing full-sequence access.
    """
    config.spmd_input_config = qwen3_vl_spmd_input_config()
    set_qwen3_vl_vision_sharding_config(config, enable_tp=enable_tp)
    set_qwen3_sharding_config(
        config,
        loss_parallel=loss_parallel,
        enable_tp=enable_tp,
        enable_sp=False,
        enable_ep=enable_ep,
        chunked_loss=chunked_loss,
    )
