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
from torchtitan.protocols.sharding import LocalSpmdConfig, NamedPlacement, ShardingConfig
from torchtitan.protocols.types import MeshAxisName

if TYPE_CHECKING:
    from torchtitan.models.qwen3_vl.model import Qwen3VLModel


def _qwen3_vl_spmd_input_config():
    config = _decoder_spmd_input_config()
    multimodal_input: NamedPlacement = {MeshAxisName.DP: spmd.V, MeshAxisName.TP: spmd.I}
    config.extra_inputs.update(
        {
            "pixel_values": multimodal_input,
            "pixel_values_videos": multimodal_input,
            "grid_thw": multimodal_input,
            "grid_thw_videos": multimodal_input,
        }
    )
    return config


def _vision_activation_placement(
    *, tp: spmd.PerMeshAxisSpmdType = spmd.I
) -> NamedPlacement:
    return {MeshAxisName.DP: spmd.V, MeshAxisName.TP: tp}


def _vision_param_placement() -> NamedPlacement:
    return {MeshAxisName.DP: spmd.R, MeshAxisName.TP: spmd.I}


def _vision_buffer_placement() -> NamedPlacement:
    return {MeshAxisName.DP: spmd.R, MeshAxisName.TP: spmd.R}


def _vision_state_config(*, has_bias: bool = True) -> ShardingConfig:
    state = {"weight": _vision_param_placement()}
    if has_bias:
        state["bias"] = _vision_param_placement()
    return ShardingConfig(state_shardings=state)


def _vision_rowwise_config(*, has_bias: bool) -> ShardingConfig:
    config = rowwise_config()
    if has_bias:
        config.state_tp_ir = {"bias"}
    return config


def _vision_colwise_input_config() -> ShardingConfig:
    config = colwise_config()
    config.in_src_shardings = {"input": _vision_activation_placement()}
    config.in_dst_shardings = {"input": _vision_activation_placement(tp=spmd.R)}
    return config


def _set_qwen3_vl_vision_sharding_config(
    config: "Qwen3VLModel.Config",
    *,
    enable_tp: bool,
) -> None:
    config.vision_encoder.sharding_config = ShardingConfig(
        state_shardings={"pos_embed": _vision_param_placement()},
    )
    config.vision_encoder.norm_sharding_config = _vision_state_config()
    config.vision_encoder.rotary_sharding_config = ShardingConfig(
        state_shardings={"inv_freq": _vision_buffer_placement()},
    )

    if enable_tp:
        vision_tp_in = {"hidden_states": _vision_activation_placement()}
        vision_tp_dst = {"hidden_states": _vision_activation_placement(tp=spmd.R)}
        config.vision_encoder.attn_sharding_config = ShardingConfig(
            in_src_shardings=vision_tp_in,
            in_dst_shardings=vision_tp_dst,
        )
        config.vision_encoder.mlp_sharding_config = ShardingConfig(
            in_src_shardings={"hidden_state": _vision_activation_placement()},
            in_dst_shardings={
                "hidden_state": _vision_activation_placement(tp=spmd.R)
            },
        )
        config.vision_encoder.attn_qkv.sharding_config = colwise_config()
        config.vision_encoder.attn_proj.sharding_config = _vision_rowwise_config(
            has_bias=config.vision_encoder.attn_proj.bias,
        )
        config.vision_encoder.mlp_fc1.sharding_config = colwise_config()
        config.vision_encoder.mlp_fc2.sharding_config = _vision_rowwise_config(
            has_bias=config.vision_encoder.mlp_fc2.bias,
        )
        config.vision_encoder.merger_fc1.sharding_config = _vision_colwise_input_config()
        config.vision_encoder.merger_fc2.sharding_config = _vision_rowwise_config(
            has_bias=config.vision_encoder.merger_fc2.bias,
        )
    else:
        for linear_cfg in (
            config.vision_encoder.attn_qkv,
            config.vision_encoder.attn_proj,
            config.vision_encoder.mlp_fc1,
            config.vision_encoder.mlp_fc2,
            config.vision_encoder.merger_fc1,
            config.vision_encoder.merger_fc2,
        ):
            linear_cfg.sharding_config = _vision_state_config(
                has_bias=linear_cfg.bias,
            )

    config.vision_encoder.patch_embed_proj.sharding_config = ShardingConfig(
        state_shardings={
            "weight": _vision_param_placement(),
            "bias": _vision_param_placement(),
        },
        in_dst_shardings={"input": _vision_activation_placement()},
        out_src_shardings=_vision_activation_placement(),
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
    config.spmd_input_config = _qwen3_vl_spmd_input_config()
    _set_qwen3_vl_vision_sharding_config(config, enable_tp=enable_tp)
    set_qwen3_sharding_config(
        config,
        loss_parallel=loss_parallel,
        enable_tp=enable_tp,
        enable_sp=False,
        enable_ep=enable_ep,
        chunked_loss=chunked_loss,
    )
