# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding configs for Kimi K3. Same pattern as ``qwen3_5/sharding.py``.

Declarations only: functions here set ``ShardingConfig`` on sub-configs of an
already-built config tree, and ``model.parallelize()`` applies them through the
Module protocol. Nothing here touches a mesh or a device.
"""

from typing import TYPE_CHECKING

import spmd_types as spmd
from spmd_types import SpmdType

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.moe_sharding import set_moe_sharding_config
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig

if TYPE_CHECKING:
    from torchtitan.models.kimi_k3.model import KimiK3Model
    from torchtitan.models.kimi_k3.vision_encoder import KimiK3VisionEncoder


DP = MeshAxisName.DP


def _set_inner_kda_sharding(inner_kda: Module.Config) -> None:
    """Set the local boundary around Attention Gym's KDA kernels."""
    token_channels = SpmdType({DP: spmd.V}, partition_spec=spmd.PartitionSpec(DP, None))
    token_heads = SpmdType(
        {DP: spmd.V}, partition_spec=spmd.PartitionSpec(DP, None, None)
    )
    parameter = SpmdType({DP: spmd.R})

    inner_kda.sharding_config = ShardingConfig(
        in_src_shardings={
            "query_TC": token_channels,
            "key_TC": token_channels,
            "value_TC": token_channels,
            "raw_gate_THK": token_heads,
            "raw_beta_TH": token_channels,
            "conv_q_weight_C1W": parameter,
            "conv_k_weight_C1W": parameter,
            "conv_v_weight_C1W": parameter,
            "A_log_H": parameter,
            "dt_bias_HK": parameter,
        },
        in_dst_shardings={
            "query_TC": token_channels,
            "key_TC": token_channels,
            "value_TC": token_channels,
            "raw_gate_THK": token_heads,
            "raw_beta_TH": token_channels,
            "conv_q_weight_C1W": parameter,
            "conv_k_weight_C1W": parameter,
            "conv_v_weight_C1W": parameter,
            "A_log_H": parameter,
            "dt_bias_HK": parameter,
        },
        out_src_shardings=token_heads,
        out_dst_shardings=token_heads,
        local_map=LocalMapConfig(
            in_grad_placements=(
                token_channels,
                token_channels,
                token_channels,
                token_heads,
                token_channels,
                parameter,
                parameter,
                parameter,
                parameter,
                parameter,
            ),
        ),
    )


def set_kimi_k3_sharding_config(
    config: "KimiK3Model.Config", *, enable_ep: bool, enable_sp: bool = False
) -> None:
    """Declare Kimi K3 vision-buffer and expert sharding.

    Vision buffers replicate across DP ranks. The routed experts shard on the
    expert axis; ``set_moe_sharding_config`` declares that layout, and its
    input boundary lifts the plain incoming activations itself.
    """
    if config.vision_encoder is not None:
        _set_vision_buffer_sharding(config.vision_encoder)

    for layer in config.layers:
        if layer.delta_attention is not None:
            _set_inner_kda_sharding(layer.delta_attention.inner_kda)
        if layer.moe is not None:
            set_moe_sharding_config(
                layer.moe,
                enable_ep=enable_ep,
                # TODO: flip to True from the caller once the
                # tensor-parallel PR lands; with EP alone the internals run
                # without sequence parallel.
                enable_sp=enable_sp,
                expert_param_layout={
                    "w1_EFD": spmd.S(1),
                    "w2_EDF": spmd.S(2),
                    "w3_EFD": spmd.S(1),
                },
            )


def _set_vision_buffer_sharding(config: "KimiK3VisionEncoder.Config") -> None:
    """Declare the K3 rotary buffer replicated across DP ranks."""
    config.rotary_pos_emb.sharding_config = ShardingConfig(
        state_shardings={"inv_freq": SpmdType({DP: spmd.R})},
    )
