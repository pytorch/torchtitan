# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import spmd_types as spmd
from spmd_types import SpmdType

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
)
from torchtitan.models.common.moe.sharding import replicated_param_placement_sparse
from torchtitan.models.common.vision_encoder_sharding import (
    invariant_norm_sharding_config,
)
from torchtitan.models.muse_spark.vision_encoder_sharding import (
    cp_replicated_vision_sharding_config,
)


DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP


def multimodal_bank_indices_placement(*, enable_sp: bool) -> SpmdType:
    """Placement for token-aligned indices into one perception bank."""
    token_axes = (DP, CP, TP) if enable_sp else (DP, CP)
    return SpmdType(
        {
            DP: spmd.V,
            CP: spmd.V,
            TP: spmd.V if enable_sp else spmd.I,
        },
        partition_spec=spmd.PartitionSpec(token_axes),
    )


def muse_spark_activation_placement(*, enable_sp: bool) -> SpmdType:
    return (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
    )


def audio_wrapper_sharding_config(input_name: str) -> ShardingConfig:
    invariant = SpmdType({DP: spmd.V, CP: spmd.I, TP: spmd.I})
    return ShardingConfig(
        in_src_shardings={input_name: invariant},
        in_dst_shardings={input_name: invariant},
        out_src_shardings=invariant,
        out_dst_shardings=invariant,
    )


def audio_invariant_state_sharding_config(
    *,
    has_bias: bool = False,
) -> ShardingConfig:
    config = audio_wrapper_sharding_config("input")
    invariant_state = SpmdType({DP: spmd.R, CP: spmd.I, TP: spmd.I})
    config.state_shardings = {"weight": invariant_state}
    if has_bias:
        config.state_shardings["bias"] = invariant_state
    return config


def transformer_block_sharding_config(*, enable_sp: bool) -> ShardingConfig:
    activation = muse_spark_activation_placement(enable_sp=enable_sp)
    return ShardingConfig(
        in_src_shardings={"x": activation},
        in_dst_shardings={"x": activation},
        out_src_shardings=activation,
        out_dst_shardings=activation,
    )


def residual_gate_sharding_config(*, enable_sp: bool) -> ShardingConfig:
    activation = muse_spark_activation_placement(enable_sp=enable_sp)
    return ShardingConfig(
        state_shardings={
            "gate": dense_param_placement(tp=spmd.R if enable_sp else spmd.I)
        },
        in_src_shardings={"stream_TD": activation, "branch_TD": activation},
        in_dst_shardings={"stream_TD": activation, "branch_TD": activation},
        out_src_shardings=activation,
        out_dst_shardings=activation,
    )


def compressed_moe_linear_sharding_config(*, has_bias: bool = False) -> ShardingConfig:
    activation = dense_sequence_parallel_placement()
    state = {"weight": dense_param_placement(tp=spmd.R)}
    if has_bias:
        state["bias"] = dense_param_placement(tp=spmd.R)
    return ShardingConfig(
        state_shardings=state,
        in_src_shardings={"input": activation},
        in_dst_shardings={"input": activation},
        out_src_shardings=activation,
        out_dst_shardings=activation,
    )


def post_expert_norm_sharding_config(*, enable_ep: bool) -> ShardingConfig:
    return ShardingConfig(
        state_shardings={
            "weight": replicated_param_placement_sparse()
            if enable_ep
            else dense_param_placement(tp=spmd.R)
        },
    )


def audio_encoder_sharding_config() -> ShardingConfig:
    invariant = SpmdType({DP: spmd.V, CP: spmd.I, TP: spmd.I})
    return ShardingConfig(
        in_src_shardings={
            "waveforms_SC": invariant,
            "waveform_padded_lengths_N": invariant,
        },
        in_dst_shardings={
            "waveforms_SC": invariant,
            "waveform_padded_lengths_N": invariant,
        },
        out_src_shardings=invariant,
        out_dst_shardings=SpmdType({DP: spmd.V, CP: spmd.R, TP: spmd.I}),
    )


def perception_norm_sharding_config(*, enable_sp: bool) -> ShardingConfig:
    config = cp_replicated_vision_sharding_config(invariant_norm_sharding_config())
    if enable_sp:
        config.out_dst_shardings = SpmdType({DP: spmd.V, CP: spmd.R, TP: spmd.R})
    return config
