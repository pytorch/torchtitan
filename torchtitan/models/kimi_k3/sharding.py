# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding configs for Kimi K3.

Declarations only: ``flavors.py`` sets these on each sub-config as it builds
it, and ``model.parallelize()`` applies them through the Module protocol.
Nothing here touches a mesh or a device.
"""

import spmd_types as spmd
from spmd_types import SpmdType

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.decoder_sharding import (
    attention_activation_placement,
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    norm_config,
    residual_activation_placement,
)


DP = MeshAxisName.DP
TP = MeshAxisName.TP


def tp_unsharded_weight_config(*, token_sharded: bool) -> ShardingConfig:
    """Keep the weight TP-unsharded: R when tokens are TP-sharded, for gradient reduction; otherwise I."""
    return ShardingConfig(
        state_shardings=norm_config(enable_sp=token_sharded).state_shardings
    )


def mla_attention_config(*, enable_sp: bool) -> ShardingConfig:
    """MLA input boundary in the residual layout."""
    return ShardingConfig(
        in_src_shardings={"x_TD": residual_activation_placement(enable_sp=enable_sp)},
    )


def kda_conv_config() -> ShardingConfig:
    """Depthwise KDA convolution weights, sharded on channels (heads)."""
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
    )


def inner_kda_config() -> ShardingConfig:
    """The KDA kernel runs on the rank-local heads in one local SPMD region."""
    projected_placement = dense_activation_placement(tp=spmd.S(1), cp=spmd.S(0))
    head_placement = attention_activation_placement()
    parameter_placement = dense_param_placement(tp=spmd.S(0))
    kernel_inputs = {
        "query_TC": projected_placement,
        "key_TC": projected_placement,
        "value_TC": projected_placement,
        "raw_gate_THK": head_placement,
        "raw_beta_TH": projected_placement,
        "conv_q_weight_C1W": parameter_placement,
        "conv_k_weight_C1W": parameter_placement,
        "conv_v_weight_C1W": parameter_placement,
        "A_log_H": parameter_placement,
        "dt_bias_HK": parameter_placement,
    }
    return ShardingConfig(
        in_src_shardings=kernel_inputs,
        in_dst_shardings=kernel_inputs,
        out_src_shardings=head_placement,
        out_dst_shardings=head_placement,
        local_spmd=True,
    )


def kda_config(*, enable_sp: bool) -> ShardingConfig:
    """Head-sharded TP for KDA, as Qwen3.5's GatedDeltaNet.

    The low-rank ``forget_a`` and the output norm keep replicated weights.
    """
    parameter_placement = dense_param_placement(tp=spmd.S(0))
    return ShardingConfig(
        state_shardings={
            "A_log": parameter_placement,
            "dt_bias": parameter_placement,
        },
        in_src_shardings={"x_TD": residual_activation_placement(enable_sp=enable_sp)},
    )


def latent_routed_down_config() -> ShardingConfig:
    """Down-projection into the routed experts' token-sharded latent space."""
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)},
        in_src_shardings={"input": dense_sequence_parallel_placement()},
    )


def latent_routed_up_config() -> ShardingConfig:
    """Up-projection out of the token-sharded latent space."""
    routed_up = tp_unsharded_weight_config(token_sharded=True)
    routed_up.out_src_shardings = dense_sequence_parallel_placement()
    return routed_up


def _block_residual_placement(*, tp: spmd.PerMeshAxisSpmdType) -> SpmdType:
    """Placement of the ``(tokens, entries, hidden)`` block-residual stack."""
    if isinstance(tp, spmd.Shard):
        return SpmdType(
            {DP: spmd.V, TP: spmd.V},
            partition_spec=spmd.PartitionSpec((DP, TP), None, None),
        )
    return SpmdType(
        {DP: spmd.V, TP: tp}, partition_spec=spmd.PartitionSpec(DP, None, None)
    )


def multimodal_first_layer_config(*, enable_sp: bool) -> ShardingConfig:
    """Layer 0 boundary after a TP-replicated multimodal token embedding.

    Restores the decoder's layout for the stream and the block-residual stack.
    """
    replicated = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    return ShardingConfig(
        in_src_shardings={
            "x_TD": replicated,
            "block_residual_TND": _block_residual_placement(tp=spmd.R),
        },
        in_dst_shardings={
            "x_TD": residual_activation_placement(enable_sp=enable_sp),
            "block_residual_TND": _block_residual_placement(
                tp=spmd.S(0) if enable_sp else spmd.I
            ),
        },
    )
