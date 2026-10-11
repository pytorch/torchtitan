# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding configs for Qwen3.5 hybrid attention model.

``flavors.py`` sets these ``ShardingConfig`` values where it builds each
sub-config, so that ``model.parallelize()`` applies TP via the Module
protocol.

Full-attention layers: TP on wq/wk/wv/wo with local SPMD inner attention;
each layer's MRoPE ``cache`` buffer is sharded Replicate.
GatedDeltaNet layers: head-sharded TP on projections (ColwiseParallel) and
out_proj (RowwiseParallel); the GDN kernel and depthwise Conv1d run on local
tensors in a local SPMD region.
"""

import spmd_types as spmd
from spmd_types import SpmdType

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.decoder_sharding import (
    attention_activation_placement,
    dense_activation_placement,
    dense_param_placement,
    replicated_activation_placement,
    residual_activation_placement,
    stacked_colwise_config,
)

DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP


def transformer_block_config(
    *, enable_sp: bool, after_vision_scatter: bool
) -> ShardingConfig:
    """Transformer block boundary in the residual layout.

    The first block after the vision scatter takes a Replicate input and
    restores the decoder layout.
    """
    layer_layout = residual_activation_placement(enable_sp=enable_sp)
    input_layout = (
        replicated_activation_placement() if after_vision_scatter else layer_layout
    )
    return ShardingConfig(
        in_src_shardings={"x_TD": input_layout},
        in_dst_shardings={"x_TD": layer_layout},
        out_src_shardings=layer_layout,
    )


def decoder_norm_config(*, enable_sp: bool) -> ShardingConfig:
    """Pre-attention norm: Replicate weight, residual-layout activations."""
    activation_layout = residual_activation_placement(enable_sp=enable_sp)
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)},
        in_src_shardings={"input": activation_layout},
        out_src_shardings=activation_layout,
    )


def attention_config(*, enable_sp: bool) -> ShardingConfig:
    """Full-attention boundary: ``x_TD`` in and out in the residual layout."""
    layout = residual_activation_placement(enable_sp=enable_sp)
    return ShardingConfig(
        in_src_shardings={"x_TD": layout},
        out_src_shardings=layout,
    )


def conv_weight_config() -> ShardingConfig:
    """Depthwise Conv1d weight sharded Shard(0) on out-channels (head-sharded)."""
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
    )


def gated_rms_norm_config() -> ShardingConfig:
    """GatedDeltaNet output norm over head-sharded ``[T, H, V]`` activations."""
    head_placement = attention_activation_placement()
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)},
        in_src_shardings={
            "x": head_placement,
            "gate": head_placement,
        },
        in_dst_shardings={
            "x": head_placement,
            "gate": head_placement,
        },
        out_src_shardings=head_placement,
        out_dst_shardings=head_placement,
    )


def inner_gated_delta_net_config() -> ShardingConfig:
    """Local SPMD boundary for the head-parallel convolution and recurrence.

    The projections are 2D [T, C], while the recurrence output is 3D
    [T, H, V]. Both shard the feature/head axis on TP.
    """
    projected_placement = dense_activation_placement(tp=spmd.S(1), cp=spmd.S(0))
    head_placement = attention_activation_placement()
    parameter_placement = dense_param_placement(tp=spmd.S(0))
    input_shardings = {
        "query_TC": projected_placement,
        "key_TC": projected_placement,
        "value_TC": projected_placement,
        "a_TH": projected_placement,
        "b_TH": projected_placement,
        "conv_q_weight_C1W": parameter_placement,
        "conv_k_weight_C1W": parameter_placement,
        "conv_v_weight_C1W": parameter_placement,
        "A_log_H": parameter_placement,
        "dt_bias_H": parameter_placement,
    }
    return ShardingConfig(
        in_src_shardings=input_shardings,
        in_dst_shardings=dict(input_shardings),
        out_src_shardings=head_placement,
        out_dst_shardings=head_placement,
        local_spmd=True,
    )


def gated_delta_net_config(*, enable_sp: bool) -> ShardingConfig:
    """GatedDeltaNet boundary with head-sharded ``A_log``/``dt_bias``."""
    layout = residual_activation_placement(enable_sp=enable_sp)
    parameter_placement = dense_param_placement(tp=spmd.S(0))
    return ShardingConfig(
        state_shardings={
            "A_log": parameter_placement,
            "dt_bias": parameter_placement,
        },
        in_src_shardings={"x_TD": layout},
        out_src_shardings=layout,
    )


def sigmoid_gated_w13_config() -> ShardingConfig:
    """Shared-expert ``w13`` over the input the shared FFN gathers once."""
    return stacked_colwise_config(input_layout=replicated_activation_placement())


def sigmoid_gate_config() -> ShardingConfig:
    """Shared-expert sigmoid gate: Replicate weights and activations."""
    replicated = replicated_activation_placement()
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.R),
            "bias": dense_param_placement(tp=spmd.R),
        },
        in_src_shardings={"input": replicated},
        out_src_shardings=replicated,
    )


def vision_encoder_config() -> ShardingConfig:
    """Vision encoder root: invariant ``pos_embed``, replicated output.

    All activations flow without SP in the vision encoder.
    """
    return ShardingConfig(
        state_shardings={"pos_embed": SpmdType({DP: spmd.R, CP: spmd.R, TP: spmd.I})},
        out_src_shardings=SpmdType({DP: spmd.V, CP: spmd.R, TP: spmd.I}),
        out_dst_shardings=SpmdType({DP: spmd.V, CP: spmd.R, TP: spmd.R}),
    )


def vision_rotary_embedding_config() -> ShardingConfig:
    """Vision rotary embedding with a TP-invariant ``inv_freq``."""
    state_placement = SpmdType({DP: spmd.R, CP: spmd.R, TP: spmd.I})
    return ShardingConfig(
        state_shardings={"inv_freq": state_placement},
        out_src_shardings=state_placement,
    )
