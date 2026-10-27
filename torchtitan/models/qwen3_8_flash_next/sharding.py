# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Declarative sharding for residual mixing, lexical embeddings, and QSA."""

import spmd_types as spmd
from spmd_types import SpmdType

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.decoder_sharding import (
    attention_activation_placement,
    dense_param_placement,
    replicated_activation_placement,
    residual_activation_placement,
    token_id_placement,
)

DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP


def replicated_linear_sharding(layout: SpmdType) -> ShardingConfig:
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)},
        in_src_shardings={"input": layout},
        out_src_shardings=layout,
    )


def grouped_norm_sharding(layout: SpmdType) -> ShardingConfig:
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)},
        in_src_shardings={"x_TC": layout},
        out_src_shardings=layout,
        local_spmd=True,
    )


def residual_mixer_sharding(*, enable_sp: bool, use_combine: bool) -> ShardingConfig:
    layout = residual_activation_placement(enable_sp=enable_sp)
    return ShardingConfig(
        in_src_shardings={"x_TC": layout},
        out_src_shardings=(layout, layout, layout) if use_combine else layout,
        local_spmd=True,
    )


def block_sharding(*, enable_sp: bool, first_layer: bool) -> ShardingConfig:
    layout = residual_activation_placement(enable_sp=enable_sp)
    return ShardingConfig(
        in_src_shardings={
            "x_TC": replicated_activation_placement() if first_layer else layout
        },
        in_dst_shardings={"x_TC": layout},
        out_src_shardings=layout,
    )


def ngram_embedding_sharding() -> ShardingConfig:
    return ShardingConfig(
        state_shardings={
            name: dense_param_placement(tp=spmd.R)
            for name in (
                "layer_multipliers",
                "ngram_heads_vocab_sizes",
                "ngram_heads_offsets",
            )
        },
        in_src_shardings={
            "input_ids_T": token_id_placement(),
            "document_positions_T": token_id_placement(),
        },
        out_src_shardings=replicated_activation_placement(),
    )


def ngram_table_sharding() -> ShardingConfig:
    hashes = SpmdType(
        {DP: spmd.V, CP: spmd.V, TP: spmd.R}, spmd.PartitionSpec((DP, CP), None)
    )
    output = SpmdType(
        {DP: spmd.V, CP: spmd.V, TP: spmd.V},
        spmd.PartitionSpec((DP, CP), None, TP),
    )
    replicated = SpmdType(
        {DP: spmd.V, CP: spmd.V, TP: spmd.R}, spmd.PartitionSpec((DP, CP), None, None)
    )
    return ShardingConfig(
        # HF chunks rows; sharding features on TP keeps conversion distributed.
        state_shardings={"weight": dense_param_placement(tp=spmd.S(1))},
        in_src_shardings={"input": hashes},
        out_src_shardings=output,
        out_dst_shardings=replicated,
        local_spmd=True,
    )


def ple_sharding(*, enable_sp: bool) -> ShardingConfig:
    replicated = replicated_activation_placement()
    return ShardingConfig(
        in_src_shardings={
            "x_TC": residual_activation_placement(enable_sp=enable_sp),
            "input_ids_T": token_id_placement(),
            "document_positions_T": token_id_placement(),
        },
        in_dst_shardings={"x_TC": replicated},
        out_src_shardings=replicated,
        out_dst_shardings=residual_activation_placement(enable_sp=enable_sp),
        local_spmd=True,
    )


def indexer_sharding() -> ShardingConfig:
    replicated = replicated_activation_placement()
    return ShardingConfig(
        in_src_shardings={"x_TD": replicated},
        out_src_shardings=(replicated, replicated),
        local_spmd=True,
    )


def qsa_inner_sharding() -> ShardingConfig:
    heads = attention_activation_placement()
    return ShardingConfig(
        in_src_shardings={"query_THK": heads, "key_THK": heads, "value_THV": heads},
        out_src_shardings=heads,
        local_spmd=True,
    )
