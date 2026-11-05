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


def unsharded_linear_sharding(layout: SpmdType) -> ShardingConfig:
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(
                tp=spmd.I if layout.local_type[TP] == spmd.I else spmd.R
            )
        },
        input_shardings={"input": layout},
        output_shardings=layout,
    )


def grouped_norm_sharding(layout: SpmdType) -> ShardingConfig:
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(
                tp=spmd.I if layout.local_type[TP] == spmd.I else spmd.R
            )
        },
        input_shardings={"input": layout},
        output_shardings=layout,
        local_spmd=True,
    )


def residual_mixer_sharding(*, enable_sp: bool, use_combine: bool) -> ShardingConfig:
    layout = residual_activation_placement(enable_sp=enable_sp)
    return ShardingConfig(
        input_shardings={"x_TC": layout},
        output_shardings=(layout, layout, layout) if use_combine else layout,
        local_spmd=True,
    )


def block_sharding(*, enable_sp: bool) -> ShardingConfig:
    layout = residual_activation_placement(enable_sp=enable_sp)
    return ShardingConfig(
        input_shardings={"x_TC": layout},
        output_shardings=layout,
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
        input_shardings={
            "input_ids_T": token_id_placement(),
            "document_positions_T": token_id_placement(),
        },
        output_shardings=replicated_activation_placement(),
    )


def ngram_table_sharding() -> ShardingConfig:
    hashes = SpmdType(
        {DP: spmd.V, CP: spmd.V, TP: spmd.R}, spmd.PartitionSpec((DP, CP), None)
    )
    output = SpmdType(
        {DP: spmd.V, CP: spmd.V, TP: spmd.V},
        spmd.PartitionSpec((DP, CP), None, TP),
    )
    return ShardingConfig(
        # DP owns rows; TP gathers only the selected feature vectors.
        state_shardings={
            "weight": SpmdType({DP: spmd.S(0), CP: spmd.R, TP: spmd.S(1)})
        },
        input_shardings={"input": hashes},
        # NGramEmbedding gathers the feature shards to TP-Replicate.
        output_shardings=output,
        local_spmd=True,
    )


def ple_sharding() -> ShardingConfig:
    # The block gathers the residual stream to Replicate around PLE.
    replicated = replicated_activation_placement()
    return ShardingConfig(
        input_shardings={
            "x_TC": replicated,
            "input_ids_T": token_id_placement(),
            "document_positions_T": token_id_placement(),
        },
        output_shardings=replicated,
        local_spmd=True,
    )


def indexer_sharding() -> ShardingConfig:
    replicated = replicated_activation_placement()
    return ShardingConfig(
        input_shardings={"x_TD": replicated},
        output_shardings=(replicated, replicated),
        local_spmd=True,
    )


def qsa_inner_sharding() -> ShardingConfig:
    heads = attention_activation_placement()
    return ShardingConfig(
        input_shardings={"query_THK": heads, "key_THK": heads, "value_THV": heads},
        output_shardings=heads,
        local_spmd=True,
    )
