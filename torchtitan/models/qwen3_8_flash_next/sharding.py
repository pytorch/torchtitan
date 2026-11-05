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


def _unsharded_param_placement(layout: SpmdType) -> SpmdType:
    """TP-unsharded weight read by an activation in ``layout``: ``I`` for a
    TP-``I`` input (TP without SP), else ``R``, as in the shared
    ``norm_sharding_config``."""
    tp = spmd.I if layout.local_type[TP] is spmd.I else spmd.R
    return dense_param_placement(tp=tp)


def unsharded_linear_sharding(layout: SpmdType) -> ShardingConfig:
    return ShardingConfig(
        state_shardings={"weight": _unsharded_param_placement(layout)},
        input_shardings={"input": layout},
        output_shardings=layout,
    )


def grouped_norm_sharding(layout: SpmdType) -> ShardingConfig:
    return ShardingConfig(
        state_shardings={"weight": _unsharded_param_placement(layout)},
        input_shardings={"input": layout},
        output_shardings=layout,
    )


def residual_mixer_sharding(*, enable_sp: bool, use_combine: bool) -> ShardingConfig:
    layout = residual_activation_placement(enable_sp=enable_sp)
    return ShardingConfig(
        input_shardings={"x_TC": layout},
        output_shardings=(layout, layout, layout) if use_combine else layout,
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
        # Local: the per-document convolution reset indexes the token axis
        # with rank-local positions (torch.arange), which global SPMD models
        # as one DP-partitioned axis.
        local_spmd=True,
    )


def indexer_sharding() -> ShardingConfig:
    replicated = replicated_activation_placement()
    return ShardingConfig(
        input_shardings={"x_TD": replicated},
        output_shardings=(replicated, replicated),
        # Local: block selection indexes the token axis with rank-local,
        # data-dependent positions, which global SPMD models as one
        # DP-partitioned axis.
        local_spmd=True,
    )


def qsa_inner_sharding() -> ShardingConfig:
    heads = attention_activation_placement()
    return ShardingConfig(
        input_shardings={"query_THK": heads, "key_THK": heads, "value_THV": heads},
        output_shardings=heads,
        # Local: the indexer teacher slices the token axis into query chunks,
        # which global SPMD rejects on the DP-partitioned axis.
        local_spmd=True,
    )
