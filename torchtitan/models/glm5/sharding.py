# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Declarative SPMD layouts for GLM-5-specific attention components."""

import spmd_types as spmd
from spmd_types import SpmdType

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    residual_activation_placement,
    token_id_placement,
)

DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP


def dsa_mask_placement() -> SpmdType:
    """Placement for a dense ``(1, query, key)`` legality mask."""
    return SpmdType(
        {DP: spmd.V, CP: spmd.V, TP: spmd.R},
        partition_spec=spmd.PartitionSpec(None, (DP, CP), None),
    )


def glm5_attention_sharding_config(*, enable_sp: bool) -> ShardingConfig:
    """Keep a GLM attention block on the decoder residual layout."""
    residual = residual_activation_placement(enable_sp=enable_sp)
    return ShardingConfig(in_src_shardings={"x_TD": residual})


def glm5_inner_attention_sharding_config() -> ShardingConfig:
    """Extend the common compact MLA boundary with DSA query metadata."""
    from torchtitan.models.common.decoder_sharding import (
        mla_attention_local_spmd_sharding_config,
    )

    config = mla_attention_local_spmd_sharding_config()
    extra = {
        "attention_metadata": dsa_mask_placement(),
        "topk_indices_QS": dense_activation_placement(tp=spmd.R, cp=spmd.S(0)),
    }
    assert config.in_src_shardings is not None
    assert config.in_dst_shardings is not None
    config.in_src_shardings = {**config.in_src_shardings, **extra}
    config.in_dst_shardings = {**config.in_dst_shardings, **extra}
    return config


def glm5_indexer_sharding_config() -> ShardingConfig:
    """Replicate indexer features on TP before the discrete top-k."""
    activation = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    return ShardingConfig(
        in_src_shardings={
            "hidden_states_TD": activation,
            "q_resid_TR": activation,
            "positions_T": token_id_placement(),
            "attention_mask_TK": activation,
        },
        in_dst_shardings={
            "hidden_states_TD": activation,
            "q_resid_TR": activation,
            "positions_T": token_id_placement(),
            "attention_mask_TK": activation,
        },
        out_src_shardings=activation,
    )


def glm5_indexer_topk_sharding_config() -> ShardingConfig:
    """Run top-k on TP-replicated scores and CP-local queries."""
    query = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    query_features = SpmdType(
        query.local_type,
        partition_spec=spmd.PartitionSpec((DP, CP), None, None),
    )
    key = dense_activation_placement(tp=spmd.R, cp=spmd.R)
    return ShardingConfig(
        in_src_shardings={
            "q_QNH": query_features,
            "k_KH": key,
            "weights_QN": query,
            "attention_mask_QK": query,
        },
        out_src_shardings=query,
        local_spmd=True,
    )
