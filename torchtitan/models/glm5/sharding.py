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
    dense_param_placement,
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
    attention = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    positions = token_id_placement()
    return ShardingConfig(
        in_src_shardings={"x_TD": residual, "positions": positions},
        in_dst_shardings={"x_TD": attention, "positions": positions},
        out_src_shardings=attention,
        out_dst_shardings=residual,
    )


def glm5_inner_attention_sharding_config() -> ShardingConfig:
    """Localize head-sharded DSA tensors at the FlexAttention boundary."""
    attention = dense_activation_placement(tp=spmd.S(1), cp=spmd.S(0))
    metadata = dsa_mask_placement()
    indices = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    return ShardingConfig(
        in_src_shardings={
            "q_THK": attention,
            "k_THK": attention,
            "v_THV": attention,
            "attention_metadata": metadata,
            "topk_indices_QS": indices,
        },
        in_dst_shardings={
            "q_THK": attention,
            "k_THK": attention,
            "v_THV": attention,
            "attention_metadata": metadata,
            "topk_indices_QS": indices,
        },
        out_src_shardings=attention,
        out_dst_shardings=attention,
        local_spmd=True,
    )


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
    key = dense_activation_placement(tp=spmd.R, cp=spmd.R)
    return ShardingConfig(
        in_src_shardings={
            "q_QNH": query,
            "k_KH": key,
            "weights_QN": query,
            "attention_mask_QK": query,
        },
        out_src_shardings=query,
        local_spmd=True,
    )


def replicated_weight_sharding_config(*, bias: bool = False) -> ShardingConfig:
    """Replicate a GLM-specific projection or norm across TP."""
    states = {"weight": dense_param_placement(tp=spmd.R)}
    if bias:
        states["bias"] = dense_param_placement(tp=spmd.R)
    return ShardingConfig(state_shardings=states)
