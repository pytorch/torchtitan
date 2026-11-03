# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Placements for carried CSA2 state and single-pass HC coefficients."""

import spmd_types as spmd

from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.decoder_sharding import (
    attention_activation_placement,
    dense_activation_placement,
    dense_param_placement,
    residual_activation_placement,
    token_id_placement,
)
from torchtitan.models.common.moe.sharding import moe_sharding_config
from torchtitan.models.deepseek_v4.sharding import hc_pre_sharding_config


def hc_coefficients_sharding_config(*, enable_sp: bool):
    config = hc_pre_sharding_config(enable_sp=enable_sp)
    assert isinstance(config.out_src_shardings, tuple)
    _, post, comb = config.out_src_shardings
    config.out_src_shardings = (post, post, comb)
    return config


def shared_attention_sharding_config(*, enable_sp: bool):
    replicated = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    residual = residual_activation_placement(enable_sp=enable_sp)
    return ShardingConfig(
        in_src_shardings={
            "x_TD": residual,
            **dict.fromkeys(
                (
                    "global_k_SD",
                    "index_k_SD",
                    "indices_TK",
                    "scores_TK",
                    "candidates_TC",
                ),
                replicated,
            ),
            "positions": token_id_placement(),
            "padding_mask": token_id_placement(),
        },
        out_src_shardings=(
            residual,
            replicated,
            replicated,
            replicated,
            replicated,
            replicated,
        ),
    )


def csa2_inner_sharding_config():
    query = attention_activation_placement()
    replicated = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    inputs = {
        "q_THD": query,
        **dict.fromkeys(
            ("local_k_TD", "global_k_SD", "indices_TK", "scores_TK"), replicated
        ),
        "sink_H": dense_param_placement(tp=spmd.S(0)),
        "padding_mask_T": token_id_placement(),
    }
    return ShardingConfig(
        in_src_shardings=inputs,
        in_dst_shardings=dict(inputs),
        out_src_shardings=query,
        local_spmd=True,
    )


def engram_hash_placement():
    return spmd.SpmdType(
        {"dp": spmd.V, "cp": spmd.V, "tp": spmd.R},
        partition_spec=spmd.PartitionSpec(("dp", "cp"), None, None),
    )


def engram_embedding_sharding_config(*, enable_sp: bool):
    hash_columns = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    sharded_features = spmd.SpmdType(
        {"dp": spmd.V, "cp": spmd.V, "tp": spmd.V},
        partition_spec=spmd.PartitionSpec(("dp", "cp"), None, "tp"),
    )
    features = spmd.SpmdType(
        {"dp": spmd.V, "cp": spmd.V, "tp": spmd.V if enable_sp else spmd.I},
        partition_spec=spmd.PartitionSpec(
            ("dp", "cp", "tp") if enable_sp else ("dp", "cp"), None, None
        ),
    )
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(1))},
        in_src_shardings={"input": hash_columns},
        in_dst_shardings={"input": hash_columns},
        out_src_shardings=sharded_features,
        out_dst_shardings=features,
        local_spmd=True,
    )


def compressor_sharding_config():
    replicated = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    inputs = {"x_TD": replicated, "positions_T": token_id_placement()}
    return ShardingConfig(
        in_src_shardings=inputs,
        in_dst_shardings=dict(inputs),
        out_src_shardings=(replicated, token_id_placement()),
        local_spmd=True,
    )


def indexer_sharding_config():
    replicated = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    inputs = dict.fromkeys(
        ("x_TD", "qr_TR", "latent_SD", "index_k_SD", "candidates_TC"), replicated
    )
    inputs.update(
        dict.fromkeys(("key_positions_S", "positions_T"), token_id_placement())
    )
    return ShardingConfig(
        in_src_shardings=inputs,
        in_dst_shardings=dict(inputs),
        out_src_shardings=(replicated,) * 4,
        local_spmd=True,
    )


def modality_moe_sharding_config(*, enable_sp: bool):
    config = moe_sharding_config(enable_sp=enable_sp)
    assert config.in_src_shardings is not None
    config.state_shardings["expert_bias_vl_E"] = dense_param_placement(tp=spmd.R)
    return config
