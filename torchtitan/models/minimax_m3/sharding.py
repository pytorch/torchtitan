# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import spmd_types as spmd

from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.decoder_sharding import (
    attention_activation_placement,
    dense_activation_placement,
    replicated_activation_placement,
    residual_activation_placement,
    token_id_placement,
)

__all__ = ["msa_attention_sharding_config", "msa_inner_attention_sharding_config"]


def msa_attention_sharding_config(*, enable_sp: bool) -> ShardingConfig:
    """MSA boundary: gather ``x_TD`` once for the q/k/v and index projections."""
    layout = residual_activation_placement(enable_sp=enable_sp)
    return ShardingConfig(
        in_src_shardings={"x_TD": layout},
        in_dst_shardings={"x_TD": replicated_activation_placement()},
        out_src_shardings=layout,
    )


def msa_inner_attention_sharding_config() -> ShardingConfig:
    """Localize MSA's q/k/v, index queries (one head per KV group) and shared index keys."""
    heads = attention_activation_placement()
    input_shardings = {
        "q_THK": heads,
        "k_THK": heads,
        "v_THV": heads,
        "index_q_TGI": heads,
        "index_k_TI": dense_activation_placement(tp=spmd.R, cp=spmd.S(0)),
        "padding_mask_T": token_id_placement(),
    }
    return ShardingConfig(
        in_src_shardings=input_shardings,
        in_dst_shardings=dict(input_shardings),
        out_src_shardings=heads,
        local_spmd=True,
    )
