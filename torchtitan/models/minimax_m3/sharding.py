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
    token_id_placement,
)

__all__ = ["msa_inner_attention_sharding_config"]


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
