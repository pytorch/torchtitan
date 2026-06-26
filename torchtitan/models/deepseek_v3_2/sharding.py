# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

import spmd_types as spmd

from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
)
from torchtitan.protocols.sharding import (
    LocalMapConfig,
    ShardingConfig,
)

if TYPE_CHECKING:
    from torchtitan.models.deepseek_v3_2.model import DeepSeekV32Model


def set_deepseek_v3_2_sharding_config(
    config: "DeepSeekV32Model.Config",
) -> None:
    """Add V32-specific shardings on top of the V3 base.

    Per transformer layer:
      - Indexer: ``wq_b``/``weights_proj`` colwise ``S(0)@TP``;
        ``wk`` + ``k_norm`` + rope cache ``Replicate@TP``.
      - Inner attention: a ``LocalMapConfig`` whose six positional inputs
        are head-allgathered to ``Replicate@TP`` (and ``Replicate@CP`` on
        k/v/idx_k). q/k/v are gathered for the indexer KL loss that sums
        attention scores over heads; idx_* are gathered for ``select()``'s
        reduce_sum over index heads. Because the inputs are replicated, the
        local_map output is replicated too, so ``out_src`` is ``Replicate@TP``;
        ``out_dst`` then reshards it to ``S(2)`` to feed the rowwise ``wo``.
    """
    for layer_cfg in config.layers:
        indexer_cfg = layer_cfg.attention.indexer
        indexer_cfg.wq_b.sharding_config = ShardingConfig(
            state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
        )
        indexer_cfg.weights_proj.sharding_config = ShardingConfig(
            state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
        )
        indexer_cfg.wk.sharding_config = ShardingConfig(
            state_shardings={"weight": dense_param_placement(tp=spmd.R)},
        )
        indexer_cfg.k_norm.sharding_config = ShardingConfig(
            state_shardings={
                "weight": dense_param_placement(tp=spmd.R),
                "bias": dense_param_placement(tp=spmd.R),
            },
        )
        indexer_cfg.rope.sharding_config = ShardingConfig(
            state_shardings={"cache": dense_param_placement(tp=spmd.R)},
        )

        inner_cfg = layer_cfg.attention.inner_attention
        inner_cfg.sharding_config = ShardingConfig(
            in_src_shardings={
                "q_BLNH": dense_activation_placement(tp=spmd.S(2)),
                "k_BLNH": dense_activation_placement(tp=spmd.S(2)),
                "v_BLNH": dense_activation_placement(tp=spmd.S(2)),
                "idx_q_BLNH": dense_activation_placement(tp=spmd.S(2)),
                "idx_k_BLH": dense_activation_placement(tp=spmd.R),
                "idx_w_BLN": dense_activation_placement(tp=spmd.S(2)),
            },
            in_dst_shardings={
                "q_BLNH": dense_activation_placement(tp=spmd.R),
                "k_BLNH": dense_activation_placement(tp=spmd.R, cp=spmd.R),
                "v_BLNH": dense_activation_placement(tp=spmd.R, cp=spmd.R),
                "idx_q_BLNH": dense_activation_placement(tp=spmd.R),
                "idx_k_BLH": dense_activation_placement(tp=spmd.R, cp=spmd.R),
                "idx_w_BLN": dense_activation_placement(tp=spmd.R),
            },
            out_src_shardings=dense_activation_placement(tp=spmd.R),
            out_dst_shardings=dense_activation_placement(tp=spmd.S(2)),
            local_map=LocalMapConfig(
                in_grad_placements=(
                    dense_activation_placement(tp=spmd.P),
                    dense_activation_placement(tp=spmd.P, cp=spmd.P),
                    dense_activation_placement(tp=spmd.P, cp=spmd.P),
                    None,
                    None,
                    None,
                ),
            ),
        )
