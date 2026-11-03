# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding config for the Muse Spark Open NaViT encoder."""

from dataclasses import replace

import spmd_types as spmd
from spmd_types import SpmdType
from torch.utils._pytree import tree_map

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.base.sharding import ShardingConfig


DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP


def cp_replicated_vision_sharding_config(config: ShardingConfig) -> ShardingConfig:
    """Keep post-encoder perception projections CP-replicated."""

    def replicate_cp(layout):
        if isinstance(layout, SpmdType):
            return SpmdType(
                {**layout.local_type, CP: spmd.R},
                partition_spec=layout.partition_spec,
            )
        return layout

    def replicate_tree(value):
        return tree_map(replicate_cp, value, is_leaf=lambda x: isinstance(x, SpmdType))

    return replace(
        config,
        state_shardings=replicate_tree(config.state_shardings),
        in_src_shardings=replicate_tree(config.in_src_shardings),
        in_dst_shardings=replicate_tree(config.in_dst_shardings),
        out_src_shardings=replicate_tree(config.out_src_shardings),
        out_dst_shardings=replicate_tree(config.out_dst_shardings),
    )


def navit_encoder_sharding_config() -> ShardingConfig:
    parameter_invariant = SpmdType({DP: spmd.R, CP: spmd.I, TP: spmd.I})
    activation_invariant = SpmdType({DP: spmd.V, CP: spmd.I, TP: spmd.I})
    return ShardingConfig(
        state_shardings={"positional_embedding_vlm": parameter_invariant},
        out_src_shardings=activation_invariant,
        out_dst_shardings=SpmdType({DP: spmd.V, CP: spmd.R, TP: spmd.I}),
    )


def vision_rope_freq_sharding_config() -> ShardingConfig:
    return ShardingConfig(
        state_shardings={"inv_freq": SpmdType({DP: spmd.R, CP: spmd.I, TP: spmd.I})},
    )
