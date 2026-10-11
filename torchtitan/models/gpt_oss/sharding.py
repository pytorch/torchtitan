# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPT-OSS-specific sharding configs."""

import spmd_types as spmd

from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.decoder_sharding import (
    dense_param_placement,
    residual_activation_placement,
)
from torchtitan.models.common.moe_sharding import routed_expert_weight_placement


def attention_config(*, enable_sp: bool) -> ShardingConfig:
    """Attention boundary; the ``sinks`` parameter is sharded across heads."""
    layout = residual_activation_placement(enable_sp=enable_sp)
    return ShardingConfig(
        state_shardings={"sinks": dense_param_placement(tp=spmd.S(0))},
        in_src_shardings={"x": layout},
        out_src_shardings=layout,
    )


def routed_expert_linear_config(*, enable_ep: bool) -> ShardingConfig:
    """Routed-expert linear whose bias shares its weight's placement."""
    placement = routed_expert_weight_placement(enable_ep=enable_ep)
    return ShardingConfig(state_shardings={"weight": placement, "bias": placement})
