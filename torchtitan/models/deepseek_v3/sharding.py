# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding configs for DeepSeek V3 components.

Declarations only: ``flavors.py`` sets these on each sub-config as it builds
it, and ``model.parallelize()`` applies them through the Module protocol.
"""

import spmd_types as spmd

from torchtitan.models.base.sharding import ShardingConfig

from torchtitan.models.common.decoder_sharding import (
    dense_param_placement,
    residual_activation_placement,
    token_id_placement,
)


def mla_attention_config(*, enable_sp: bool) -> ShardingConfig:
    """MLA attention boundary: ``x`` in and out in the residual layout.

    ``Attention.forward`` explicitly gathers x once for all MLA branches. This
    config only declares its external input and output layout contracts.
    """
    layout = residual_activation_placement(enable_sp=enable_sp)
    return ShardingConfig(
        in_src_shardings={"x": layout},
        out_src_shardings=layout,
    )


def mtp_block_config(*, enable_sp: bool) -> ShardingConfig | None:
    """MTP block boundary: the valid mask follows token IDs under SP."""
    if not enable_sp:
        return None
    return ShardingConfig(
        in_src_shardings={
            "mtp_input_valid_mask": token_id_placement(),
        },
    )


def mtp_projection_config(*, enable_sp: bool) -> ShardingConfig:
    """MTP ``eh_proj``: replicated weight over residual-layout activations."""
    activation = residual_activation_placement(enable_sp=enable_sp)
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.R),
        },
        in_src_shardings={"input": activation},
        out_src_shardings=activation,
    )
