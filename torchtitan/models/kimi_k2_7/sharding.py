# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config-based sharding for Kimi K2.5 (MoonViT3d + DeepSeekV3).

Sets ``ShardingConfig`` on every sub-config so ``model.parallelize()`` applies
TP/EP/SP uniformly via the Module protocol.

- Decoder (MLA + MoE): reuses ``set_deepseek_v3_sharding_config``. Vision
  embeddings are gathered for local tokens, so the decoder uses standard
  DeepSeek sequence parallelism.
- Vision encoder: activations flow ``Invariant`` (no SP -- the patch sequence is
  short, so sequence-sharding would add gather/scatter around the block-diagonal
  attention for little memory gain). Only the linear layers are Colwise/Rowwise
  sharded for memory; norms and position embeddings stay ``Invariant``.
"""

from typing import TYPE_CHECKING

import spmd_types as spmd
from spmd_types import SpmdType

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.vision_encoder_sharding import (
    invariant_norm_config,
    set_vision_transformer_block_sharding_config,
    vision_colwise_config,
    vision_invariant_linear_config,
    vision_scaled_bias_rowwise_config,
)
from torchtitan.models.deepseek_v3.sharding import set_deepseek_v3_sharding_config
from torchtitan.protocols.sharding import ShardingConfig

DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP

if TYPE_CHECKING:
    from torchtitan.models.kimi_k2_7.model import KimiK25Model


def set_kimi_k2_5_sharding_config(
    config: "KimiK25Model.Config",
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    set_deepseek_v3_sharding_config(
        config,
        enable_sp=enable_sp,
        enable_ep=enable_ep,
    )
    if config.vision_encoder is not None:
        _set_vision_encoder_sharding(config.vision_encoder)


def _set_vision_encoder_sharding(ve_cfg) -> None:
    """Invariant-activation TP plan for the MoonViT3d vision encoder.

    Linear layers are Colwise/Rowwise sharded for memory; norms and the
    learnable position table stay Invariant. ``patch_embed`` wraps the plain
    ``pixel_values`` input as a TP-invariant tensor so the rest of the encoder
    runs in distributed tensor space.
    """
    # The encoder's own ``pos_embed`` table is invariant across TP ranks.
    ve_cfg.sharding_config = ShardingConfig(
        state_shardings={
            "pos_embed": SpmdType({DP: spmd.R, CP: spmd.R, TP: spmd.I}),
        },
        out_src_shardings=SpmdType({DP: spmd.V, CP: spmd.R, TP: spmd.I}),
        out_dst_shardings=SpmdType({DP: spmd.V, CP: spmd.R, TP: spmd.R}),
    )
    ve_cfg.rotary_pos_emb.sharding_config = ShardingConfig(
        state_shardings={
            "inv_freq": SpmdType({DP: spmd.R, CP: spmd.R, TP: spmd.I}),
        },
        out_src_shardings=SpmdType({DP: spmd.R, CP: spmd.R, TP: spmd.I}),
    )

    ve_cfg.patch_embed_proj.sharding_config = vision_invariant_linear_config(
        include_cp_axis=True
    )

    set_vision_transformer_block_sharding_config(
        ve_cfg.block,
        rope_cache_dp=spmd.V,
        include_cp_axis=True,
    )

    # Final norm + projector.
    ve_cfg.final_norm.sharding_config = invariant_norm_config(include_cp_axis=True)
    proj = ve_cfg.projector
    proj.pre_norm.sharding_config = invariant_norm_config(include_cp_axis=True)
    proj.linear_1.sharding_config = vision_colwise_config(include_cp_axis=True)
    proj.linear_2.sharding_config = vision_scaled_bias_rowwise_config(
        include_cp_axis=True
    )
