# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config-based sharding for Kimi K2.5 (MoonViT3d + DeepSeekV3).

Sets ``ShardingConfig`` on every sub-config so ``model.parallelize()`` applies
TP/EP/SP uniformly via the Module protocol.

- Decoder (MLA + MoE): reuses ``set_deepseek_v3_sharding_config``. Multimodal
  configs keep the token embedding ``Replicate`` for the vision scatter and
  resume SP at layer 0 (see ``_shard_decoder_after_embedding_scatter``).
- Vision encoder: activations flow ``Invariant`` (no SP -- the patch sequence is
  short, so sequence-sharding would add gather/scatter around the block-diagonal
  attention for little memory gain). Only the linear layers are Colwise/Rowwise
  sharded for memory; norms and position embeddings stay ``Invariant``.
"""

from typing import TYPE_CHECKING

import spmd_types as spmd
from spmd_types import SpmdType

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    token_id_placement,
)
from torchtitan.models.common.vision_encoder_sharding import (
    invariant_norm_config,
    set_vision_transformer_block_sharding_config,
    vision_colwise_config,
    vision_invariant_linear_config,
    vision_scaled_bias_rowwise_config,
)
from torchtitan.models.deepseek_v3.sharding import set_deepseek_v3_sharding_config
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig

DP = MeshAxisName.DP
TP = MeshAxisName.TP

if TYPE_CHECKING:
    from torchtitan.models.kimi_k2_7.model import KimiK25Model

_REPLICATE_ACT = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))


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
        if enable_sp:
            _shard_decoder_after_embedding_scatter(config)
        _set_vision_encoder_sharding(config.vision_encoder)


def _shard_decoder_after_embedding_scatter(config: "KimiK25Model.Config") -> None:
    """Keep ``tok_embeddings`` ``Replicate`` and resume SP at layer 0's output.

    The vision scatter writes features at arbitrary sequence positions, so it
    needs the full (``Replicate``) embedding -- a ``Shard(0)`` one cannot be
    indexed by sequence position locally. Layer 0 then takes a ``Replicate``
    input and its rowwise ``wo`` reduce-scatters back to ``Shard(0)``, so the
    residual is sequence-parallel from layer 0's output and layers ``1..N-1``
    are unchanged full SP.
    """
    config.tok_embeddings.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
        in_src_shardings={"input": token_id_placement()},
        in_dst_shardings={"input": token_id_placement()},
        out_src_shardings=dense_activation_placement(tp=spmd.P, cp=spmd.S(0)),
        out_dst_shardings=_REPLICATE_ACT,
        local_map=LocalMapConfig(in_grad_placements=None),
    )

    layer0 = config.layers[0]
    layer0.sharding_config = ShardingConfig(
        in_src_shardings={"x": _REPLICATE_ACT},
        in_dst_shardings={"x": dense_sequence_parallel_placement()},
        out_src_shardings=dense_sequence_parallel_placement(),
    )


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
            "pos_embed": SpmdType({DP: spmd.R, TP: spmd.I}),
        },
        out_src_shardings=SpmdType({DP: spmd.V, TP: spmd.I}),
        out_dst_shardings=SpmdType({DP: spmd.V, TP: spmd.R}),
    )
    ve_cfg.rotary_pos_emb.sharding_config = ShardingConfig(
        state_shardings={
            "inv_freq": SpmdType({DP: spmd.R, TP: spmd.I}),
        },
        out_src_shardings=SpmdType({DP: spmd.R, TP: spmd.I}),
    )

    ve_cfg.patch_embed_proj.sharding_config = vision_invariant_linear_config()

    set_vision_transformer_block_sharding_config(
        ve_cfg.block,
        rope_cache_dp=spmd.V,
    )

    # Final norm + projector.
    ve_cfg.final_norm.sharding_config = invariant_norm_config()
    proj = ve_cfg.projector
    proj.pre_norm.sharding_config = invariant_norm_config()
    proj.linear_1.sharding_config = vision_colwise_config()
    proj.linear_2.sharding_config = vision_scaled_bias_rowwise_config()
