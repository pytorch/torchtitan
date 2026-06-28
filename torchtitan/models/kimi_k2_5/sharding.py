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
- Vision encoder: activations flow ``Replicate`` (no SP -- the patch sequence is
  short, so sequence-sharding would add gather/scatter around the block-diagonal
  attention for little memory gain). Only the linear layers are Colwise/Rowwise
  sharded for memory; norms and position embeddings stay ``Replicate``.
"""

from typing import TYPE_CHECKING

import spmd_types as spmd

from torchtitan.models.common.decoder_sharding import (
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    rowwise_config,
    set_gqa_inner_attention_local_map,
)
from torchtitan.models.deepseek_v3.sharding import set_deepseek_v3_sharding_config
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig

if TYPE_CHECKING:
    from torchtitan.models.kimi_k2_5.model import KimiK25Model

_REPLICATE_PARAM = dense_param_placement(tp=spmd.R)
_REPLICATE_ACT = dense_activation_placement(tp=spmd.R)

_REPLICATE_NORM = ShardingConfig(
    state_shardings={"weight": _REPLICATE_PARAM, "bias": _REPLICATE_PARAM},
    in_src_shardings={"input": _REPLICATE_ACT},
    in_dst_shardings={"input": _REPLICATE_ACT},
    out_dst_shardings=_REPLICATE_ACT,
)


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
    needs the full (``Replicate``) embedding -- a ``Shard(1)`` one cannot be
    indexed by sequence position locally. Layer 0 then takes a ``Replicate``
    input and its rowwise ``wo`` reduce-scatters back to ``Shard(1)``, so the
    residual is sequence-parallel from layer 0's output and layers ``1..N-1``
    are unchanged full SP.
    """
    config.tok_embeddings.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
        in_src_shardings={"input": _REPLICATE_ACT},
        in_dst_shardings={"input": _REPLICATE_ACT},
        out_src_shardings=dense_activation_placement(tp=spmd.P),
        out_dst_shardings=_REPLICATE_ACT,
        local_map=LocalMapConfig(in_grad_placements=None),
    )

    layer0 = config.layers[0]
    layer0.attention_norm.sharding_config = ShardingConfig(
        state_shardings={"weight": _REPLICATE_PARAM},
        in_src_shardings={"input": _REPLICATE_ACT},
        out_src_shardings=_REPLICATE_ACT,
    )
    layer0.attention.sharding_config = ShardingConfig(
        in_src_shardings={"x": _REPLICATE_ACT},
        in_dst_shardings={"x": _REPLICATE_ACT},
    )


def _set_vision_encoder_sharding(ve_cfg) -> None:
    """Replicate-activation TP plan for the MoonViT3d vision encoder.

    Linear layers are Colwise/Rowwise sharded for memory; norms and the
    learnable position table are Replicate. ``patch_embed`` wraps the plain
    ``pixel_values`` input as ``DTensor(Replicate)`` so the rest of the encoder
    runs in DTensor space.
    """
    # The encoder's own ``pos_embed`` table is Replicate (F.interpolate runs on it).
    ve_cfg.sharding_config = ShardingConfig(
        state_shardings={"pos_embed": _REPLICATE_PARAM},
    )

    # patch_embed (Linear): receives plain pixel_values -> wrap as Replicate.
    ve_cfg.patch_embed_proj.sharding_config = ShardingConfig(
        state_shardings={"weight": _REPLICATE_PARAM, "bias": _REPLICATE_PARAM},
        in_src_shardings={"input": _REPLICATE_ACT},
        in_dst_shardings={"input": _REPLICATE_ACT},
        out_dst_shardings=_REPLICATE_ACT,
    )

    # Transformer block sub-modules (shared VisionTransformerBlock: norm1/norm2).
    block = ve_cfg.block
    block.norm1.sharding_config = _REPLICATE_NORM
    block.norm2.sharding_config = _REPLICATE_NORM

    # The stacked 2D rope_cache enters the attention as a plain (Replicate)
    # tensor input so it is DTensor-wrapped before meeting head-sharded q/k.
    block.attn.sharding_config = ShardingConfig(
        in_src_shardings={"rope_cache": _REPLICATE_ACT},
        in_dst_shardings={"rope_cache": _REPLICATE_ACT},
    )
    block.attn.wq.sharding_config = colwise_config()
    block.attn.wk.sharding_config = colwise_config()
    block.attn.wv.sharding_config = colwise_config()
    block.attn.proj.sharding_config = rowwise_config(output_sp=False)
    set_gqa_inner_attention_local_map(block.attn.inner_attention)

    block.mlp.fc1.sharding_config = colwise_config()
    block.mlp.fc2.sharding_config = rowwise_config(output_sp=False)

    # Final norm + projector.
    ve_cfg.final_norm.sharding_config = _REPLICATE_NORM
    proj = ve_cfg.projector
    proj.pre_norm.sharding_config = _REPLICATE_NORM
    proj.linear_1.sharding_config = colwise_config()
    proj.linear_2.sharding_config = rowwise_config(output_sp=False)
