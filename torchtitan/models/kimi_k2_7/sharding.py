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
import torch

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    set_gqa_inner_attention_local_map,
)
from torchtitan.models.deepseek_v3.sharding import set_deepseek_v3_sharding_config
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig, SpmdLayout

if TYPE_CHECKING:
    from torchtitan.models.kimi_k2_7.model import KimiK25Model

_REPLICATE_PARAM = dense_param_placement(tp=spmd.R)
_REPLICATE_ACT = dense_activation_placement(tp=spmd.R)

_VISION_INVARIANT_PARAM = dense_param_placement(tp=spmd.I)
_VISION_INVARIANT_ACT = dense_activation_placement(tp=spmd.I)

_VISION_INVARIANT_NORM = ShardingConfig(
    state_shardings={
        "weight": _VISION_INVARIANT_PARAM,
        "bias": _VISION_INVARIANT_PARAM,
    },
    in_src_shardings={"input": _VISION_INVARIANT_ACT},
    in_dst_shardings={"input": _VISION_INVARIANT_ACT},
    out_src_shardings=_VISION_INVARIANT_ACT,
    out_dst_shardings=_VISION_INVARIANT_ACT,
)


def _vision_colwise_config(
    *, input_tp: spmd.PerMeshAxisSpmdType = spmd.I
) -> ShardingConfig:
    input_layout = dense_activation_placement(tp=input_tp)
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.S(0)),
            "bias": dense_param_placement(tp=spmd.S(0)),
        },
        in_src_shardings={"input": input_layout},
        in_dst_shardings={"input": dense_activation_placement(tp=spmd.R)},
        out_src_shardings=dense_activation_placement(tp=spmd.S(-1)),
    )


def _vision_scaled_bias_rowwise_config() -> ShardingConfig:
    input_layout = dense_activation_placement(tp=spmd.S(2))
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.S(1)),
            "bias": dense_param_placement(tp=spmd.R),
        },
        in_src_shardings={"input": input_layout},
        in_dst_shardings={"input": input_layout},
        out_src_shardings=dense_activation_placement(tp=spmd.P),
        out_dst_shardings=dense_activation_placement(tp=spmd.I),
        local_map=LocalMapConfig(in_grad_placements=(input_layout,)),
    )


def annotate_multimodal_input_spmd_types(
    *,
    pixel_values: torch.Tensor | None,
    grid_thw: torch.Tensor | None,
    pixel_values_videos: torch.Tensor | None,
    grid_thw_videos: torch.Tensor | None,
) -> None:
    """Annotate Kimi K2.5 multimodal inputs with their local SPMD types."""
    multimodal_type = {
        MeshAxisName.DP: spmd.V,
        MeshAxisName.TP: spmd.I,
    }
    for tensor in (
        pixel_values,
        grid_thw,
        pixel_values_videos,
        grid_thw_videos,
    ):
        if tensor is not None:
            spmd.assert_type(tensor, multimodal_type)


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
    """Keep ``tok_embeddings`` ``Replicate`` and resume SP at layer 0's input.

    The vision scatter writes features at arbitrary sequence positions, so it
    needs the full (``Replicate``) embedding -- a ``Shard(1)`` one cannot be
    indexed by sequence position locally. Explicitly shard the completed
    multimodal embedding before layer 0 so its residual and attention output
    use the same sequence-parallel layout.
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
    sequence_parallel = dense_sequence_parallel_placement()
    # Enter standard SP before layer 0 instead of relying on DTensor to shard
    # only the residual at the first attention add.
    layer0.sharding_config = ShardingConfig(
        in_src_shardings={"x": _REPLICATE_ACT},
        in_dst_shardings={"x": sequence_parallel},
        out_src_shardings=sequence_parallel,
    )


def _set_vision_encoder_sharding(ve_cfg) -> None:
    """Invariant-activation TP plan for the MoonViT3d vision encoder.

    Linear layers are Colwise/Rowwise sharded for memory; norms and the
    learnable position table are Invariant. Colwise regions convert their
    Invariant input to Replicate, and matching Rowwise regions reduce Partial
    outputs back to Invariant.
    """
    ve_cfg.sharding_config = ShardingConfig(
        state_shardings={"pos_embed": _VISION_INVARIANT_PARAM},
        # The surrounding multimodal scatter operates on TP-replicated values.
        out_src_shardings=SpmdLayout(
            {MeshAxisName.DP: spmd.V, MeshAxisName.TP: spmd.I}
        ),
        out_dst_shardings=SpmdLayout(
            {MeshAxisName.DP: spmd.V, MeshAxisName.TP: spmd.R}
        ),
    )
    ve_cfg.rotary_pos_emb.sharding_config = ShardingConfig(
        state_shardings={"inv_freq": _VISION_INVARIANT_PARAM},
        out_src_shardings=_VISION_INVARIANT_PARAM,
    )

    ve_cfg.patch_embed_proj.sharding_config = ShardingConfig(
        state_shardings={
            "weight": _VISION_INVARIANT_PARAM,
            "bias": _VISION_INVARIANT_PARAM,
        },
        in_src_shardings={"input": _VISION_INVARIANT_ACT},
        in_dst_shardings={"input": _VISION_INVARIANT_ACT},
        out_src_shardings=_VISION_INVARIANT_ACT,
        out_dst_shardings=_VISION_INVARIANT_ACT,
    )

    # Transformer block sub-modules (shared VisionTransformerBlock: norm1/norm2).
    block = ve_cfg.block
    block.norm1.sharding_config = _VISION_INVARIANT_NORM
    block.norm2.sharding_config = _VISION_INVARIANT_NORM

    # Gather x and rope_cache at the attention boundary before head-sharded Q/K/V.
    block.attn.sharding_config = ShardingConfig(
        in_src_shardings={
            "x": _VISION_INVARIANT_ACT,
            "rope_cache": _VISION_INVARIANT_ACT,
        },
        in_dst_shardings={
            "x": _REPLICATE_ACT,
            "rope_cache": _REPLICATE_ACT,
        },
    )
    block.attn.wq.sharding_config = _vision_colwise_config(input_tp=spmd.R)
    block.attn.wk.sharding_config = _vision_colwise_config(input_tp=spmd.R)
    block.attn.wv.sharding_config = _vision_colwise_config(input_tp=spmd.R)
    block.attn.proj.sharding_config = _vision_scaled_bias_rowwise_config()
    set_gqa_inner_attention_local_map(block.attn.inner_attention)

    block.mlp.fc1.sharding_config = _vision_colwise_config()
    block.mlp.fc2.sharding_config = _vision_scaled_bias_rowwise_config()

    # Final norm + projector.
    ve_cfg.final_norm.sharding_config = _VISION_INVARIANT_NORM
    proj = ve_cfg.projector
    proj.pre_norm.sharding_config = _VISION_INVARIANT_NORM
    proj.linear_1.sharding_config = _vision_colwise_config()
    proj.linear_2.sharding_config = _vision_scaled_bias_rowwise_config()
