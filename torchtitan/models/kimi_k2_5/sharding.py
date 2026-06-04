# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config-based sharding for Kimi K2.5 (MoonViT3d + DeepSeekV3).

Sets ``ShardingConfig`` on every sub-config so ``model.parallelize()`` applies
TP/EP uniformly via the Module protocol — the same approach as qwen3_5.

- Decoder (MLA + MoE): delegates to ``set_deepseek_v3_sharding_config`` with
  ``enable_sp=False``. Sequence parallelism is intentionally disabled so the
  decoder hidden states stay ``Replicate`` (full sequence), which the
  multimodal forward needs to scatter vision features at placeholder positions.
- Vision encoder: activations flow ``Replicate``; only the linear layers are
  Colwise/Rowwise sharded for memory. Norms and position embeddings stay
  ``Replicate``.
"""

from typing import TYPE_CHECKING

from torch.distributed.tensor import Replicate

from torchtitan.models.common.decoder_sharding import (
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    rowwise_config,
    set_gqa_inner_attention_local_map,
)
from torchtitan.models.deepseek_v3.sharding import set_deepseek_v3_sharding_config
from torchtitan.protocols.sharding import ShardingConfig

if TYPE_CHECKING:
    from torchtitan.models.kimi_k2_5.model import KimiK25Model

_REPLICATE_PARAM = dense_param_placement(tp=Replicate())
_REPLICATE_ACT = dense_activation_placement(tp=Replicate())

# Norm / module that receives and emits Replicate activations.
_REPLICATE_NORM = ShardingConfig(
    state_shardings={"weight": _REPLICATE_PARAM, "bias": _REPLICATE_PARAM},
    in_src_shardings={"input": _REPLICATE_ACT},
    in_dst_shardings={"input": _REPLICATE_ACT},
    out_dst_shardings=_REPLICATE_ACT,
)


def set_kimi_k2_5_sharding_config(
    config: "KimiK25Model.Config",
    *,
    loss_parallel: bool,
    enable_ep: bool,
) -> None:
    """Fill ``sharding_config`` on all Kimi K2.5 sub-configs.

    The decoder reuses DeepSeek V3's sharding (with ``enable_sp=False``); the
    vision encoder gets its own Replicate-activation TP plan.
    """
    set_deepseek_v3_sharding_config(
        config,
        loss_parallel=loss_parallel,
        enable_sp=False,
        enable_ep=enable_ep,
    )
    _set_vision_encoder_sharding(config.vision_encoder)


def _set_vision_encoder_sharding(ve_cfg) -> None:
    """Replicate-activation TP plan for the MoonViT3d vision encoder.

    Linear layers are Colwise/Rowwise sharded for memory; norms and position
    embeddings are Replicate. ``patch_embed`` wraps the plain ``pixel_values``
    input as ``DTensor(Replicate)`` so the rest of the encoder runs in DTensor
    space.
    """
    # patch_embed (Linear): receives plain pixel_values -> wrap as Replicate.
    ve_cfg.patch_embed_proj.sharding_config = ShardingConfig(
        state_shardings={"weight": _REPLICATE_PARAM, "bias": _REPLICATE_PARAM},
        in_src_shardings={"input": _REPLICATE_ACT},
        in_dst_shardings={"input": _REPLICATE_ACT},
        out_dst_shardings=_REPLICATE_ACT,
    )

    # Learnable position embedding: Replicate weight + temporal buffer.
    # F.interpolate runs on the Replicate weight (same as qwen3_vl/qwen3_5).
    ve_cfg.pos_emb.sharding_config = ShardingConfig(
        state_shardings={
            "weight": _REPLICATE_PARAM,
            "time_weight": _REPLICATE_PARAM,
        },
    )

    # Transformer block sub-modules.
    block = ve_cfg.block
    block.norm0.sharding_config = _REPLICATE_NORM
    block.norm1.sharding_config = _REPLICATE_NORM

    # rope_cos / rope_sin enter the attention as plain (Replicate) tensors.
    block.attn.sharding_config = ShardingConfig(
        in_src_shardings={"rope_cos": _REPLICATE_ACT, "rope_sin": _REPLICATE_ACT},
        in_dst_shardings={"rope_cos": _REPLICATE_ACT, "rope_sin": _REPLICATE_ACT},
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
