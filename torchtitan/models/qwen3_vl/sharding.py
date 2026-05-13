# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

from torch.distributed.tensor import Placement, Replicate, Shard

from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
    norm_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    set_gqa_attention_sharding,
    set_gqa_inner_attention_local_map,
)
from torchtitan.models.common.moe_sharding import set_moe_sharding_config
from torchtitan.protocols.sharding import ShardingConfig

if TYPE_CHECKING:
    from torchtitan.models.qwen3_vl.model import Qwen3VLModel


# Routed-expert layout (shared with qwen3 / llama4 / deepseek_v3).
_GROUPED_EXPERTS_PARAM_LAYOUT: dict[str, Placement] = {
    "w1": Shard(1),
    "w2": Shard(2),
    "w3": Shard(1),
}


def set_qwen3_vl_sharding_config(
    config: "Qwen3VLModel.Config",
    *,
    loss_parallel: bool,
    tp_enabled: bool,
    ep_enabled: bool,
) -> None:
    """Fill ``sharding_config`` on all Qwen3-VL sub-configs.

    Qwen3-VL runs WITHOUT SequenceParallel (``enable_sp=False``): the VLM
    needs full-sequence access in the decoder for vision scatter and
    DeepStack. Hidden states flow as DTensor(Replicate) on TP between
    blocks; only the per-block linears shard on TP and immediately
    redistribute back to Replicate.

    Vision encoder is sharded with the same Replicate-output discipline.
    """
    # Decoder dense (top-level: tok_embeddings, norm, lm_head).
    set_decoder_sharding_config(config, loss_parallel=loss_parallel, enable_sp=False)
    for layer_cfg in config.layers:
        _set_qwen3_vl_layer_sharding(
            layer_cfg,
            tp_enabled=tp_enabled,
            ep_enabled=ep_enabled,
        )

    # Vision encoder.
    if config.vision_encoder is not None:
        _set_vision_encoder_sharding(config.vision_encoder)


def _set_qwen3_vl_layer_sharding(
    layer_cfg,
    *,
    tp_enabled: bool,
    ep_enabled: bool,
) -> None:
    """Set sharding on one Qwen3-VL decoder layer.

    Mirrors qwen3 with enable_sp=False.
    """
    attention = layer_cfg.attention
    assert isinstance(attention, GQAttention.Config)

    norm = norm_config(enable_sp=False)
    layer_cfg.attention_norm.sharding_config = norm
    layer_cfg.ffn_norm.sharding_config = norm

    set_gqa_attention_sharding(attention, enable_sp=False)
    set_gqa_inner_attention_local_map(attention.inner_attention)

    # QK norms: shard on head dim (dim=2) -- independent of SP.
    if attention.qk_norm is not None:
        attention.qk_norm.sharding_config = ShardingConfig(
            state_shardings={"weight": dense_param_placement(tp=Replicate())},
            in_src_shardings={"input": dense_activation_placement(tp=Shard(2))},
            in_dst_shardings={"input": dense_activation_placement(tp=Shard(2))},
            out_dst_shardings=dense_activation_placement(tp=Shard(2)),
        )

    # Dense FFN (non-MoE layers only).
    if layer_cfg.feed_forward is not None:
        set_dense_ffn_sharding(
            layer_cfg.feed_forward,
            attn_x_placement=Replicate(),
            enable_sp=False,
        )

    # MoE FFN (MoE-enabled layers only).
    if layer_cfg.moe is not None and (tp_enabled or ep_enabled):
        set_moe_sharding_config(
            layer_cfg.moe,
            tp_enabled=tp_enabled,
            ep_enabled=ep_enabled,
            enable_sp=False,
            expert_param_layout=_GROUPED_EXPERTS_PARAM_LAYOUT,
        )


def _set_vision_encoder_sharding(ve_cfg) -> None:
    """Set sharding on the Qwen3-VL vision encoder.

    Vision activations flow as DTensor(Replicate) throughout. Linears
    shard on TP for memory but immediately redistribute back to
    Replicate on output (matching the legacy ``_apply_tp_to_vision_encoder``
    pattern with ``use_local_output=False``).

    ``pos_embed`` is a root-level parameter on the encoder; declared via
    ``state_shardings`` on the encoder root.

    LayerNorm / GELU classes from ``models.common.nn_modules`` carry
    sharding plans via their auto-generated ``Config`` slot.
    """
    # Encoder root: pos_embed is a Replicate parameter on TP.
    ve_cfg.sharding_config = ShardingConfig(
        state_shardings={"pos_embed": dense_param_placement(tp=Replicate())},
    )

    # patch_embed: weight stays Replicate; LinearConfig is colwise-style
    # but vision encoder uses replicate-output everywhere.
    ve_cfg.patch_embed.proj.sharding_config = _replicate_linear_config()

    # Vision blocks: norm1/norm2 (LayerNorm.Config), attn.qkv (colwise
    # with Replicate output), attn.proj (rowwise with Replicate output),
    # mlp.linear_fc1 (colwise replicate), mlp.linear_fc2 (rowwise replicate).
    block_norm = norm_config(enable_sp=False)
    for layer_cfg in (
        ve_cfg.layers.values() if hasattr(ve_cfg.layers, "values") else ve_cfg.layers
    ):
        layer_cfg.norm1.sharding_config = block_norm
        layer_cfg.norm2.sharding_config = block_norm
        layer_cfg.attn.qkv.sharding_config = _vision_colwise_config()
        layer_cfg.attn.proj.sharding_config = _replicate_output_rowwise_config()
        layer_cfg.mlp.fc1.sharding_config = _vision_colwise_config()
        layer_cfg.mlp.fc2.sharding_config = _replicate_output_rowwise_config()

    # Mergers (main + deepstack): same plan as vision MLP.
    _set_merger_sharding(ve_cfg.merger)
    for merger_cfg in (
        ve_cfg.deepstack_merger_list if hasattr(ve_cfg, "deepstack_merger_list") else []
    ):
        _set_merger_sharding(merger_cfg)


def _set_merger_sharding(merger_cfg) -> None:
    merger_cfg.norm.sharding_config = norm_config(enable_sp=False)
    merger_cfg.fc1.sharding_config = _vision_colwise_config()
    merger_cfg.fc2.sharding_config = _replicate_output_rowwise_config()


def _replicate_linear_config() -> ShardingConfig:
    """Linear with all-Replicate weight on TP, used for patch_embed.proj."""
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=Replicate()),
            "bias": dense_param_placement(tp=Replicate()),
        },
        in_src_shardings={"input": dense_activation_placement(tp=Replicate())},
        in_dst_shardings={"input": dense_activation_placement(tp=Replicate())},
        out_dst_shardings=dense_activation_placement(tp=Replicate()),
    )


def _vision_colwise_config() -> ShardingConfig:
    """Colwise weight with Replicate input/output (vision encoder convention).

    Differs from ``decoder_sharding.colwise_config`` (which has
    Shard(-1) output) -- vision encoder keeps activations Replicate
    throughout because there's no SP and the residual adds expect
    Replicate.
    """
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=Shard(0)),
            "bias": dense_param_placement(tp=Shard(0)),
        },
        in_src_shardings={"input": dense_activation_placement(tp=Replicate())},
        in_dst_shardings={"input": dense_activation_placement(tp=Replicate())},
        out_dst_shardings=dense_activation_placement(tp=Replicate()),
    )


def _replicate_output_rowwise_config() -> ShardingConfig:
    """Rowwise weight with Replicate input expected (Shard(-1)) and Replicate output."""
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=Shard(1)),
            "bias": dense_param_placement(tp=Replicate()),
        },
        in_src_shardings={"input": dense_activation_placement(tp=Shard(1))},
        in_dst_shardings={"input": dense_activation_placement(tp=Shard(1))},
        out_dst_shardings=dense_activation_placement(tp=Replicate()),
    )
