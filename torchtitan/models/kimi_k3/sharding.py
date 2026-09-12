# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding configs for Kimi K3. Same pattern as ``qwen3_5/sharding.py``.

Declarations only: functions here set ``ShardingConfig`` on sub-configs of an
already-built config tree, and ``model.parallelize()`` applies them through the
Module protocol. Nothing here touches a mesh or a device.
"""

from typing import TYPE_CHECKING

import spmd_types as spmd
from spmd_types import SpmdType

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.decoder_sharding import (
    attention_activation_placement,
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    norm_config,
    rowwise_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    set_gqa_inner_attention_local_spmd,
    token_id_placement,
)
from torchtitan.models.common.moe_sharding import set_moe_sharding_config
from torchtitan.models.kimi_k2_7.sharding import set_moonvit_sharding_config
from torchtitan.protocols.sharding import ShardingConfig

if TYPE_CHECKING:
    from torchtitan.models.kimi_k3.kda import KDA
    from torchtitan.models.kimi_k3.model import (
        KimiK3Model,
        KimiK3TransformerBlock,
        KimiMLAAttention,
    )
    from torchtitan.models.kimi_k3.moe import KimiLatentMoE


DP = MeshAxisName.DP
TP = MeshAxisName.TP

_GROUPED_EXPERTS_PARAM_LAYOUT: dict[str, spmd.PerMeshAxisSpmdType] = {
    "w1_EFD": spmd.S(1),
    "w2_EDF": spmd.S(2),
    "w3_EFD": spmd.S(1),
}


def set_kimi_k3_sharding_config(
    config: "KimiK3Model.Config",
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    """Fill ``sharding_config`` on all Kimi K3 sub-configs.

    Populated unconditionally, as in DeepSeek V3: ``Module.parallelize``
    filters disabled axes at runtime.
    """
    set_decoder_sharding_config(config, enable_sp=enable_sp)
    layer_input_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
    )
    if config.vision_encoder is not None:
        _shard_decoder_after_embedding_scatter(
            config, layer_input_layout, enable_sp=enable_sp
        )
        set_moonvit_sharding_config(config.vision_encoder, projector_norm="post_norm")
    config.output_res_norm.sharding_config = _stream_weight_config(enable_sp=enable_sp)
    config.output_res_proj.sharding_config = _stream_weight_config(enable_sp=enable_sp)
    for layer_cfg in config.layers:
        _set_kimi_k3_layer_sharding(
            layer_cfg,
            attn_x_layout=layer_input_layout,
            enable_sp=enable_sp,
            enable_ep=enable_ep,
        )


def _set_kimi_k3_layer_sharding(
    layer_cfg: "KimiK3TransformerBlock.Config",
    *,
    attn_x_layout: SpmdType,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    """Set sharding on one Kimi K3 layer: MLA or KDA, then dense FFN or latent MoE."""
    norm = norm_config(enable_sp=enable_sp)
    layer_cfg.attention_norm.sharding_config = norm
    layer_cfg.ffn_norm.sharding_config = norm
    # Layer 0 has no attention residual.
    for res_cfg in (
        layer_cfg.attention_res_norm,
        layer_cfg.attention_res_proj,
        layer_cfg.ffn_res_norm,
        layer_cfg.ffn_res_proj,
    ):
        if res_cfg is not None:
            res_cfg.sharding_config = _stream_weight_config(enable_sp=enable_sp)

    if layer_cfg.attention is not None:
        _set_mla_sharding(
            layer_cfg.attention, attn_x_layout=attn_x_layout, enable_sp=enable_sp
        )
    else:
        assert layer_cfg.delta_attention is not None
        _set_kda_sharding(
            layer_cfg.delta_attention, attn_x_layout=attn_x_layout, enable_sp=enable_sp
        )

    if layer_cfg.feed_forward is not None:
        set_dense_ffn_sharding(
            layer_cfg.feed_forward, attn_x_layout=attn_x_layout, enable_sp=enable_sp
        )
    else:
        assert layer_cfg.moe is not None
        _set_latent_moe_sharding(
            layer_cfg.moe, enable_sp=enable_sp, enable_ep=enable_ep
        )


def _set_mla_sharding(
    attention_cfg: "KimiMLAAttention.Config",
    *,
    attn_x_layout: SpmdType,
    enable_sp: bool,
) -> None:
    """DeepSeek V3's MLA plan, without RoPE and with a colwise output ``gate``."""
    attention_cfg.sharding_config = ShardingConfig(
        in_src_shardings={"x_TD": attn_x_layout},
        in_dst_shardings={"x_TD": dense_activation_placement(tp=spmd.R, cp=spmd.S(0))},
    )
    replicate_weight = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)},
    )
    attention_cfg.wq_a.sharding_config = replicate_weight
    attention_cfg.q_norm.sharding_config = replicate_weight
    attention_cfg.wkv_a.sharding_config = replicate_weight
    attention_cfg.kv_norm.sharding_config = replicate_weight
    attention_cfg.wq_b.sharding_config = colwise_config()
    attention_cfg.wkv_b.sharding_config = colwise_config()
    attention_cfg.gate.sharding_config = colwise_config()
    attention_cfg.wo.sharding_config = rowwise_config(output_sp=enable_sp)
    set_gqa_inner_attention_local_spmd(attention_cfg.inner_attention)


def _set_kda_sharding(
    kda_cfg: "KDA.Config",
    *,
    attn_x_layout: SpmdType,
    enable_sp: bool,
) -> None:
    """Head-sharded TP for KDA, as Qwen3.5's GatedDeltaNet; low-rank ``forget_a`` is
    replicated.
    """
    for name in ("q_proj", "k_proj", "v_proj", "forget_b", "beta", "output_gate"):
        getattr(kda_cfg, name).sharding_config = colwise_config()
    replicate_weight = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)},
    )
    kda_cfg.forget_a.sharding_config = replicate_weight
    kda_cfg.output_norm.sharding_config = replicate_weight
    kda_cfg.output_proj.sharding_config = rowwise_config(output_sp=enable_sp)

    projected_placement = dense_activation_placement(tp=spmd.S(1), cp=spmd.S(0))
    head_placement = attention_activation_placement()
    parameter_placement = dense_param_placement(tp=spmd.S(0))
    for name in ("q_conv", "k_conv", "v_conv"):
        getattr(kda_cfg, name).sharding_config = ShardingConfig(
            state_shardings={"weight": parameter_placement},
        )

    # The kernel runs on the rank-local heads in one local SPMD region.
    kernel_inputs = {
        "query_TC": projected_placement,
        "key_TC": projected_placement,
        "value_TC": projected_placement,
        "raw_gate_THK": head_placement,
        "raw_beta_TH": projected_placement,
        "conv_q_weight_C1W": parameter_placement,
        "conv_k_weight_C1W": parameter_placement,
        "conv_v_weight_C1W": parameter_placement,
        "A_log_H": parameter_placement,
        "dt_bias_HK": parameter_placement,
    }
    kda_cfg.inner_kda.sharding_config = ShardingConfig(
        in_src_shardings=kernel_inputs,
        in_dst_shardings=kernel_inputs,
        out_src_shardings=head_placement,
        out_dst_shardings=head_placement,
        local_spmd=True,
    )

    kda_cfg.sharding_config = ShardingConfig(
        state_shardings={
            "A_log": parameter_placement,
            "dt_bias": parameter_placement,
        },
        in_src_shardings={"x_TD": attn_x_layout},
        in_dst_shardings={"x_TD": dense_activation_placement(tp=spmd.R, cp=spmd.S(0))},
    )


def _set_latent_moe_sharding(
    moe_cfg: "KimiLatentMoE.Config",
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    """Core's MoE plan, plus the latent projections around the routed experts."""
    set_moe_sharding_config(
        moe_cfg,
        enable_ep=enable_ep,
        enable_sp=enable_sp,
        expert_param_layout=_GROUPED_EXPERTS_PARAM_LAYOUT,
    )
    # Replicated when the experts are TP-sharded; under EP they are whole on
    # every tp rank, so routed_down follows the stream's rule.
    moe_cfg.routed_down.sharding_config = (
        _stream_weight_config(enable_sp=enable_sp)
        if enable_ep
        else ShardingConfig(
            state_shardings={"weight": dense_param_placement(tp=spmd.R)}
        )
    )
    routed_norm = norm_config(enable_sp=enable_sp)
    routed_up = _stream_weight_config(enable_sp=enable_sp)
    if not enable_sp:
        # The experts' Partial output is reduced at the norm's boundary;
        # routed_up re-enters Partial so the MoE exit reduces it once.
        routed_norm.in_src_shardings = {
            "x": dense_activation_placement(tp=spmd.P, cp=spmd.S(0))
        }
        routed_norm.in_dst_shardings = {
            "x": dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
        }
        routed_up.out_src_shardings = dense_activation_placement(
            tp=spmd.I, cp=spmd.S(0)
        )
        routed_up.out_dst_shardings = dense_activation_placement(
            tp=spmd.P, cp=spmd.S(0)
        )
    moe_cfg.routed_norm.sharding_config = routed_norm
    moe_cfg.routed_up.sharding_config = routed_up


def _stream_weight_config(*, enable_sp: bool) -> ShardingConfig:
    """Weight on the token stream, state only, with the norms' TP rule."""
    return ShardingConfig(
        state_shardings=norm_config(enable_sp=enable_sp).state_shardings
    )


def _block_residual_placement(*, tp: spmd.PerMeshAxisSpmdType) -> SpmdType:
    """Placement of the ``(tokens, entries, hidden)`` block-residual stack."""
    if isinstance(tp, spmd.Shard):
        return SpmdType(
            {DP: spmd.V, TP: spmd.V},
            partition_spec=spmd.PartitionSpec((DP, TP), None, None),
        )
    return SpmdType(
        {DP: spmd.V, TP: tp}, partition_spec=spmd.PartitionSpec(DP, None, None)
    )


def _shard_decoder_after_embedding_scatter(
    config: "KimiK3Model.Config", layer_input_layout: SpmdType, *, enable_sp: bool
) -> None:
    """Keep ``tok_embeddings`` TP-replicated for the vision scatter; layer 0's
    input boundary restores the decoder's layout for the stream and the stack.
    """
    replicated = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    config.tok_embeddings.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
        in_src_shardings={"input": token_id_placement()},
        in_dst_shardings={"input": token_id_placement()},
        out_src_shardings=dense_activation_placement(tp=spmd.P, cp=spmd.S(0)),
        out_dst_shardings=replicated,
        local_spmd=True,
    )
    config.layers[0].sharding_config = ShardingConfig(
        in_src_shardings={
            "x_TD": replicated,
            "block_residual_TND": _block_residual_placement(tp=spmd.R),
        },
        in_dst_shardings={
            "x_TD": layer_input_layout,
            "block_residual_TND": _block_residual_placement(
                tp=spmd.S(0) if enable_sp else spmd.I
            ),
        },
    )
