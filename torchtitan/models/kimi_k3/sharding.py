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
    set_gqa_inner_attention_local_map,
)
from torchtitan.models.common.moe_sharding import set_moe_sharding_config
from torchtitan.models.common.vision_encoder_sharding import (
    invariant_norm_config,
    set_vision_transformer_block_sharding_config,
    vision_invariant_linear_config,
)
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig

if TYPE_CHECKING:
    from torchtitan.models.kimi_k3.model import KimiK3Model

DP = MeshAxisName.DP
TP = MeshAxisName.TP
CP = MeshAxisName.CP


def set_kimi_k3_sharding_config(
    config: "KimiK3Model.Config", *, enable_ep: bool, enable_sp: bool = False
) -> None:
    """Declare the sharding expert parallel acts on.

    The routed experts shard on the expert axis; ``set_moe_sharding_config``
    declares that layout, and its input boundary lifts the plain incoming
    activations itself, so no decoder-level declaration is needed.
    """
    for layer in config.layers:
        if layer.moe is not None:
            set_moe_sharding_config(
                layer.moe,
                enable_ep=enable_ep,
                # TODO: flip to True from the caller once the
                # tensor-parallel PR lands; with EP alone the internals run
                # without sequence parallel.
                enable_sp=enable_sp,
                expert_param_layout={
                    "w1_EFD": spmd.S(1),
                    "w2_EDF": spmd.S(2),
                    "w3_EFD": spmd.S(1),
                },
            )


def _stream_param_config(*, enable_sp: bool) -> ShardingConfig:
    """Weight that reads and feeds the token stream between the TP modules.

    Without SP the stream is invariant on TP and every module converts it
    I -> R on entry, so the gradient reaching this weight is identical on every
    rank: invariant, where replicated would sum the copies. Under SP the stream
    is the sequence shard and the gradient a per-rank partial: replicated, and
    FSDP sums it. The rule ``norm_config`` applies to the norms on that stream.
    """
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.R if enable_sp else spmd.I)
        }
    )


def _tp_replicate_config() -> ShardingConfig:
    """Weight replicated on the TP axis, with no activation boundary declared.

    The replicated member of the colwise/rowwise family, which core does not
    have: declaring the activation boundaries would lift the input to a
    DTensor while ``Linear.forward`` unwraps its own weight to local.
    """
    return ShardingConfig(state_shardings={"weight": dense_param_placement(tp=spmd.R)})


def _set_mla_sharding(
    attention_cfg, *, enable_sp: bool, invariant_stream: bool = False
) -> None:
    """Head-parallel TP for MLA.

    The projections that produce or consume the head axis split on it; the
    two compressions stay whole because they are rank-sized, not head-sized.
    """
    attention_cfg.wq_b.sharding_config = colwise_config()
    attention_cfg.wkv_b.sharding_config = colwise_config()
    attention_cfg.gate.sharding_config = colwise_config()
    attention_cfg.wo.sharding_config = rowwise_config(output_sp=enable_sp)
    if enable_sp:
        # The module boundary gathers the sequence shard on the way in -- the
        # attention core needs the full sequence -- and wo reduce-scatters
        # back to Shard(0), the GQA pattern.
        attention_cfg.sharding_config = ShardingConfig(
            in_src_shardings={"x_TD": dense_sequence_parallel_placement()},
            in_dst_shardings={
                "x_TD": dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
            },
        )
    if invariant_stream and not enable_sp:
        # Under spmd_types the block stream is invariant on TP while the
        # attention body is replicated with sharded heads: entering converts
        # I -> R (no-op forward, all-reduce of the input gradient in backward,
        # since wq_b/wkv_b are colwise); wo's rowwise boundary hands the
        # stream back invariant.
        attention_cfg.sharding_config = ShardingConfig(
            in_src_shardings={
                "x_TD": dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
            },
            in_dst_shardings={
                "x_TD": dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
            },
        )
    attention_cfg.wq_a.sharding_config = _tp_replicate_config()
    attention_cfg.wkv_a.sharding_config = _tp_replicate_config()
    # Inside the replicated body the activations are R and the norms feed the
    # colwise wq_b/wkv_b, so their weights take partial gradients: replicated,
    # state only (norm_config's invariant [T, D] boundary is the stream's).
    for name in ("q_norm", "kv_norm"):
        getattr(attention_cfg, name).sharding_config = ShardingConfig(
            state_shardings={"weight": dense_param_placement(tp=spmd.R)}
        )
    # An identity boundary on the cp axis: a CP kernel issues its own exchange.
    set_gqa_inner_attention_local_map(attention_cfg.inner_attention)


def _set_kda_sharding(
    delta_attention_cfg, *, enable_sp: bool, invariant_stream: bool = False
) -> None:
    """Head-parallel TP for KDA.

    The delta rule is independent per head, so the projections that produce
    or consume the head axis split on it, the per-head state (``A_log``,
    ``dt_bias``, the depthwise convolutions) shards with the heads, and the
    kernel runs on the local heads behind a ``local_map`` on ``inner_kda``.
    The one low-rank compression, ``forget_a``, is rank-sized and stays whole.
    """
    for name in ("q_proj", "k_proj", "v_proj", "forget_b", "beta", "output_gate"):
        getattr(delta_attention_cfg, name).sharding_config = colwise_config()
    delta_attention_cfg.forget_a.sharding_config = _tp_replicate_config()
    delta_attention_cfg.output_proj.sharding_config = rowwise_config(
        output_sp=enable_sp
    )
    head_param = dense_param_placement(tp=spmd.S(0))
    for name in ("q_conv", "k_conv", "v_conv"):
        getattr(delta_attention_cfg, name).sharding_config = ShardingConfig(
            state_shardings={"weight": head_param}
        )
    delta_attention_cfg.output_norm.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)}
    )
    kda_module_config = ShardingConfig(
        state_shardings={"A_log": head_param, "dt_bias": head_param}
    )
    if enable_sp:
        # Every projection reads the stream, so the module boundary gathers
        # the sequence shard once; output_proj reduce-scatters back.
        kda_module_config.in_src_shardings = {
            "x_TD": dense_sequence_parallel_placement()
        }
        kda_module_config.in_dst_shardings = {
            "x_TD": dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
        }
    if invariant_stream and not enable_sp:
        # The MLA boundary's twin: the invariant stream enters replicated (the
        # projections are colwise, so the backward all-reduces the input
        # gradient) and output_proj's rowwise exit hands it back invariant.
        kda_module_config.in_src_shardings = {
            "x_TD": dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
        }
        kda_module_config.in_dst_shardings = {
            "x_TD": dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
        }
    delta_attention_cfg.sharding_config = kda_module_config
    features = dense_activation_placement(tp=spmd.S(-1), cp=spmd.S(0))
    heads = attention_activation_placement()
    inputs = {
        "query_TC": features,
        "key_TC": features,
        "value_TC": features,
        "raw_gate_THK": heads,
        "raw_beta_TH": features,
        "conv_q_weight_C1W": head_param,
        "conv_k_weight_C1W": head_param,
        "conv_v_weight_C1W": head_param,
        "A_log_H": head_param,
        "dt_bias_HK": head_param,
        # Packed-document boundaries, the same on every tp rank (None under flex).
        "cu_seqlens": dense_activation_placement(tp=spmd.I, cp=spmd.V),
    }
    delta_attention_cfg.inner_kda.sharding_config = ShardingConfig(
        in_src_shardings=inputs,
        in_dst_shardings=inputs,
        out_src_shardings=heads,
        local_map=LocalMapConfig(in_grad_placements=tuple(inputs.values())),
    )


def set_tensor_parallel_sharding_config(
    config: "KimiK3Model.Config",
    *,
    enable_sp: bool = False,
    declare_vision_encoder: bool = False,
    spmd_types: bool = False,
) -> None:
    """Declare the sharding tensor parallel acts on.

    Head and feature axes shard. With ``enable_sp`` the token stream between
    modules carries the TP-axis Shard(0) of sequence parallel: norms compute
    on the shard, the attention module boundaries gather it (the cores need
    the full sequence) and the rowwise outputs reduce-scatter back, the
    llama3 template; without it the stream stays whole on the TP axis. The
    MoE internals are declared by ``set_kimi_k3_sharding_config``; this adds
    the latent projections around them.
    """
    set_decoder_sharding_config(config, enable_sp=enable_sp)
    if declare_vision_encoder and config.vision_encoder is not None:
        # Under spmd_types every parameter needs a layout. Under partial_dtensor
        # the tower stays undeclared: MoonViT's position lookup indexes the
        # table with a plain tensor, which a DTensor table refuses.
        _set_vision_encoder_sharding(config.vision_encoder, enable_sp=enable_sp)
    config.output_res_norm.sharding_config = norm_config(enable_sp=enable_sp)
    config.output_res_proj.sharding_config = _stream_param_config(enable_sp=enable_sp)
    attn_x_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
    )
    for layer in config.layers:
        for name in (
            "attention_norm",
            "ffn_norm",
            "attention_res_norm",
            "ffn_res_norm",
        ):
            cfg = getattr(layer, name, None)
            if cfg is not None:
                cfg.sharding_config = norm_config(enable_sp=enable_sp)
        for name in ("attention_res_proj", "ffn_res_proj"):
            cfg = getattr(layer, name, None)
            if cfg is not None:
                cfg.sharding_config = _stream_param_config(enable_sp=enable_sp)
        if layer.attention is not None:
            _set_mla_sharding(
                layer.attention, enable_sp=enable_sp, invariant_stream=spmd_types
            )
        if layer.delta_attention is not None:
            _set_kda_sharding(
                layer.delta_attention, enable_sp=enable_sp, invariant_stream=spmd_types
            )
        if layer.feed_forward is not None:
            set_dense_ffn_sharding(
                layer.feed_forward, attn_x_layout=attn_x_layout, enable_sp=enable_sp
            )
        if layer.moe is not None:
            # routed_down runs on the stream the MoE boundary gathered; the
            # experts hand their output back sequence-sharded under SP, so the
            # norm after them and routed_up run on the shard.
            # routed_down feeds the TP-sharded experts (partial gradients:
            # replicated); routed_up feeds the stream from the reduced norm
            # output, so it follows the stream's rule.
            layer.moe.routed_down.sharding_config = _tp_replicate_config()
            routed_up_cfg = _stream_param_config(enable_sp=enable_sp)
            routed_norm_cfg = norm_config(enable_sp=enable_sp)
            if spmd_types and not enable_sp:
                # The experts hand the norm their rowwise output, Partial on
                # TP, and nothing between reduces it under spmd_types
                # (partial_dtensor's DTensor did so implicitly): the norm's
                # boundary reduces, keyed "x", the argument nn.RMSNorm.forward
                # takes. routed_up re-enters the Partial domain so its sum
                # with the shared experts' Partial output types and core's MoE
                # exit reduces once for both; that exit then returns to the
                # invariant stream.
                routed_norm_cfg.in_src_shardings = {
                    "x": dense_activation_placement(tp=spmd.P, cp=spmd.S(0))
                }
                routed_norm_cfg.in_dst_shardings = {
                    "x": dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
                }
                routed_up_cfg.out_src_shardings = dense_activation_placement(
                    tp=spmd.I, cp=spmd.S(0)
                )
                routed_up_cfg.out_dst_shardings = dense_activation_placement(
                    tp=spmd.P, cp=spmd.S(0)
                )
                if layer.moe.sharding_config is not None:
                    layer.moe.sharding_config.out_dst_shardings = (
                        dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
                    )
            layer.moe.routed_up.sharding_config = routed_up_cfg
            layer.moe.routed_norm.sharding_config = routed_norm_cfg


def _set_vision_encoder_sharding(ve_cfg, *, enable_sp: bool) -> None:
    """Invariant plan for the MoonViT tower, the kimi_k2_7 shape.

    The tower runs whole on every rank, as under partial_dtensor: every linear
    invariant at TP and the attention rank-local over cp. K3's projector norms
    after its second linear.
    """
    # The exit follows the stream the features splice into: invariant without
    # SP; replicated under SP, where the gathered shard is, and the I -> R
    # exit's backward all-reduce sums the shards' feature gradients into the
    # one gradient the tower's invariant weights expect.
    ve_cfg.sharding_config = ShardingConfig(
        state_shardings={"pos_embed": SpmdType({DP: spmd.R, CP: spmd.R, TP: spmd.I})},
        out_src_shardings=SpmdType({DP: spmd.V, CP: spmd.V, TP: spmd.I}),
        out_dst_shardings=SpmdType(
            {DP: spmd.V, CP: spmd.V, TP: spmd.R if enable_sp else spmd.I}
        ),
    )
    ve_cfg.rotary_pos_emb.sharding_config = ShardingConfig(
        state_shardings={"inv_freq": SpmdType({DP: spmd.R, CP: spmd.R, TP: spmd.I})},
        out_src_shardings=SpmdType({DP: spmd.R, CP: spmd.R, TP: spmd.I}),
    )
    ve_cfg.patch_embed_proj.sharding_config = vision_invariant_linear_config(
        include_cp_axis=True
    )
    set_vision_transformer_block_sharding_config(
        ve_cfg.block, rope_cache_dp=spmd.V, include_cp_axis=True
    )
    block = ve_cfg.block
    for linear in (
        block.attn.wq,
        block.attn.wk,
        block.attn.wv,
        block.attn.proj,
        block.mlp.fc1,
        block.mlp.fc2,
    ):
        linear.sharding_config = vision_invariant_linear_config(include_cp_axis=True)
    invariant_stream = SpmdType({DP: spmd.V, CP: spmd.V, TP: spmd.I})
    block.attn.sharding_config = ShardingConfig(
        in_src_shardings={"x": invariant_stream, "rope_cache": invariant_stream},
        in_dst_shardings={"x": invariant_stream, "rope_cache": invariant_stream},
    )
    block.attn.inner_attention.sharding_config = ShardingConfig(
        in_src_shardings={
            "q_THK": invariant_stream,
            "k_THK": invariant_stream,
            "v_THV": invariant_stream,
        },
        in_dst_shardings={
            "q_THK": invariant_stream,
            "k_THK": invariant_stream,
            "v_THV": invariant_stream,
        },
        out_src_shardings=invariant_stream,
        local_map=LocalMapConfig(
            in_grad_placements=(invariant_stream, invariant_stream, invariant_stream)
        ),
    )
    ve_cfg.final_norm.sharding_config = invariant_norm_config(include_cp_axis=True)
    proj = ve_cfg.projector
    proj.linear_1.sharding_config = vision_invariant_linear_config(include_cp_axis=True)
    proj.linear_2.sharding_config = vision_invariant_linear_config(include_cp_axis=True)
    proj.post_norm.sharding_config = invariant_norm_config(include_cp_axis=True)
