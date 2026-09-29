# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding configs for Qwen3.5 hybrid attention model.

Sets ``ShardingConfig`` on all sub-configs so that ``model.parallelize()``
applies TP via the Module protocol. Same pattern as ``qwen3/sharding.py``.

Full-attention layers: TP on wq/wk/wv/wo with local_map for inner attention;
each layer's MRoPE ``cache`` buffer is sharded Replicate.
GatedDeltaNet layers: head-sharded TP on projections (ColwiseParallel) and
out_proj (RowwiseParallel); the GDN kernel and depthwise Conv1d run on local
tensors via local_map.
"""

from typing import TYPE_CHECKING

import spmd_types as spmd
from spmd_types import SpmdType

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.attention import VarlenMetadata
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
    token_id_placement,
)
from torchtitan.models.common.moe_sharding import set_moe_sharding_config
from torchtitan.models.common.vision_encoder_sharding import (
    invariant_norm_config,
    set_vision_transformer_block_sharding_config,
    vision_colwise_config,
    vision_invariant_linear_config,
    vision_scaled_bias_rowwise_config,
)
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig

DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP

if TYPE_CHECKING:
    from torchtitan.models.common import SigmoidGatedFeedForward
    from torchtitan.models.qwen3_5.gdn import GatedDeltaNet
    from torchtitan.models.qwen3_5.model import (
        Qwen35Attention,
        Qwen35AttentionMaskDict,
        Qwen35Model,
        Qwen35TransformerBlock,
    )
    from torchtitan.models.qwen3_5.vision_encoder import Qwen35VisionEncoder


def annotate_deltanet_cu_seqlens(attention_masks: "Qwen35AttentionMaskDict") -> None:
    """Annotate the nested GatedDeltaNet ``cu_seq_q`` offsets as DP-varying.

    ``cu_seq_q`` sits inside a ``VarlenMetadata`` inside the attention-mask
    dict, so it is unreachable by name through ``input_sharding``; the caller
    invokes this under the dense SPMD mesh.
    """
    deltanet_metadata = attention_masks.get("deltanet")
    if not isinstance(deltanet_metadata, VarlenMetadata):
        return
    spmd.assert_type(
        deltanet_metadata.cu_seq_q,
        {MeshAxisName.DP: spmd.V, MeshAxisName.TP: spmd.R},
    )


def _qk_norm_sharding() -> ShardingConfig:
    """Per-head QK-norm sharding: weight Replicate, activations Shard(1)."""
    head_plc = attention_activation_placement()
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)},
        in_src_shardings={"input": head_plc},
        in_dst_shardings={"input": head_plc},
        out_src_shardings=head_plc,
        out_dst_shardings=head_plc,
    )


def _decoder_norm_sharding(activation_layout: SpmdType) -> ShardingConfig:
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)},
        in_src_shardings={"input": activation_layout},
        out_src_shardings=activation_layout,
    )


def _conv_weight_sharding() -> ShardingConfig:
    """Depthwise Conv1d weight sharded Shard(0) on out-channels (head-sharded)."""
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
    )


_GROUPED_EXPERTS_PARAM_LAYOUT: dict[str, spmd.PerMeshAxisSpmdType] = {
    "w1_EFD": spmd.S(1),
    "w2_EDF": spmd.S(2),
    "w3_EFD": spmd.S(1),
}


def set_qwen35_sharding_config(
    config: "Qwen35Model.Config",
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    """Fill ``sharding_config`` on all Qwen3.5 sub-configs."""
    set_decoder_sharding_config(config, enable_sp=enable_sp)
    layer_input_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
    )
    first_layer_input_layout = layer_input_layout
    if config.vision_encoder is not None:
        # Vision scatter needs the full embedding sequence on every TP rank.
        config.tok_embeddings.sharding_config = ShardingConfig(
            state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
            in_src_shardings={"input": token_id_placement()},
            in_dst_shardings={"input": token_id_placement()},
            out_src_shardings=dense_activation_placement(tp=spmd.P, cp=spmd.S(0)),
            out_dst_shardings=dense_activation_placement(tp=spmd.R, cp=spmd.S(0)),
            local_map=LocalMapConfig(in_grad_placements=None),
        )
        _set_vision_encoder_sharding(config.vision_encoder)
        # The first layer restores the decoder layout after replicated vision scatter.
        first_layer_input_layout = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    for layer_idx, layer_cfg in enumerate(config.layers):
        input_layout = (
            first_layer_input_layout if layer_idx == 0 else layer_input_layout
        )
        layer_cfg.sharding_config = ShardingConfig(
            in_src_shardings={"x_TD": input_layout},
            in_dst_shardings={"x_TD": layer_input_layout},
            out_src_shardings=layer_input_layout,
        )
        _set_qwen35_layer_sharding(
            layer_cfg,
            attention_input_layout=layer_input_layout,
            enable_sp=enable_sp,
            enable_ep=enable_ep,
        )


def _set_qwen35_layer_sharding(
    layer_cfg: "Qwen35TransformerBlock.Config",
    *,
    attention_input_layout: SpmdType,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    layer_cfg.attention_norm.sharding_config = _decoder_norm_sharding(
        attention_input_layout
    )
    layer_cfg.ffn_norm.sharding_config = norm_config(enable_sp=enable_sp)

    if layer_cfg.attention is not None:
        _set_full_attention_sharding(
            layer_cfg.attention,
            attention_input_layout=attention_input_layout,
            enable_sp=enable_sp,
        )
    else:
        assert layer_cfg.delta_net is not None
        _set_deltanet_sharding(
            layer_cfg.delta_net,
            attention_input_layout=attention_input_layout,
            enable_sp=enable_sp,
        )

    if layer_cfg.feed_forward is not None:
        attn_x_layout = (
            dense_sequence_parallel_placement()
            if enable_sp
            else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
        )
        set_dense_ffn_sharding(
            layer_cfg.feed_forward,
            attn_x_layout=attn_x_layout,
            enable_sp=enable_sp,
        )

    if layer_cfg.moe is not None:
        set_moe_sharding_config(
            layer_cfg.moe,
            enable_ep=enable_ep,
            enable_sp=enable_sp,
            expert_param_layout=_GROUPED_EXPERTS_PARAM_LAYOUT,
        )
        _set_shared_expert_gate_sharding(
            # pyrefly: ignore [missing-attribute]
            layer_cfg.moe.shared_experts,
            enable_sp=enable_sp,
        )


def _set_shared_expert_gate_sharding(
    shared_experts: "SigmoidGatedFeedForward.Config | None",
    *,
    enable_sp: bool,
) -> None:
    """Shard Qwen3.5's shared-expert sigmoid gate.

    The common MoE sharding handles the shared FFN (w1/w2/w3) and the
    module-boundary gather that feeds the gate a Replicate ``x``. Here we only
    add the gate: its weight and local output are Replicate. With SP, the output
    is sliced into the sequence-sharded layout produced by the shared FFN. With
    SP disabled, it remains Replicate and scales the shared FFN's Partial output.
    ``getattr`` keeps this a no-op when the MoE has no shared expert (``None``);
    Qwen3.5's shared expert always carries the gate.
    """
    gate = getattr(shared_experts, "gate", None)
    if gate is None:
        return
    gate_output_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    )
    gate.sharding_config = ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.R),
            "bias": dense_param_placement(tp=spmd.R),
        },
        out_src_shardings=dense_activation_placement(tp=spmd.R, cp=spmd.S(0)),
        out_dst_shardings=gate_output_layout,
    )


def _set_vision_encoder_sharding(ve_cfg: "Qwen35VisionEncoder.Config") -> None:
    """Sharding for the vision encoder.

    All activations flow without SP in the vision encoder.
    Linear layers are ColwiseParallel/RowwiseParallel for memory savings.
    Norms are Replicate. pos_embed is Replicate via state_shardings.
    """
    ve_cfg.sharding_config = ShardingConfig(
        state_shardings={"pos_embed": SpmdType({DP: spmd.R, TP: spmd.I})},
        out_src_shardings=SpmdType({DP: spmd.V, TP: spmd.I}),
        out_dst_shardings=SpmdType({DP: spmd.V, TP: spmd.R}),
    )
    ve_cfg.rotary_pos_emb.sharding_config = ShardingConfig(
        state_shardings={"inv_freq": SpmdType({DP: spmd.R, TP: spmd.I})},
        out_src_shardings=SpmdType({DP: spmd.R, TP: spmd.I}),
    )

    ve_cfg.patch_embed_proj.sharding_config = vision_invariant_linear_config()
    set_vision_transformer_block_sharding_config(
        ve_cfg.block,
        rope_cache_dp=spmd.V,
    )

    # Merger sub-modules
    merger = ve_cfg.merger
    merger.norm.sharding_config = invariant_norm_config()
    merger.fc1.sharding_config = vision_colwise_config()
    merger.fc2.sharding_config = vision_scaled_bias_rowwise_config()


def _set_full_attention_sharding(
    attention_cfg: "Qwen35Attention.Config",
    *,
    attention_input_layout: SpmdType,
    enable_sp: bool,
) -> None:
    """TP sharding for Qwen35Attention (output gating + partial RoPE)."""
    attention_cfg.sharding_config = ShardingConfig(
        in_src_shardings={"x_TD": attention_input_layout},
        in_dst_shardings={"x_TD": dense_activation_placement(tp=spmd.R, cp=spmd.S(0))},
    )
    # The per-layer rope ``cache`` buffer is a Replicate DTensor; MRoPE builds the
    # position-resolved cache from it (``positions`` stays a plain input).
    attention_cfg.rope.sharding_config = ShardingConfig(
        state_shardings={"cache": dense_param_placement(tp=spmd.R)},
    )
    attention_cfg.wq.sharding_config = colwise_config()
    attention_cfg.wk.sharding_config = colwise_config()
    attention_cfg.wv.sharding_config = colwise_config()
    # RowwiseParallel out_proj: reduce-scatter to Shard(1) under SP, else all-reduce
    # to Replicate.
    attention_cfg.wo.sharding_config = rowwise_config(output_sp=enable_sp)

    attention_cfg.q_norm.sharding_config = _qk_norm_sharding()
    attention_cfg.k_norm.sharding_config = _qk_norm_sharding()

    set_gqa_inner_attention_local_map(attention_cfg.inner_attention)


def _set_deltanet_sharding(
    deltanet_cfg: "GatedDeltaNet.Config",
    *,
    attention_input_layout: SpmdType,
    enable_sp: bool,
) -> None:
    """Configure head-sharded TP for GatedDeltaNet.

    Input projections are ColwiseParallel (head-sharded output) and out_proj is
    RowwiseParallel. Conv weights and per-head A_log/dt_bias are Shard(0). The
    recurrence runs on rank-local heads via a single local_map boundary.
    """
    for name in (
        "in_proj_q",
        "in_proj_k",
        "in_proj_v",
        "in_proj_z",
        "in_proj_a",
        "in_proj_b",
    ):
        getattr(deltanet_cfg, name).sharding_config = colwise_config()

    # Depthwise conv weights: Shard(0) on out-channels (head-sharded).
    deltanet_cfg.conv_q.sharding_config = _conv_weight_sharding()
    deltanet_cfg.conv_k.sharding_config = _conv_weight_sharding()
    deltanet_cfg.conv_v.sharding_config = _conv_weight_sharding()

    # RowwiseParallel out_proj: reduce-scatter to Shard(1) under SP, else all-reduce
    # to Replicate.
    deltanet_cfg.out_proj.sharding_config = rowwise_config(output_sp=enable_sp)

    # The projections are 2D [T, C], while the norm and recurrence output are
    # 3D [T, H, V]. Both shard the feature/head axis on TP.
    projected_placement = dense_activation_placement(tp=spmd.S(1), cp=spmd.S(0))
    head_placement = attention_activation_placement()
    parameter_placement = dense_param_placement(tp=spmd.S(0))
    replicated_placement = dense_param_placement(tp=spmd.R)
    cu_seqlens_placement = SpmdType(
        {
            DP: spmd.V,
            CP: spmd.R,
            TP: spmd.R,
        }
    )

    deltanet_cfg.norm.sharding_config = ShardingConfig(
        state_shardings={"weight": replicated_placement},
        in_src_shardings={
            "x": head_placement,
            "gate": head_placement,
        },
        in_dst_shardings={
            "x": head_placement,
            "gate": head_placement,
        },
        out_src_shardings=head_placement,
        out_dst_shardings=head_placement,
    )

    # The inner GDN is the DTensor-to-local boundary for the head-parallel
    # convolution and recurrence.
    deltanet_cfg.inner_gated_delta_net.sharding_config = ShardingConfig(
        in_src_shardings={
            "query_TC": projected_placement,
            "key_TC": projected_placement,
            "value_TC": projected_placement,
            "a_TH": projected_placement,
            "b_TH": projected_placement,
            "conv_q_weight_C1W": parameter_placement,
            "conv_k_weight_C1W": parameter_placement,
            "conv_v_weight_C1W": parameter_placement,
            "A_log_H": parameter_placement,
            "dt_bias_H": parameter_placement,
            "cu_seqlens": cu_seqlens_placement,
        },
        in_dst_shardings={
            "query_TC": projected_placement,
            "key_TC": projected_placement,
            "value_TC": projected_placement,
            "a_TH": projected_placement,
            "b_TH": projected_placement,
            "conv_q_weight_C1W": parameter_placement,
            "conv_k_weight_C1W": parameter_placement,
            "conv_v_weight_C1W": parameter_placement,
            "A_log_H": parameter_placement,
            "dt_bias_H": parameter_placement,
            "cu_seqlens": cu_seqlens_placement,
        },
        out_src_shardings=head_placement,
        out_dst_shardings=head_placement,
        local_map=LocalMapConfig(
            # cu_seqlens varies across DP ranks and is replicated across TP.
            # It has no gradient, but local_map still requires its placement.
            in_grad_placements=(
                projected_placement,
                projected_placement,
                projected_placement,
                projected_placement,
                projected_placement,
                parameter_placement,
                parameter_placement,
                parameter_placement,
                parameter_placement,
                parameter_placement,
                cu_seqlens_placement,
            ),
        ),
    )

    deltanet_cfg.sharding_config = ShardingConfig(
        state_shardings={
            "A_log": parameter_placement,
            "dt_bias": parameter_placement,
        },
        in_src_shardings={"x_TD": attention_input_layout},
        in_dst_shardings={"x_TD": dense_activation_placement(tp=spmd.R, cp=spmd.S(0))},
        out_src_shardings=(
            dense_sequence_parallel_placement()
            if enable_sp
            else dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
        ),
        out_dst_shardings=(
            dense_sequence_parallel_placement()
            if enable_sp
            else dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
        ),
    )
