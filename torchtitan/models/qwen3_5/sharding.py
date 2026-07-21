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
out_proj (RowwiseParallel); the FLA kernel and depthwise Conv1d run on local
tensors via local_map.
"""

from typing import TYPE_CHECKING

import spmd_types as spmd

from torchtitan.models.common.decoder_sharding import (
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
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig, SpmdLayout

if TYPE_CHECKING:
    from torchtitan.models.qwen3_5.model import (
        GatedDeltaNet,
        Qwen35Attention,
        Qwen35Model,
        Qwen35TransformerBlock,
        SharedExperts,
    )
    from torchtitan.models.qwen3_5.vision_encoder import Qwen35VisionEncoder


def _replicate_norm() -> ShardingConfig:
    """Replicate norm (weight/bias and activations) — used by the vision
    encoder, which runs without sequence parallelism."""
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.R),
            "bias": dense_param_placement(tp=spmd.R),
        },
        in_src_shardings={"input": dense_activation_placement(tp=spmd.R)},
        in_dst_shardings={"input": dense_activation_placement(tp=spmd.R)},
        out_dst_shardings=dense_activation_placement(tp=spmd.R),
    )


def _qk_norm_sharding() -> ShardingConfig:
    """Per-head QK-norm sharding: weight Replicate, activations Shard(2)."""
    head_plc = dense_activation_placement(tp=spmd.S(2))
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)},
        in_src_shardings={"input": head_plc},
        in_dst_shardings={"input": head_plc},
        out_dst_shardings=head_plc,
    )


def _decoder_norm_sharding(activation_layout: SpmdLayout) -> ShardingConfig:
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
    enable_ep: bool,
) -> None:
    """Fill ``sharding_config`` on all Qwen3.5 sub-configs.

    Uses SP for decoder layers, norm, and lm_head. tok_embeddings output
    stays Replicate so vision scatter and MRoPE can access the full sequence.
    The model forward redistributes to Shard(1) before entering the layers.
    """
    # SP on norm, lm_head, and layers. Each full-attention layer owns its rope;
    # its cache buffer is sharded Replicate in _set_full_attention_sharding.
    set_decoder_sharding_config(config, enable_sp=True)
    # Override tok_embeddings: output Replicate (not Shard(1)) for vision scatter
    config.tok_embeddings.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
        in_src_shardings={"input": dense_activation_placement(tp=spmd.R)},
        in_dst_shardings={"input": dense_activation_placement(tp=spmd.R)},
        out_src_shardings=dense_activation_placement(tp=spmd.P),
        out_dst_shardings=dense_activation_placement(tp=spmd.R),
        local_map=LocalMapConfig(in_grad_placements=None),
    )
    _set_vision_encoder_sharding(config.vision_encoder)
    # The embedding path stays replicated through multimodal vision scatter.
    # The first attention block restores SP; later decoder block inputs are SP.
    first_layer_input_layout = dense_activation_placement(tp=spmd.R)
    layer_input_layout = dense_sequence_parallel_placement()
    for layer_idx, layer_cfg in enumerate(config.layers):
        _set_qwen35_layer_sharding(
            layer_cfg,
            attention_input_layout=(
                first_layer_input_layout if layer_idx == 0 else layer_input_layout
            ),
            enable_ep=enable_ep,
        )


def _set_qwen35_layer_sharding(
    layer_cfg: "Qwen35TransformerBlock.Config",
    *,
    attention_input_layout: SpmdLayout,
    enable_ep: bool,
) -> None:
    layer_cfg.attention_norm.sharding_config = _decoder_norm_sharding(
        attention_input_layout
    )
    layer_cfg.ffn_norm.sharding_config = norm_config(enable_sp=True)

    if layer_cfg.attention is not None:
        _set_full_attention_sharding(
            layer_cfg.attention,
            attention_input_layout=attention_input_layout,
        )
    else:
        assert layer_cfg.delta_net is not None
        _set_deltanet_sharding(
            layer_cfg.delta_net,
            attention_input_layout=attention_input_layout,
        )

    if layer_cfg.feed_forward is not None:
        set_dense_ffn_sharding(
            layer_cfg.feed_forward,
            attn_x_layout=dense_sequence_parallel_placement(),
            enable_sp=True,
        )

    if layer_cfg.moe is not None:
        set_moe_sharding_config(
            layer_cfg.moe,
            enable_ep=enable_ep,
            enable_sp=True,
            expert_param_layout=_GROUPED_EXPERTS_PARAM_LAYOUT,
        )
        # pyrefly: ignore [missing-attribute]
        _set_shared_expert_gate_sharding(layer_cfg.moe.shared_experts)


def _set_shared_expert_gate_sharding(
    shared_experts: "SharedExperts.Config | None",
) -> None:
    """Shard Qwen3.5's shared-expert sigmoid gate.

    The common MoE sharding handles the shared FFN (w1/w2/w3) and the
    module-boundary gather that feeds the gate a Replicate ``x``. Here we only
    add the gate: its weight is Replicate and its output is Replicate, so
    ``sigmoid(gate(x)) * ffn(x)`` is ``Replicate * Partial = Partial`` with no
    extra collective. ``getattr`` keeps this a no-op when the MoE has no shared
    expert (``None``); Qwen3.5's shared expert always carries the gate.
    """
    gate = getattr(shared_experts, "gate", None)
    if gate is None:
        return
    gate.sharding_config = ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.R),
            "bias": dense_param_placement(tp=spmd.R),
        },
        out_dst_shardings=dense_activation_placement(tp=spmd.R),
    )


def _set_vision_encoder_sharding(ve_cfg: "Qwen35VisionEncoder.Config") -> None:
    """Sharding for the vision encoder.

    All activations flow as Replicate — no SP in the vision encoder.
    Linear layers are ColwiseParallel/RowwiseParallel for memory savings.
    Norms are Replicate. pos_embed is Replicate via state_shardings.
    """
    ve_cfg.sharding_config = ShardingConfig(
        state_shardings={"pos_embed": dense_param_placement(tp=spmd.R)},
    )

    # patch_embed receives plain pixel_values — wrap as DTensor(Replicate)
    ve_cfg.patch_embed_proj.sharding_config = ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.R),
            "bias": dense_param_placement(tp=spmd.R),
        },
        in_src_shardings={"input": dense_activation_placement(tp=spmd.R)},
        in_dst_shardings={"input": dense_activation_placement(tp=spmd.R)},
        out_dst_shardings=dense_activation_placement(tp=spmd.R),
    )

    # Block sub-modules
    block = ve_cfg.block
    block.norm1.sharding_config = _replicate_norm()
    block.norm2.sharding_config = _replicate_norm()

    block.attn.sharding_config = ShardingConfig(
        in_src_shardings={"rope_cache": dense_activation_placement(tp=spmd.R)},
        in_dst_shardings={"rope_cache": dense_activation_placement(tp=spmd.R)},
    )
    block.attn.wq.sharding_config = colwise_config()
    block.attn.wk.sharding_config = colwise_config()
    block.attn.wv.sharding_config = colwise_config()
    block.attn.proj.sharding_config = rowwise_config(output_sp=False)
    set_gqa_inner_attention_local_map(block.attn.inner_attention)

    block.mlp.fc1.sharding_config = colwise_config()
    block.mlp.fc2.sharding_config = rowwise_config(output_sp=False)

    # Merger sub-modules
    merger = ve_cfg.merger
    merger.norm.sharding_config = _replicate_norm()
    merger.fc1.sharding_config = colwise_config()
    merger.fc2.sharding_config = rowwise_config(output_sp=False)


def _set_full_attention_sharding(
    attention_cfg: "Qwen35Attention.Config",
    *,
    attention_input_layout: SpmdLayout,
) -> None:
    """TP sharding for Qwen35Attention (output gating + partial RoPE)."""
    attention_cfg.sharding_config = ShardingConfig(
        in_src_shardings={"x": attention_input_layout},
        in_dst_shardings={"x": dense_activation_placement(tp=spmd.R)},
    )
    # The per-layer rope ``cache`` buffer is a Replicate DTensor; MRoPE builds the
    # position-resolved cache from it (``positions`` stays a plain input).
    attention_cfg.rope.sharding_config = ShardingConfig(
        state_shardings={"cache": dense_param_placement(tp=spmd.R)},
    )
    attention_cfg.wq.sharding_config = colwise_config()
    attention_cfg.wk.sharding_config = colwise_config()
    attention_cfg.wv.sharding_config = colwise_config()
    attention_cfg.wo.sharding_config = rowwise_config(output_sp=True)

    attention_cfg.q_norm.sharding_config = _qk_norm_sharding()
    attention_cfg.k_norm.sharding_config = _qk_norm_sharding()

    set_gqa_inner_attention_local_map(attention_cfg.inner_attention)


def _set_deltanet_sharding(
    deltanet_cfg: "GatedDeltaNet.Config",
    *,
    attention_input_layout: SpmdLayout,
) -> None:
    """Sharding for GatedDeltaNet: head-sharded TP on projections.

    Input is allgathered (Shard(1)→Replicate) so that the recurrence
    sees the full sequence. Projections are ColwiseParallel (head-sharded
    output). The FLA kernel runs on local tensors via local_map.
    out_proj is RowwiseParallel (reduce-scatter back to Shard(1)).

    A_log and dt_bias are per-head parameters, Shard(0) on TP.
    Conv1d weights are Shard(0) (out-channels); the DTensor->local conversion
    for the depthwise conv is handled in the model's ``_causal_conv1d``.
    """
    # ColwiseParallel on all input projections
    for name in (
        "in_proj_q",
        "in_proj_k",
        "in_proj_v",
        "in_proj_z",
        "in_proj_a",
        "in_proj_b",
    ):
        getattr(deltanet_cfg, name).sharding_config = colwise_config()

    # Depthwise Conv1d weights: Shard(0) on out-channels (head-sharded).
    deltanet_cfg.conv_q.sharding_config = _conv_weight_sharding()
    deltanet_cfg.conv_k.sharding_config = _conv_weight_sharding()
    deltanet_cfg.conv_v.sharding_config = _conv_weight_sharding()

    # RowwiseParallel on output projection (reduce-scatter to SP)
    deltanet_cfg.out_proj.sharding_config = rowwise_config(output_sp=True)

    # RMSNormGated: per-head norm, weight Replicate, activations Shard(2)
    _norm_plc = dense_activation_placement(tp=spmd.S(2))
    deltanet_cfg.norm.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)},
        in_src_shardings={"x": _norm_plc, "gate": _norm_plc},
        in_dst_shardings={"x": _norm_plc, "gate": _norm_plc},
        out_dst_shardings=_norm_plc,
    )

    # GatedDeltaKernel: local_map converts DTensor q/k/v/g/beta to local
    _kernel_plc = dense_activation_placement(tp=spmd.S(2))
    deltanet_cfg.kernel.sharding_config = ShardingConfig(
        in_dst_shardings={
            "q": _kernel_plc,
            "k": _kernel_plc,
            "v": _kernel_plc,
            "g": _kernel_plc,
            "beta": _kernel_plc,
        },
        out_src_shardings=_kernel_plc,
        local_map=LocalMapConfig(
            in_grad_placements=(_kernel_plc,) * 5,
        ),
    )

    deltanet_cfg.sharding_config = ShardingConfig(
        state_shardings={
            "A_log": dense_param_placement(tp=spmd.S(0)),
            "dt_bias": dense_param_placement(tp=spmd.S(0)),
        },
        in_src_shardings={"x": attention_input_layout},
        in_dst_shardings={"x": dense_activation_placement(tp=spmd.R)},
        out_dst_shardings=dense_sequence_parallel_placement(),
    )
