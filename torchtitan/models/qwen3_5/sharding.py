# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding configs for Qwen3.5 hybrid attention model.

Sets ``ShardingConfig`` on all sub-configs so that ``model.parallelize()``
applies TP via the Module protocol. Same pattern as ``qwen3/sharding.py``.

Full attention layers: TP on wq/wk/wv/wo with local_map for inner attention.
GatedDeltaNet layers: head-sharded TP on projections (ColwiseParallel) and
out_proj (RowwiseParallel). FLA kernel uses local_map for DTensor→local
conversion. Conv1d sharding is set on built modules.
"""

from typing import TYPE_CHECKING

from torch.distributed.tensor import Placement, Replicate, Shard

from torchtitan.models.common.decoder_sharding import (
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    norm_config,
    rowwise_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    set_gqa_inner_attention_local_map,
)
from torchtitan.models.common.moe_sharding import set_moe_sharding_config
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig

if TYPE_CHECKING:
    from torchtitan.models.qwen3_5.model import (
        Qwen35Attention,
        Qwen35Model,
        Qwen35TransformerBlock,
    )

_REPLICATE_PARAM = dense_param_placement(tp=Replicate())
_REPLICATE_STATE = ShardingConfig(
    state_shardings={"weight": _REPLICATE_PARAM, "bias": _REPLICATE_PARAM}
)
_REPLICATE_ACT = dense_activation_placement(tp=Replicate())

# For norms/modules that receive and emit Replicate activations
_REPLICATE_NORM = ShardingConfig(
    state_shardings={"weight": _REPLICATE_PARAM, "bias": _REPLICATE_PARAM},
    in_src_shardings={"input": _REPLICATE_ACT},
    in_dst_shardings={"input": _REPLICATE_ACT},
    out_dst_shardings=_REPLICATE_ACT,
)


_GROUPED_EXPERTS_PARAM_LAYOUT: dict[str, Placement] = {
    "w1_EFD": Shard(1),
    "w2_EDF": Shard(2),
    "w3_EFD": Shard(1),
}


def set_qwen35_sharding_config(
    config: "Qwen35Model.Config",
    *,
    loss_parallel: bool,
    enable_ep: bool,
) -> None:
    """Fill ``sharding_config`` on all Qwen3.5 sub-configs.

    Uses SP for decoder layers, norm, and lm_head. tok_embeddings output
    stays Replicate so vision scatter and MRoPE can access the full sequence.
    The model forward redistributes to Shard(1) before entering the layers.
    """
    # SP on norm, lm_head, and layers. freqs_cis stays Replicate (set by base).
    set_decoder_sharding_config(config, loss_parallel=loss_parallel, enable_sp=True)
    # Override tok_embeddings: output Replicate (not Shard(1)) for vision scatter
    config.tok_embeddings.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=Shard(0))},
        in_src_shardings={"input": _REPLICATE_ACT},
        in_dst_shardings={"input": _REPLICATE_ACT},
        out_dst_shardings=_REPLICATE_ACT,
    )
    _set_vision_encoder_sharding(config.vision_encoder)
    for layer_cfg in config.layers:
        _set_qwen35_layer_sharding(layer_cfg, enable_ep=enable_ep)


def _set_qwen35_layer_sharding(
    layer_cfg: "Qwen35TransformerBlock.Config",
    *,
    enable_ep: bool,
) -> None:
    norm = norm_config(enable_sp=True)
    layer_cfg.attention_norm.sharding_config = norm
    layer_cfg.ffn_norm.sharding_config = norm

    if layer_cfg.attention is not None:
        _set_full_attention_sharding(layer_cfg.attention)
    else:
        assert layer_cfg.delta_net is not None
        _set_deltanet_sharding(layer_cfg.delta_net)

    if layer_cfg.feed_forward is not None:
        set_dense_ffn_sharding(
            layer_cfg.feed_forward,
            attn_x_placement=Shard(1),
            enable_sp=True,
        )

    if layer_cfg.moe is not None:
        set_moe_sharding_config(
            layer_cfg.moe,
            enable_ep=enable_ep,
            enable_sp=True,
            expert_param_layout=_GROUPED_EXPERTS_PARAM_LAYOUT,
        )


def _set_vision_encoder_sharding(ve_cfg) -> None:
    """Sharding for the vision encoder.

    All activations flow as Replicate — no SP in the vision encoder.
    Linear layers are ColwiseParallel/RowwiseParallel for memory savings.
    Norms are Replicate. pos_embed is Replicate via state_shardings.
    """
    ve_cfg.sharding_config = ShardingConfig(
        state_shardings={"pos_embed": _REPLICATE_PARAM},
    )

    # patch_embed receives plain pixel_values — wrap as DTensor(Replicate)
    ve_cfg.patch_embed_proj.sharding_config = ShardingConfig(
        state_shardings={"weight": _REPLICATE_PARAM, "bias": _REPLICATE_PARAM},
        in_src_shardings={"input": _REPLICATE_ACT},
        in_dst_shardings={"input": _REPLICATE_ACT},
        out_dst_shardings=_REPLICATE_ACT,
    )

    # Block sub-modules
    block = ve_cfg.block
    block.norm1.sharding_config = _REPLICATE_NORM
    block.norm2.sharding_config = _REPLICATE_NORM

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

    # Merger sub-modules
    merger = ve_cfg.merger
    merger.norm.sharding_config = _REPLICATE_NORM
    merger.fc1.sharding_config = colwise_config()
    merger.fc2.sharding_config = rowwise_config(output_sp=False)


def _set_full_attention_sharding(
    attention_cfg: "Qwen35Attention.Config",
) -> None:
    """TP sharding for Qwen35Attention (output gating + partial RoPE)."""
    attention_cfg.sharding_config = ShardingConfig(
        in_src_shardings={
            "x": dense_activation_placement(tp=Shard(1)),
            "rope_cache": dense_param_placement(tp=Replicate()),
        },
        in_dst_shardings={
            "x": dense_activation_placement(tp=Replicate()),
            "rope_cache": dense_param_placement(tp=Replicate()),
        },
    )
    attention_cfg.wq.sharding_config = colwise_config()
    attention_cfg.wk.sharding_config = colwise_config()
    attention_cfg.wv.sharding_config = colwise_config()
    attention_cfg.wo.sharding_config = rowwise_config(output_sp=True)

    _head_plc = dense_activation_placement(tp=Shard(2))
    qk_norm_sharding = ShardingConfig(
        state_shardings={"weight": _REPLICATE_PARAM},
        in_src_shardings={"input": _head_plc},
        in_dst_shardings={"input": _head_plc},
        out_dst_shardings=_head_plc,
    )
    attention_cfg.q_norm.sharding_config = qk_norm_sharding
    attention_cfg.k_norm.sharding_config = qk_norm_sharding

    set_gqa_inner_attention_local_map(attention_cfg.inner_attention)


def _set_deltanet_sharding(deltanet_cfg) -> None:
    """Sharding for GatedDeltaNet: head-sharded TP on projections.

    Input is allgathered (Shard(1)→Replicate) so that the recurrence
    sees the full sequence. Projections are ColwiseParallel (head-sharded
    output). The FLA kernel runs on local tensors via local_map.
    out_proj is RowwiseParallel (reduce-scatter back to Shard(1)).

    A_log and dt_bias are per-head parameters, Shard(0) on TP.
    Conv1d weights are Shard(0) (out-channels); the DTensor->local conversion
    for the depthwise conv is handled in the model's ``_causal_conv``.
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
    _conv_shard = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=Shard(0))},
    )
    deltanet_cfg.conv_q.sharding_config = _conv_shard
    deltanet_cfg.conv_k.sharding_config = _conv_shard
    deltanet_cfg.conv_v.sharding_config = _conv_shard

    # RowwiseParallel on output projection (reduce-scatter to SP)
    deltanet_cfg.out_proj.sharding_config = rowwise_config(output_sp=True)

    # RMSNormGated: per-head norm, weight Replicate, activations Shard(2)
    _norm_plc = dense_activation_placement(tp=Shard(2))
    deltanet_cfg.norm.sharding_config = ShardingConfig(
        state_shardings={"weight": _REPLICATE_PARAM},
        in_src_shardings={"x": _norm_plc, "gate": _norm_plc},
        in_dst_shardings={"x": _norm_plc, "gate": _norm_plc},
        out_dst_shardings=_norm_plc,
    )

    # GatedDeltaKernel: local_map converts DTensor q/k/v/g/beta to local
    _kernel_plc = dense_activation_placement(tp=Shard(2))
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
            "A_log": dense_param_placement(tp=Shard(0)),
            "dt_bias": dense_param_placement(tp=Shard(0)),
        },
        in_src_shardings={"x": dense_activation_placement(tp=Shard(1))},
        in_dst_shardings={"x": dense_activation_placement(tp=Replicate())},
        out_dst_shardings=dense_activation_placement(tp=Shard(1)),
    )
