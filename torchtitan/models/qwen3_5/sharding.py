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
out_proj (RowwiseParallel). Conv1d and FLA kernel forwards are wrapped for
DTensor→local conversion.
"""

import types
from typing import TYPE_CHECKING

import torch.nn.functional as F

from torch import nn
from torch.distributed.tensor import DTensor, Replicate, Shard

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
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig

if TYPE_CHECKING:
    from torchtitan.models.qwen3_5.model import (
        Qwen35Attention,
        Qwen35Model,
        Qwen35TransformerBlock,
    )

TP = "tp"

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


def set_qwen35_sharding_config(
    config: "Qwen35Model.Config",
    *,
    loss_parallel: bool,
) -> None:
    """Fill ``sharding_config`` on all Qwen3.5 sub-configs.

    Uses SP for decoder layers, norm, and lm_head. tok_embeddings output
    stays Replicate so vision scatter and MRoPE can access the full sequence.
    The model forward redistributes to Shard(1) before entering the layers.
    """
    # SP on norm, lm_head, and layers
    set_decoder_sharding_config(config, loss_parallel=loss_parallel, enable_sp=True)
    # Override: don't distribute freqs_cis — MRoPE indexes it with plain tensors
    config.sharding_config = ShardingConfig()
    # Override tok_embeddings: output Replicate (not Shard(1)) for vision scatter
    config.tok_embeddings.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=Shard(0))},
        in_src_shardings={"input": _REPLICATE_ACT},
        in_dst_shardings={"input": _REPLICATE_ACT},
        out_dst_shardings=_REPLICATE_ACT,
    )
    _set_vision_encoder_sharding(config.vision_encoder)
    for layer_cfg in config.layers:
        _set_qwen35_layer_sharding(layer_cfg)


def _set_qwen35_layer_sharding(
    layer_cfg: "Qwen35TransformerBlock.Config",
) -> None:
    norm = norm_config(enable_sp=True)
    layer_cfg.attention_norm.sharding_config = norm
    layer_cfg.ffn_norm.sharding_config = norm

    if layer_cfg.layer_type == "full_attn":
        assert layer_cfg.attention is not None
        _set_full_attention_sharding(layer_cfg.attention)
    else:
        assert layer_cfg.deltanet is not None
        _set_deltanet_sharding(layer_cfg.deltanet)

    if layer_cfg.feed_forward is not None:
        set_dense_ffn_sharding(
            layer_cfg.feed_forward,
            attn_x_placement=Shard(1),
            enable_sp=True,
        )

    if layer_cfg.shared_ffn is not None:
        set_dense_ffn_sharding(
            layer_cfg.shared_ffn,
            attn_x_placement=Shard(1),
            enable_sp=True,
        )
    if layer_cfg.shared_gate is not None:
        layer_cfg.shared_gate.sharding_config = _REPLICATE_STATE


def _set_vision_encoder_sharding(ve_cfg) -> None:
    """Sharding for the vision encoder.

    All activations flow as Replicate — no SP in the vision encoder.
    Linear layers are ColwiseParallel/RowwiseParallel for memory savings.
    Norms and patch_embed are Replicate. pos_embed is distributed as
    Replicate via state_shardings on the encoder config.
    """
    ve_cfg.sharding_config = ShardingConfig(
        state_shardings={"pos_embed": _REPLICATE_PARAM},
    )

    ve_cfg.patch_embed_proj.sharding_config = _REPLICATE_STATE

    # Separate Q/K/V: colwise sharding
    ve_cfg.attn_wq.sharding_config = colwise_config()
    ve_cfg.attn_wk.sharding_config = colwise_config()
    ve_cfg.attn_wv.sharding_config = colwise_config()
    ve_cfg.attn_proj.sharding_config = rowwise_config(output_sp=False)
    ve_cfg.mlp_fc1.sharding_config = colwise_config()
    ve_cfg.mlp_fc2.sharding_config = rowwise_config(output_sp=False)

    ve_cfg.merger_fc1.sharding_config = colwise_config()
    ve_cfg.merger_fc2.sharding_config = rowwise_config(output_sp=False)


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

    qk_norm_sharding = ShardingConfig(
        state_shardings={"weight": _REPLICATE_PARAM},
        in_src_shardings={"input": dense_activation_placement(tp=Shard(2))},
        in_dst_shardings={"input": dense_activation_placement(tp=Shard(2))},
        out_dst_shardings=dense_activation_placement(tp=Shard(2)),
    )
    attention_cfg.q_norm.sharding_config = qk_norm_sharding
    attention_cfg.k_norm.sharding_config = qk_norm_sharding

    set_gqa_inner_attention_local_map(attention_cfg.inner_attention)


def _set_deltanet_sharding(deltanet_cfg) -> None:
    """Sharding for GatedDeltaNet: head-sharded TP on projections.

    Input is allgathered on both TP and CP (Shard(1)→Replicate) because
    the recurrence needs the full sequence. Projections are ColwiseParallel
    (head-sharded output). Conv1d and FLA kernels are wrapped for
    DTensor→local conversion. out_proj is RowwiseParallel (reduce-scatter
    back to Shard(1)).

    A_log and dt_bias are per-head parameters, Shard(0) on TP.
    Sub-module sharding is set on built modules by
    ``set_deltanet_sub_module_sharding`` before ``model.parallelize()``.
    """
    deltanet_cfg.sharding_config = ShardingConfig(
        state_shardings={
            "A_log": dense_param_placement(tp=Shard(0)),
            "dt_bias": dense_param_placement(tp=Shard(0)),
        },
        in_src_shardings={"x": dense_activation_placement(tp=Shard(1))},
        # cp=Replicate: GatedDeltaNet is recurrent — needs full sequence
        in_dst_shardings={
            "x": dense_activation_placement(tp=Replicate(), cp=Replicate())
        },
        out_dst_shardings=dense_activation_placement(tp=Shard(1)),
    )


def set_vision_encoder_sub_module_sharding(vision_encoder) -> None:
    """Set _sharding_config on vision encoder sub-modules built inline.

    Norms (LayerNorm) in VisionTransformerBlock and PatchMerger are created
    via Module.from_nn_module(nn.LayerNorm) — not from config fields.
    Must be called after model build but before model.parallelize().
    """
    for layer in vision_encoder.layers.values():
        for name in ("norm1", "norm2"):
            child = getattr(layer, name, None)
            if child is not None:
                child._sharding_config = _REPLICATE_NORM
        # VisionAttention: declare rope_cache as Replicate so plain
        # rope_cache is wrapped as DTensor to match DTensor q/k.
        layer.attn._sharding_config = ShardingConfig(
            in_src_shardings={"rope_cache": _REPLICATE_ACT},
            in_dst_shardings={"rope_cache": _REPLICATE_ACT},
        )
        # FlexAttention: local_map to convert DTensor q/k/v to local.
        # Same as set_gqa_inner_attention_local_map but on built module.
        if hasattr(layer.attn, "flex_attention"):
            qkv_plc = {TP: Shard(2)}
            layer.attn.flex_attention._sharding_config = ShardingConfig(
                local_map=LocalMapConfig(
                    # pyrefly: ignore [bad-argument-type]
                    in_placements=(qkv_plc, qkv_plc, qkv_plc),
                    # pyrefly: ignore [bad-argument-type]
                    out_placements=(qkv_plc,),
                    # pyrefly: ignore [bad-argument-type]
                    in_grad_placements=(qkv_plc, qkv_plc, qkv_plc),
                ),
            )
    # Merger norm
    if hasattr(vision_encoder.merger, "norm"):
        vision_encoder.merger.norm._sharding_config = _REPLICATE_NORM
    # Merger GELU: set None to skip protocol wrapping. Per-layer mlp.act_fn
    # doesn't need this because its parent VisionMLP has no _sharding_config,
    # but the merger's children get processed due to merger.norm having one.
    if hasattr(vision_encoder.merger, "act_fn"):
        vision_encoder.merger.act_fn._sharding_config = None
    # VisionRotaryEmbedding: don't set _sharding_config — wrapping forward
    # would break RoPE compute. inv_freq stays as a plain buffer; the
    # resulting rope_cache is wrapped as DTensor by VisionAttention's
    # in_src_shardings.

    # pos_embed interpolation: F.interpolate's decomposition doesn't
    # support DTensor. Wrap to convert pos_embed to local before use.
    _wrap_pos_embed_for_interpolation(vision_encoder)

    # patch_embed (Linear): plain pixel_values in → DTensor(Replicate) out
    vision_encoder.patch_embed._sharding_config = ShardingConfig(
        state_shardings={
            "weight": _REPLICATE_PARAM,
            "bias": _REPLICATE_PARAM,
        },
        in_src_shardings={"input": _REPLICATE_ACT},
        in_dst_shardings={"input": _REPLICATE_ACT},
        out_dst_shardings=_REPLICATE_ACT,
    )


def set_deltanet_sub_module_sharding(deltanet_module) -> None:
    """Set head-sharded TP on GatedDeltaNet sub-modules.

    Projections are ColwiseParallel (head-sharded output), out_proj is
    RowwiseParallel (reduce-scatter to SP). Conv1d weights are Shard(0)
    on the channel dim (matching head sharding). The conv and kernel
    forwards are wrapped for DTensor→local conversion (depthwise conv
    and FLA kernels don't support DTensor dispatch).

    Must be called after model build but before model.parallelize().
    """
    for name in (
        "in_proj_q",
        "in_proj_k",
        "in_proj_v",
        "in_proj_z",
        "in_proj_a",
        "in_proj_b",
    ):
        getattr(deltanet_module, name)._sharding_config = colwise_config()

    _conv_shard = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=Shard(0))},
    )
    for name in ("conv_q", "conv_k", "conv_v"):
        conv = getattr(deltanet_module, name)
        conv._sharding_config = _conv_shard
        _wrap_conv1d(conv)

    # GatedDeltaKernel: local_map converts DTensor q/k/v/g/beta to local
    # for FLA kernels, same pattern as FlexAttention's local_map.
    _kernel_plc = {TP: Shard(2)}
    deltanet_module.kernel._sharding_config = ShardingConfig(
        local_map=LocalMapConfig(
            # pyrefly: ignore [bad-argument-type]
            in_placements=(_kernel_plc,) * 5,
            # pyrefly: ignore [bad-argument-type]
            out_placements=(_kernel_plc,),
            # pyrefly: ignore [bad-argument-type]
            in_grad_placements=(_kernel_plc,) * 5,
        ),
    )

    deltanet_module.norm._sharding_config = _REPLICATE_STATE
    deltanet_module.out_proj._sharding_config = rowwise_config(output_sp=True)


def _wrap_conv1d(conv1d_module) -> None:
    """Wrap depthwise Conv1d forward for DTensor→local conversion.

    DTensor dispatch for Conv1d doesn't handle sharded groups: nn.Conv1d
    stores groups as a plain int, but when the weight is TP-sharded on
    the channel dim, the local weight has fewer channels than groups.
    This wrapper converts inputs/weights to local and uses the local
    channel count as groups.

    TODO: Remove once DTensor Conv1d dispatch handles sharded groups.
    """
    original_forward = conv1d_module.forward.__func__

    def safe_forward(self, x):
        if isinstance(x, DTensor):
            mesh, plc = x.device_mesh, x.placements
            w = self.weight
            if isinstance(w, DTensor):
                w = w.to_local()
            # self.groups is the global count; use local weight's channel dim
            local_groups = w.shape[0]
            out = F.conv1d(
                x.to_local(),
                w,
                None,
                self.stride,
                self.padding,
                self.dilation,
                local_groups,
            )
            return DTensor.from_local(out, mesh, plc, run_check=False)
        return original_forward(self, x)

    conv1d_module.forward = types.MethodType(safe_forward, conv1d_module)


def _wrap_pos_embed_for_interpolation(vision_encoder) -> None:
    """Wrap compute_position_embeddings to convert pos_embed to local.

    F.interpolate's decomposition uses _unsafe_index which doesn't support
    DTensor. Since pos_embed is Replicate, to_local is a no-op for data.

    TODO: Remove once F.interpolate on FSDP2-managed DTensors is fixed upstream.
    """
    original_fn = vision_encoder.compute_position_embeddings.__func__

    def safe_compute(self, grid_thw, max_num_patch):
        pos = self.pos_embed
        if isinstance(pos, DTensor):
            mesh, plc = pos.device_mesh, pos.placements
            self.pos_embed = nn.Parameter(pos.to_local(), requires_grad=False)
            learned_pos, rope_cache = original_fn(self, grid_thw, max_num_patch)
            self.pos_embed = pos
            learned_pos = DTensor.from_local(learned_pos, mesh, plc, run_check=False)
            return learned_pos, rope_cache
        return original_fn(self, grid_thw, max_num_patch)

    vision_encoder.compute_position_embeddings = types.MethodType(
        safe_compute, vision_encoder
    )
