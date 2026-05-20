# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

from torch.distributed.tensor import Replicate, Shard

from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
)
from torchtitan.models.qwen3.sharding import set_qwen3_sharding_config
from torchtitan.protocols.sharding import ShardingConfig

if TYPE_CHECKING:
    from torchtitan.models.qwen3_vl.model import Qwen3VLModel


def set_qwen3_vl_sharding_config(
    config: "Qwen3VLModel.Config",
    *,
    loss_parallel: bool,
    enable_ep: bool,
) -> None:
    """Fill ``sharding_config`` on all Qwen3-VL sub-configs.

    Delegates to ``set_qwen3_sharding_config`` with ``enable_sp=False``
    because Qwen3-VL keeps hidden states as ``Replicate`` (not
    ``Shard(1)``) -- no SequenceParallel due to vision scatter and
    DeepStack needing full-sequence access.

    Vision encoder is sharded with the same Replicate-output discipline:
    linears shard on TP for memory but immediately redistribute back to
    Replicate on output (matching the legacy ``_apply_tp_to_vision_encoder``
    pattern with ``use_local_output=False``).
    """
    set_qwen3_sharding_config(
        config,
        loss_parallel=loss_parallel,
        enable_sp=False,
        enable_ep=enable_ep,
    )

    if config.vision_encoder is not None:
        _set_vision_encoder_sharding(config.vision_encoder)


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

    # Qwen3VLVisionEncoder.Config uses flat Linear.Config fields shared
    # across all layers: patch_embed_proj, attn_qkv, attn_proj, mlp_fc1,
    # mlp_fc2, merger_fc1, merger_fc2. Setting sharding on these configs
    # applies to every layer that builds from them.
    ve_cfg.patch_embed_proj.sharding_config = _replicate_linear_config()
    ve_cfg.attn_qkv.sharding_config = _vision_colwise_config()
    ve_cfg.attn_proj.sharding_config = _replicate_output_rowwise_config()
    ve_cfg.mlp_fc1.sharding_config = _vision_colwise_config()
    ve_cfg.mlp_fc2.sharding_config = _replicate_output_rowwise_config()
    ve_cfg.merger_fc1.sharding_config = _vision_colwise_config()
    ve_cfg.merger_fc2.sharding_config = _replicate_output_rowwise_config()
    vision_norm = ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=Replicate()),
            "bias": dense_param_placement(tp=Replicate()),
        },
    )
    ve_cfg.block_norm.sharding_config = vision_norm
    ve_cfg.merger_norm.sharding_config = vision_norm


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
