# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

from torch.distributed.tensor import Replicate, Shard

from torchtitan.models.common.decoder_sharding import (
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    rowwise_config,
)
from torchtitan.models.qwen3.sharding import set_qwen3_sharding_config
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig

if TYPE_CHECKING:
    from torchtitan.models.qwen3_vl.model import Qwen3VLModel
    from torchtitan.models.qwen3_vl.vision_encoder import Qwen3VLVisionEncoder


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

    Vision encoder uses heads-parallel TP: each colwise/rowwise pair
    chains as ``Replicate -> Shard(-1) -> Replicate`` without any
    intermediate all-gather; attention runs on local heads inside a
    ``local_map`` on ``VisionInnerAttention`` so rope and flex_attention
    see plain Tensors with TP-sharded heads.
    """
    set_qwen3_sharding_config(
        config,
        loss_parallel=loss_parallel,
        enable_sp=False,
        enable_ep=enable_ep,
    )

    if config.vision_encoder is not None:
        _set_vision_encoder_sharding(config.vision_encoder)


def _set_vision_encoder_sharding(ve_cfg: "Qwen3VLVisionEncoder.Config") -> None:
    """Set sharding on the Qwen3-VL vision encoder.

    Activation flow under TP:

    * ``patch_embed.proj`` wraps plain ``pixel_values`` to ``DTensor(Replicate)``;
      that flows through residuals and norms (Replicate weights, pass-through).
    * Each colwise/rowwise pair: colwise output is ``Shard(-1)``, GELU is
      pointwise on DTensor, rowwise accepts ``Shard(-1)`` and all-reduces to
      ``Replicate``. No intermediate all-gather.
    * ``attn.qkv``'s ``Shard(-1)`` output propagates through reshape +
      unbind into q/k/v ``DTensor(Shard(2))`` on the heads dim.
    * ``attn.inner_attention`` is wrapped in ``local_map``: q/k/v unwrap to
      plain locals with TP-sharded heads, rope and flex_attention run on
      locals, output re-wraps to ``DTensor(Shard(2))``.
    * After ``attn_output.reshape(..., -1)`` it becomes ``DTensor(Shard(-1))``
      and feeds ``attn.proj`` rowwise -> ``Replicate``.

    ``pos_embed`` is declared on the encoder root via ``state_shardings``.
    """
    # Encoder root: pos_embed is a Replicate parameter on TP.
    ve_cfg.sharding_config = ShardingConfig(
        state_shardings={"pos_embed": dense_param_placement(tp=Replicate())},
    )

    # Flat Linear.Config fields on Qwen3VLVisionEncoder.Config are shared
    # across all layers; one sharding_config assignment propagates to every
    # layer that builds from them.
    ve_cfg.patch_embed_proj.sharding_config = _replicate_linear_config()
    ve_cfg.attn_qkv.sharding_config = colwise_config()
    ve_cfg.attn_proj.sharding_config = rowwise_config()
    ve_cfg.mlp_fc1.sharding_config = colwise_config()
    ve_cfg.mlp_fc2.sharding_config = rowwise_config()
    ve_cfg.merger_fc1.sharding_config = colwise_config()
    ve_cfg.merger_fc2.sharding_config = rowwise_config()

    # local_map on VisionInnerAttention: fused qkv arrives as
    # DTensor(Shard(-1)) on TP from attn.qkv. local_map unwraps to a plain
    # local with shape (N, L, dim*3/TP); the reshape + permute + unbind +
    # rope + flex_attention all run on local heads-parallel tensors. Output
    # (N, L, n_heads, head_dim) re-wraps to DTensor(Shard(2)) which the
    # caller reshapes into Shard(-1) for the rowwise self.proj.
    # qkv arrives as ``DTensor(Shard(2))`` from attn.qkv (colwise on the last
    # dim of a 3D activation -- use explicit ``Shard(2)`` to match local_map's
    # strict placement-equality check).
    qkv_placements = dense_activation_placement(tp=Shard(2))
    out_placements = dense_activation_placement(tp=Shard(2))
    ve_cfg.attn_inner_attention.sharding_config = ShardingConfig(
        in_dst_shardings={"qkv": qkv_placements},
        out_src_shardings=out_placements,
        local_map=LocalMapConfig(in_grad_placements=(qkv_placements,)),
    )

    vision_norm = ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=Replicate()),
            "bias": dense_param_placement(tp=Replicate()),
        },
    )
    ve_cfg.block_norm.sharding_config = vision_norm
    ve_cfg.merger_norm.sharding_config = vision_norm


def _replicate_linear_config() -> ShardingConfig:
    """Linear with all-Replicate weight on TP, used for ``patch_embed.proj``.

    Wraps the plain ``pixel_values`` input as ``DTensor(Replicate)`` so the
    rest of the encoder runs in DTensor space.
    """
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=Replicate()),
            "bias": dense_param_placement(tp=Replicate()),
        },
        in_src_shardings={"input": dense_activation_placement(tp=Replicate())},
        in_dst_shardings={"input": dense_activation_placement(tp=Replicate())},
        out_dst_shardings=dense_activation_placement(tp=Replicate()),
    )
