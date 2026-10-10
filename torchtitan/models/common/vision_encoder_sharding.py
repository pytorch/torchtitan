# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding configs for common vision encoder components."""

from dataclasses import replace
from typing import TYPE_CHECKING

import spmd_types as spmd
from spmd_types import SpmdType

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.protocols.sharding import ShardingConfig

if TYPE_CHECKING:
    from torchtitan.models.common.vision_encoder import VisionTransformerBlock
    from torchtitan.protocols.module import Module


DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP


def _vision_state_placement(
    *,
    tp: spmd.PerMeshAxisSpmdType,
) -> SpmdType:
    return SpmdType({DP: spmd.R, CP: spmd.R, TP: tp})


def _vision_activation_placement(
    *,
    dp: spmd.PerMeshAxisSpmdType = spmd.V,
    tp: spmd.PerMeshAxisSpmdType = spmd.I,
) -> SpmdType:
    return SpmdType({DP: dp, CP: spmd.R, TP: tp})


def multimodal_input_sharding(
    *, cp: spmd.PerMeshAxisSpmdType = spmd.R
) -> dict[str, SpmdType]:
    """SPMD layouts for VLM vision inputs (folded into a model's input_sharding).

    The vision tensors are DP-local (``V@DP``) -- each DP rank owns its own
    images -- and TP-invariant (``I@TP``): the model consumes them inside
    ``spmd_local_context("dp")`` (a DP-local mesh) and the vision encoder runs per-rank.
    Shared by every VLM decoder (Qwen3.5, Kimi K2.5, Muse Glimmer).
    """
    layout = SpmdType({DP: spmd.V, CP: cp, TP: spmd.I})
    return {
        "pixel_values": layout,
        "pixel_values_videos": layout,
        "grid_thw": layout,
        "grid_thw_videos": layout,
    }


def set_vision_encoder_cp_invariant(config: "Module.Config") -> None:
    """Keep encoder computation CP-invariant.

    Apply after the encoder's regular sharding plan. The caller must add an
    output I -> R boundary and exclude CP from the encoder's FSDP axes.
    """
    from torchtitan.protocols.module import Module

    def invariant(layout: SpmdType) -> SpmdType:
        assert layout.local_type[CP] in (spmd.R, spmd.I)
        return SpmdType(
            {**layout.local_type, CP: spmd.I},
            partition_spec=layout.partition_spec,
        )

    def inputs(layouts: dict[str, SpmdType] | None) -> dict[str, SpmdType] | None:
        return (
            {name: invariant(layout) for name, layout in layouts.items()}
            if layouts is not None
            else None
        )

    for _, child, _, _ in config.traverse(Module.Config, recurse=True):
        sharding = child.sharding_config
        if sharding is None:
            continue
        out_src = sharding.out_src_shardings
        child.sharding_config = replace(
            sharding,
            state_shardings={
                name: invariant(layout)
                for name, layout in sharding.state_shardings.items()
            },
            in_src_shardings=inputs(sharding.in_src_shardings),
            in_dst_shardings=inputs(sharding.in_dst_shardings),
            out_src_shardings=(
                tuple(invariant(layout) for layout in out_src)
                if isinstance(out_src, tuple)
                else invariant(out_src)
                if out_src is not None
                else None
            ),
            out_dst_shardings=(
                invariant(sharding.out_dst_shardings)
                if sharding.out_dst_shardings is not None
                else None
            ),
        )


def invariant_norm_config() -> ShardingConfig:
    """Norm whose state and activations are invariant across TP ranks."""
    return ShardingConfig(
        state_shardings={
            "weight": _vision_state_placement(tp=spmd.I),
            "bias": _vision_state_placement(tp=spmd.I),
        },
        in_src_shardings={
            "input": _vision_activation_placement(),
        },
        in_dst_shardings={
            "input": _vision_activation_placement(),
        },
        out_src_shardings=_vision_activation_placement(),
        out_dst_shardings=_vision_activation_placement(),
    )


def vision_invariant_linear_config() -> ShardingConfig:
    """Unsharded linear whose state and activations are invariant at TP."""
    return ShardingConfig(
        state_shardings={
            "weight": _vision_state_placement(tp=spmd.I),
            "bias": _vision_state_placement(tp=spmd.I),
        },
        in_src_shardings={
            "input": _vision_activation_placement(),
        },
        in_dst_shardings={
            "input": _vision_activation_placement(),
        },
        out_src_shardings=_vision_activation_placement(),
        out_dst_shardings=_vision_activation_placement(),
    )


def vision_colwise_config(
    *,
    input_tp: spmd.PerMeshAxisSpmdType = spmd.I,
) -> ShardingConfig:
    """Colwise vision linear with a TP-replicated local matmul input."""
    return ShardingConfig(
        state_shardings={
            "weight": _vision_state_placement(tp=spmd.S(0)),
            "bias": _vision_state_placement(tp=spmd.S(0)),
        },
        in_src_shardings={
            "input": _vision_activation_placement(tp=input_tp),
        },
        in_dst_shardings={
            "input": _vision_activation_placement(tp=spmd.R),
        },
        out_src_shardings=_vision_activation_placement(tp=spmd.S(-1)),
    )


def vision_rowwise_config() -> ShardingConfig:
    """Sharding contract for an invariant-output vision row projection."""
    input_layout = _vision_activation_placement(tp=spmd.S(1))
    return ShardingConfig(
        state_shardings={
            "weight": _vision_state_placement(tp=spmd.S(1)),
            "bias": _vision_state_placement(tp=spmd.I),
        },
        in_src_shardings={
            "input": input_layout,
        },
        in_dst_shardings={
            "input": input_layout,
        },
        out_src_shardings=_vision_activation_placement(),
    )


def set_vision_transformer_block_sharding_config(
    block: "VisionTransformerBlock.Config",
    *,
    rope_cache_dp: spmd.PerMeshAxisSpmdType,
) -> None:
    """Set TP sharding for the common vision transformer block."""
    block.norm1.sharding_config = invariant_norm_config()
    block.norm2.sharding_config = invariant_norm_config()

    block.attn.sharding_config = ShardingConfig(
        in_src_shardings={
            "x": _vision_activation_placement(),
            "rope_cache": _vision_activation_placement(dp=rope_cache_dp),
        },
        in_dst_shardings={
            "x": _vision_activation_placement(tp=spmd.R),
            "rope_cache": _vision_activation_placement(
                dp=rope_cache_dp,
                tp=spmd.R,
            ),
        },
    )
    block.attn.wq.sharding_config = vision_colwise_config(input_tp=spmd.R)
    block.attn.wk.sharding_config = vision_colwise_config(input_tp=spmd.R)
    block.attn.wv.sharding_config = vision_colwise_config(input_tp=spmd.R)
    block.attn.proj.sharding_config = vision_rowwise_config()
    attention_layout = _vision_activation_placement(tp=spmd.S(1))
    block.attn.inner_attention.sharding_config = ShardingConfig(
        in_src_shardings={
            "q_THK": attention_layout,
            "k_THK": attention_layout,
            "v_THV": attention_layout,
        },
        in_dst_shardings={
            "q_THK": attention_layout,
            "k_THK": attention_layout,
            "v_THV": attention_layout,
        },
        out_src_shardings=attention_layout,
        local_spmd=True,
    )

    block.mlp.fc1.sharding_config = vision_colwise_config()
    block.mlp.fc2.sharding_config = vision_rowwise_config()
