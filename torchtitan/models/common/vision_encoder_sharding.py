# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding configs for common vision encoder components."""

import spmd_types as spmd
from spmd_types import SpmdType

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.base.sharding import ShardingConfig

DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP


def _vision_state_placement(
    *,
    tp: spmd.PerMeshAxisSpmdType,
    cp: spmd.PerMeshAxisSpmdType,
) -> SpmdType:
    return SpmdType({DP: spmd.R, CP: cp, TP: tp})


def _vision_activation_placement(
    *,
    dp: spmd.PerMeshAxisSpmdType = spmd.V,
    tp: spmd.PerMeshAxisSpmdType = spmd.I,
    cp: spmd.PerMeshAxisSpmdType = spmd.I,
) -> SpmdType:
    return SpmdType({DP: dp, CP: cp, TP: tp})


def multimodal_input_sharding() -> dict[str, SpmdType]:
    """SPMD layouts for VLM vision inputs (folded into a model's input_sharding).

    The vision tensors are DP-local (``V@DP``) -- each DP rank owns its own
    images -- and invariant across TP and CP (``I@TP``, ``I@CP``). The model
    consumes them inside ``spmd_local_context("dp")``. Each CP rank processes
    the full vision input; the encoder can use TP internally.
    Shared by every VLM decoder (Qwen3.5, Kimi K2.5, Kimi K3, Muse Glimmer).
    """
    layout = _vision_activation_placement()
    return {
        "pixel_values": layout,
        "pixel_values_videos": layout,
        "grid_thw": layout,
        "grid_thw_videos": layout,
    }


def invariant_norm_sharding_config() -> ShardingConfig:
    """Norm whose state and activations are invariant across TP and CP ranks."""
    return ShardingConfig(
        state_shardings={
            "weight": _vision_state_placement(tp=spmd.I, cp=spmd.I),
            "bias": _vision_state_placement(tp=spmd.I, cp=spmd.I),
        },
        input_shardings={
            "input": _vision_activation_placement(),
        },
        output_shardings=_vision_activation_placement(),
    )


def vision_invariant_linear_sharding_config() -> ShardingConfig:
    """Unsharded linear whose state and activations are invariant at TP and CP."""
    return ShardingConfig(
        state_shardings={
            "weight": _vision_state_placement(tp=spmd.I, cp=spmd.I),
            "bias": _vision_state_placement(tp=spmd.I, cp=spmd.I),
        },
        input_shardings={
            "input": _vision_activation_placement(),
        },
        output_shardings=_vision_activation_placement(),
    )


def vision_colwise_sharding_config(
    *,
    input_tp: spmd.PerMeshAxisSpmdType = spmd.I,
) -> ShardingConfig:
    """Colwise vision linear with a TP-replicated local matmul input.

    ``input_tp`` is the input's TP layout at the module boundary: ``I`` for a
    ``ColumnParallelLinear``, which replicates it itself (vision encoders run
    under ``disable_dense_sp``), or ``R`` behind a shared boundary that already
    did (the attention q/k/v).
    """
    return ShardingConfig(
        state_shardings={
            "weight": _vision_state_placement(tp=spmd.S(0), cp=spmd.I),
            "bias": _vision_state_placement(tp=spmd.S(0), cp=spmd.I),
        },
        input_shardings={
            "input": _vision_activation_placement(tp=input_tp),
        },
        output_shardings=_vision_activation_placement(tp=spmd.S(-1)),
    )


def vision_rowwise_sharding_config() -> ShardingConfig:
    """Sharding contract for an invariant-output vision row projection."""
    input_layout = _vision_activation_placement(tp=spmd.S(1))
    return ShardingConfig(
        state_shardings={
            "weight": _vision_state_placement(tp=spmd.S(1), cp=spmd.I),
            "bias": _vision_state_placement(tp=spmd.I, cp=spmd.I),
        },
        input_shardings={
            "input": input_layout,
        },
        output_shardings=_vision_activation_placement(),
    )


def vision_attention_sharding_config(
    *,
    rope_cache_dp: spmd.PerMeshAxisSpmdType,
) -> ShardingConfig:
    """Vision attention boundary: TP-invariant ``x``, TP-replicated ``rope_cache``.

    ``VisionAttention`` replicates ``x`` across TP itself. The rope cache is
    already TP-``Replicate`` (like the decoder's), so it combines with the
    head-sharded q/k.
    """
    return ShardingConfig(
        input_shardings={
            "x": _vision_activation_placement(),
            "rope_cache": _vision_activation_placement(dp=rope_cache_dp, tp=spmd.R),
        },
    )


def vision_inner_attention_sharding_config() -> ShardingConfig:
    """Local-SPMD vision attention kernel with heads sharded across TP."""
    attention_layout = _vision_activation_placement(tp=spmd.S(1))
    return ShardingConfig(
        input_shardings={
            "q_THK": attention_layout,
            "k_THK": attention_layout,
            "v_THV": attention_layout,
        },
        output_shardings=attention_layout,
        local_spmd=True,
    )
