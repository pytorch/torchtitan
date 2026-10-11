# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muse Glimmer-specific sharding configs.

``flavors.py`` sets these on each sub-config as it builds the model. Text-only
and multimodal models use the standard decoder activation layout. With SP,
each TP rank gathers vision rows for its local token shard, so the vision bank
becomes TP-replicated before fusion. Vision activations are invariant across
TP and replicated across CP; the vision sharding does not depend on
parallelism.
"""

import spmd_types as spmd
from spmd_types import SpmdType

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.decoder_sharding import (
    attention_activation_placement,
    residual_activation_placement,
)
from torchtitan.models.common.vision_encoder_sharding import invariant_norm_config


DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP

_VISION_STATE_INVARIANT = SpmdType({DP: spmd.R, CP: spmd.R, TP: spmd.I})
_VISION_ACTIVATION_INVARIANT = SpmdType({DP: spmd.V, CP: spmd.R, TP: spmd.I})


def vision_bank_indices_placement(*, enable_sp: bool) -> SpmdType:
    """Placement for token-aligned indices into the packed vision bank."""
    token_axes = (DP, CP, TP) if enable_sp else (DP, CP)
    return SpmdType(
        {
            DP: spmd.V,
            CP: spmd.V,
            TP: spmd.V if enable_sp else spmd.I,
        },
        partition_spec=spmd.PartitionSpec(token_axes),
    )


def attention_config(*, enable_sp: bool) -> ShardingConfig:
    """Attention input boundary shared by ``qkv_linear`` and ``o_gate``."""
    return ShardingConfig(
        in_src_shardings={"x_TD": residual_activation_placement(enable_sp=enable_sp)},
    )


def scaleless_head_norm_config() -> ShardingConfig:
    """QK norm sharded on the head dim, independent of SP.

    Scaleless, so there is no weight state to distribute.
    """
    head_shard = attention_activation_placement()
    return ShardingConfig(
        in_src_shardings={"input": head_shard},
        in_dst_shardings={"input": head_shard},
        out_src_shardings=head_shard,
        out_dst_shardings=head_shard,
    )


def perception_emb_norm_config(*, enable_sp: bool) -> ShardingConfig:
    """Scaleless norm on the projected vision features before fusion.

    With SP, its output is gathered to TP-replicated so each rank can select
    vision rows for its local token shard.
    """
    vision_norm = invariant_norm_config()
    if enable_sp:
        vision_norm.out_dst_shardings = SpmdType({DP: spmd.V, CP: spmd.R, TP: spmd.R})
    return vision_norm


def vision_encoder_config() -> ShardingConfig:
    """Keep the learned positional grid Replicate.

    The grid is a raw nn.Parameter on the encoder and is bilinearly resampled
    per image (see ``_get_pos_emb``).
    """
    return ShardingConfig(
        state_shardings={"positional_embedding_vlm": _VISION_STATE_INVARIANT},
    )


def vision_rope_freq_config() -> ShardingConfig:
    """TP-invariant vision RoPE frequencies."""
    return ShardingConfig(state_shardings={"inv_freq": _VISION_STATE_INVARIANT})


def vision_pos_embed_config() -> ShardingConfig:
    """Local positional-embedding resampling with invariant input and output."""
    return ShardingConfig(
        in_src_shardings={"pos_param": _VISION_STATE_INVARIANT},
        in_dst_shardings={"pos_param": _VISION_STATE_INVARIANT},
        out_src_shardings=_VISION_ACTIVATION_INVARIANT,
        local_spmd=True,
    )


def vision_token_permute_config() -> ShardingConfig:
    """Local token permutation with invariant input and output."""
    return ShardingConfig(
        in_src_shardings={
            "x": _VISION_ACTIVATION_INVARIANT,
            "index": _VISION_ACTIVATION_INVARIANT,
        },
        in_dst_shardings={
            "x": _VISION_ACTIVATION_INVARIANT,
            "index": _VISION_ACTIVATION_INVARIANT,
        },
        out_src_shardings=_VISION_ACTIVATION_INVARIANT,
        local_spmd=True,
    )
