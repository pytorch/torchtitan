# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding configs for Kimi K2.5 (MoonViT3d + DeepSeekV3).

Declarations only: ``flavors.py`` sets these on each sub-config as it builds
it, and ``model.parallelize()`` applies TP/EP/SP through the Module protocol.

- Decoder (MLA + MoE): built with DeepSeek V3's sharded layer builder.
  Multimodal configs keep the token embedding ``Replicate`` for the vision
  scatter and resume SP at layer 0 (see ``sequence_parallel_first_layer_sharding_config``).
- Vision encoder: activations flow ``Invariant`` (no SP -- the patch sequence is
  short, so sequence-sharding would add gather/scatter around the block-diagonal
  attention for little memory gain). Only the linear layers are Colwise/Rowwise
  sharded for memory; norms and position embeddings stay ``Invariant``.
"""

import spmd_types as spmd
from spmd_types import SpmdType

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.decoder_sharding import (
    dense_sequence_parallel_placement,
    replicated_activation_placement,
)

DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP


def sequence_parallel_first_layer_sharding_config() -> ShardingConfig:
    """Layer 0 boundary that resumes SP after a ``Replicate`` embedding.

    Layer 0 takes a ``Replicate`` input and its rowwise ``wo`` reduce-scatters
    back to ``Shard(0)``, so the residual is sequence-parallel from layer 0's
    output and layers ``1..N-1`` are unchanged full SP.
    """
    return ShardingConfig(
        in_src_shardings={"x": replicated_activation_placement()},
        in_dst_shardings={"x": dense_sequence_parallel_placement()},
        out_src_shardings=dense_sequence_parallel_placement(),
    )


def _moonvit_state_placement() -> SpmdType:
    return SpmdType({DP: spmd.R, CP: spmd.I, TP: spmd.I})


def moonvit_encoder_sharding_config() -> ShardingConfig:
    """MoonViT3d encoder boundary; its ``pos_embed`` table is TP-invariant.

    ``patch_embed`` wraps the plain ``pixel_values`` input as a TP-invariant
    tensor so the rest of the encoder runs in distributed tensor space.
    """
    return ShardingConfig(
        state_shardings={"pos_embed": _moonvit_state_placement()},
        out_src_shardings=SpmdType({DP: spmd.V, CP: spmd.I, TP: spmd.R}),
        out_dst_shardings=SpmdType({DP: spmd.V, CP: spmd.R, TP: spmd.R}),
    )


def moonvit_projector_sharding_config() -> ShardingConfig:
    """Keep TP and CP on separate boundaries, with CP reduced first in backward.

    The encoder returns the projector output directly and owns the CP boundary.
    """
    return ShardingConfig(
        out_src_shardings=SpmdType({DP: spmd.V, CP: spmd.I, TP: spmd.I}),
        out_dst_shardings=SpmdType({DP: spmd.V, CP: spmd.I, TP: spmd.R}),
    )


def moonvit_rotary_sharding_config() -> ShardingConfig:
    """TP-invariant 2D rotary embedding frequencies and output."""
    state_placement = _moonvit_state_placement()
    return ShardingConfig(
        state_shardings={"inv_freq": state_placement},
        out_src_shardings=state_placement,
    )
