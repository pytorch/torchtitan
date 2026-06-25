# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .fp8_ragged_shard import (
    BlockwiseFp8Weight,
    blockwise_dequant_weight,
    blockwise_quant_weight,
    blockwise_transpose,
    convert_to_flex_shard_float8_blockwise_linear,
    FlexShardFloat8BlockwiseLinear,
    Fp8BlockwiseGroupedRaggedShard,
    Fp8TwoOrientationGroupedRaggedShard,
    make_flex_shard_float8_blockwise_all_gather_placement_fn,
    make_fp8_blockwise_grouped_ragged_placement_fn,
    make_fp8_two_orientation_grouped_ragged_placement_fn,
    promote_to_square_block,
    TORCHAO_BLOCKWISE_FP8_AVAILABLE,
    TorchAOFloat8BlockwiseLinear,
)
from .owned import (
    GroupedOwned,
    GroupedOwnedSegmentSpec,
    make_grouped_owned_expert_block_placement_fn,
    make_grouped_owned_expert_block_segments,
    make_grouped_owned_full_param_placement_fn,
    make_grouped_owned_full_param_segments,
)
from .ragged_shard import (
    GroupedRaggedShard,
    make_grouped_ragged_placement_fn,
    make_ragged_placement_fn,
    per_param_ragged_placements,
    RaggedShard,
)
from .shard import make_shard_placement_fn, per_param_placements, Shard

__all__ = [
    "BlockwiseFp8Weight",
    "blockwise_dequant_weight",
    "blockwise_quant_weight",
    "blockwise_transpose",
    "convert_to_flex_shard_float8_blockwise_linear",
    "FlexShardFloat8BlockwiseLinear",
    "Fp8BlockwiseGroupedRaggedShard",
    "Fp8TwoOrientationGroupedRaggedShard",
    "GroupedOwned",
    "GroupedOwnedSegmentSpec",
    "GroupedRaggedShard",
    "make_flex_shard_float8_blockwise_all_gather_placement_fn",
    "make_fp8_blockwise_grouped_ragged_placement_fn",
    "make_fp8_two_orientation_grouped_ragged_placement_fn",
    "make_grouped_owned_expert_block_placement_fn",
    "make_grouped_owned_expert_block_segments",
    "make_grouped_owned_full_param_placement_fn",
    "make_grouped_owned_full_param_segments",
    "make_grouped_ragged_placement_fn",
    "make_ragged_placement_fn",
    "make_shard_placement_fn",
    "per_param_ragged_placements",
    "per_param_placements",
    "promote_to_square_block",
    "RaggedShard",
    "Shard",
    "TORCHAO_BLOCKWISE_FP8_AVAILABLE",
    "TorchAOFloat8BlockwiseLinear",
]
