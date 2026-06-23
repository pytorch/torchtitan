# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .fp8_ragged_shard import (
    blockwise_dequant_weight,
    blockwise_quant_weight,
    blockwise_transpose,
    Fp8BlockwiseGroupedRaggedShard,
    Fp8TwoOrientationGroupedRaggedShard,
    make_fp8_blockwise_grouped_ragged_placement_fn,
    make_fp8_two_orientation_grouped_ragged_placement_fn,
    promote_to_square_block,
)
from .muon import (
    build_comm_free_muon_optimizers,
    build_muon_param_groups,
    build_ragged_shard_muon_optimizers,
    CombinedOptimizer,
    comm_free_muon_buckets,
    grouped_ragged_shard_muon_buckets,
    GroupedMuon,
    RaggedShardMuon,
)
from .owned import (
    assign_layer_owners_lpt,
    assign_matrix_owners_per_layer_balanced,
    GroupedOwned,
    GroupedOwnedSegmentSpec,
    make_grouped_owned_expert_block_placement_fn,
    make_grouped_owned_expert_block_segments,
    make_owned_placement_fn,
    Owned,
    param_boundary_placements,
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
    "assign_layer_owners_lpt",
    "assign_matrix_owners_per_layer_balanced",
    "blockwise_dequant_weight",
    "blockwise_quant_weight",
    "blockwise_transpose",
    "build_comm_free_muon_optimizers",
    "build_muon_param_groups",
    "build_ragged_shard_muon_optimizers",
    "CombinedOptimizer",
    "comm_free_muon_buckets",
    "Fp8BlockwiseGroupedRaggedShard",
    "Fp8TwoOrientationGroupedRaggedShard",
    "GroupedOwned",
    "GroupedOwnedSegmentSpec",
    "GroupedMuon",
    "GroupedRaggedShard",
    "grouped_ragged_shard_muon_buckets",
    "make_fp8_blockwise_grouped_ragged_placement_fn",
    "make_fp8_two_orientation_grouped_ragged_placement_fn",
    "make_grouped_owned_expert_block_placement_fn",
    "make_grouped_owned_expert_block_segments",
    "make_grouped_ragged_placement_fn",
    "promote_to_square_block",
    "make_owned_placement_fn",
    "make_ragged_placement_fn",
    "make_shard_placement_fn",
    "Owned",
    "param_boundary_placements",
    "per_param_placements",
    "per_param_ragged_placements",
    "RaggedShard",
    "RaggedShardMuon",
    "Shard",
]
