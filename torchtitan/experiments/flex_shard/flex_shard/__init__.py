# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .bucket_storage import BucketSpec, MixedPrecisionPolicy, OffloadPolicy, PlacementFn
from .flex_shard import flex_shard
from .placement_contract import (
    BucketParamStorageLayout,
    BucketStorageLayout,
    LocalStorageLayout,
    Placement,
)
from .segment_reduce import (
    CoalescedSegmentReduceDescriptor,
    PlannedSegmentReduceDescriptor,
    SegmentReduceDescriptor,
    SegmentReducePlan,
    SegmentReduceResult,
    build_segment_reduce_plan,
    owned_segment_views,
    pack_segment_reduce_scatter_input,
    segment_descriptors_from_offsets,
    segment_reduce_to_owner,
    segment_reduce_to_owner_with_plan,
)
from .sharded_param import get_global_shape, get_placements, is_flex_shard_param

__all__ = [
    "BucketParamStorageLayout",
    "BucketSpec",
    "BucketStorageLayout",
    "CoalescedSegmentReduceDescriptor",
    "flex_shard",
    "get_global_shape",
    "get_placements",
    "is_flex_shard_param",
    "LocalStorageLayout",
    "MixedPrecisionPolicy",
    "OffloadPolicy",
    "Placement",
    "PlacementFn",
    "PlannedSegmentReduceDescriptor",
    "SegmentReduceDescriptor",
    "SegmentReducePlan",
    "SegmentReduceResult",
    "build_segment_reduce_plan",
    "owned_segment_views",
    "pack_segment_reduce_scatter_input",
    "segment_descriptors_from_offsets",
    "segment_reduce_to_owner",
    "segment_reduce_to_owner_with_plan",
]
