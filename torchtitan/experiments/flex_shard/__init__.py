# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .flex_shard import (
    auto_buckets,
    BucketSpec,
    disable_active_parametrization,
    DStorage,
    flat_shard_placements,
    FlatShard,
    FlatShardParametrization,
    flex_shard,
    FlexShardModule,
    get_global_shape,
    get_placements,
    is_flex_shard_param,
    Owned,
    param_boundary_placements,
    per_param_placements,
    Placement,
    RaggedShard,
    set_sharding_info,
    Shard,
    ShardParametrization,
)


__all__ = [
    "auto_buckets",
    "BucketSpec",
    "disable_active_parametrization",
    "DStorage",
    "flat_shard_placements",
    "FlatShard",
    "FlatShardParametrization",
    "flex_shard",
    "FlexShardModule",
    "get_global_shape",
    "get_placements",
    "is_flex_shard_param",
    "Owned",
    "param_boundary_placements",
    "per_param_placements",
    "Placement",
    "RaggedShard",
    "set_sharding_info",
    "Shard",
    "ShardParametrization",
]
