# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .flex_shard import (
    auto_buckets,
    BucketSpec,
    disable_active_parametrization,
    flex_shard,
    get_global_shape,
    get_placements,
    is_flex_shard_param,
    lift_params_to_global_spmd_mesh,
    MixedPrecisionPolicy,
    per_param_placements,
    Placement,
    set_sharding_info,
    Shard,
)


__all__ = [
    "auto_buckets",
    "BucketSpec",
    "disable_active_parametrization",
    "flex_shard",
    "get_global_shape",
    "get_placements",
    "is_flex_shard_param",
    "lift_params_to_global_spmd_mesh",
    "MixedPrecisionPolicy",
    "per_param_placements",
    "Placement",
    "set_sharding_info",
    "Shard",
]
