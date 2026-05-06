# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .flex_shard import (
    auto_buckets,
    BucketSpec,
    flex_shard,
    get_global_shape,
    get_placements,
    is_flex_shard_param,
    MixedPrecisionPolicy,
    set_sharding_info,
)
from .parametrizations import disable_active_parametrization
from .placements import (
    per_param_placements,
    Placement,
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
    "MixedPrecisionPolicy",
    "per_param_placements",
    "Placement",
    "set_sharding_info",
    "Shard",
]
