# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .storage import (
    auto_buckets,
    BucketSpec,
    MixedPrecisionPolicy,
)
from .flex_shard import flex_shard
from .metadata import (
    get_global_shape,
    get_placements,
    is_flex_shard_param,
    set_sharding_info,
)
from .placements import (
    per_param_placements,
    Placement,
    Shard,
)
from .utils import disable_active_parametrization


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
