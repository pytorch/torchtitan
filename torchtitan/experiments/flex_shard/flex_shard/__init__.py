# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .bucket_storage import (
    BucketSpec,
    MixedPrecisionPolicy,
    OffloadPolicy,
)
from .flex_shard import flex_shard
from .param_access import (
    FlexShardModule,
    get_global_shape,
    get_placements,
    is_flex_shard_param,
    set_sharding_info,
)
from .placement_contract import LocalStorageLayout, Placement

__all__ = [
    "BucketSpec",
    "flex_shard",
    "FlexShardModule",
    "get_global_shape",
    "get_placements",
    "is_flex_shard_param",
    "LocalStorageLayout",
    "MixedPrecisionPolicy",
    "OffloadPolicy",
    "Placement",
    "set_sharding_info",
]
