# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .flex_shard import (
    BucketParamStorageLayout,
    BucketSpec,
    BucketStorageLayout,
    disable_flex_shard_gradient_division,
    flex_shard,
    get_global_shape,
    get_placements,
    is_flex_shard_param,
    LocalStorageLayout,
    MixedPrecisionPolicy,
    OffloadPolicy,
    Placement,
    PlacementFn,
    set_flex_shard_gradient_reduce_op,
)


__all__ = [
    "BucketParamStorageLayout",
    "BucketSpec",
    "BucketStorageLayout",
    "disable_flex_shard_gradient_division",
    "flex_shard",
    "get_global_shape",
    "get_placements",
    "is_flex_shard_param",
    "LocalStorageLayout",
    "MixedPrecisionPolicy",
    "OffloadPolicy",
    "Placement",
    "PlacementFn",
    "set_flex_shard_gradient_reduce_op",
]
