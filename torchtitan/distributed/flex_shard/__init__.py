# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flexible storage-to-compute redistribution APIs."""

from .muon import (
    AttentionPerHeadComputeView,
    build_flex_shard_muon,
    MuonComputeShardingConfig,
)
from .optimizer_reshard import (
    BucketConfig,
    ComputeLayout,
    flex_optimizer_reshard,
    NoRedistribution,
    Owned,
)

__all__ = [
    "AttentionPerHeadComputeView",
    "build_flex_shard_muon",
    "BucketConfig",
    "ComputeLayout",
    "flex_optimizer_reshard",
    "MuonComputeShardingConfig",
    "NoRedistribution",
    "Owned",
]
