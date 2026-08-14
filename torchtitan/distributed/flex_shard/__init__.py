# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flexible storage-to-compute redistribution APIs."""

from .dist_muon import (
    AttentionPerHeadComputeView,
    build_dist_muon,
    MuonComputeShardingConfig,
)
from .optimizer_reshard import BucketConfig, ComputeLayout, Owned

__all__ = [
    "AttentionPerHeadComputeView",
    "build_dist_muon",
    "BucketConfig",
    "ComputeLayout",
    "MuonComputeShardingConfig",
    "Owned",
]
