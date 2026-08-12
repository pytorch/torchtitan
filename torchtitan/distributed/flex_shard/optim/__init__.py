# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optimizers implemented with FlexShard."""

from .distributed_muon import (
    BatchedMatrixComputeView,
    build_distributed_muon,
    DistributedMuon,
    flex_optimizer_reshard,
    MuonComputeShardingConfig,
)


__all__ = [
    "BatchedMatrixComputeView",
    "build_distributed_muon",
    "DistributedMuon",
    "flex_optimizer_reshard",
    "MuonComputeShardingConfig",
]
