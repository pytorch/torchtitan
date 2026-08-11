# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed Muon optimizer, parameter preparation, and redistribution."""

from .distributed_muon import DistributedMuon
from .prep_parameters import (
    BatchedMatrixComputeView,
    build_distributed_muon,
    MuonComputeShardingConfig,
    Owned,
)


__all__ = [
    "BatchedMatrixComputeView",
    "build_distributed_muon",
    "DistributedMuon",
    "MuonComputeShardingConfig",
    "Owned",
]
