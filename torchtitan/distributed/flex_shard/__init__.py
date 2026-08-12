# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flexible storage-to-compute redistribution APIs."""

from .optim.distributed_muon import flex_optimizer_reshard
from .optimizer_reshard import BucketConfig, ComputeLayout, FlexOptimizer

# Keep each user-facing reshard API in its own module. A future model API should
# use model_reshard.py with _model_reshard_schedule.py and
# _model_reshard_runtime.py.

__all__ = [
    "BucketConfig",
    "ComputeLayout",
    "FlexOptimizer",
    "flex_optimizer_reshard",
]
