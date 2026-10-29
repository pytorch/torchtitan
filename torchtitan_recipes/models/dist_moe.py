# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared helpers for recipes selecting the optional Dist-MoE backend."""

from importlib import import_module
from types import ModuleType

from torchtitan.config import Configurable


def require_dist_moe() -> ModuleType:
    """Import the optional Dist-MoE package for a selected recipe."""
    try:
        return import_module("dist_moe")
    except ModuleNotFoundError as error:
        if error.name != "dist_moe":
            raise
        raise ModuleNotFoundError(
            "This recipe requires the optional dist_moe package. Install "
            "it from git+https://github.com/meta-pytorch/dist_moe.git@main "
            "before selecting it.",
            name="dist_moe",
        ) from error


def dist_moe_runtime_config(*, scratch_capacity_factor: float) -> Configurable.Config:
    """Build the rank-wide activation and scratch policy for Dist-MoE."""
    from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime

    return DistMoeRuntime.Config(
        activation_slot_capacity_factor=1.0,
        scratch_capacity_factor=scratch_capacity_factor,
    )
