# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3 sharding scaffold.

This file is intentionally strategy-free. Autoresearch is expected to replace
``set_qwen3_sharding_config`` with train-command-specific tensor placement
contracts for the target model flavor and machine or cluster.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchtitan.models.qwen3.model import Qwen3Model


def set_qwen3_sharding_config(
    config: "Qwen3Model.Config",
    *,
    loss_parallel: bool,
    enable_sp: bool,
) -> None:
    """Generated machine-specific Qwen3 sharding configuration hook.

    A valid implementation should attach ``sharding_config`` objects to the
    relevant Qwen3 configs before ``parallelize_qwen3`` calls
    ``model.parallelize(...)``. It may make narrow assumptions about the exact
    train command, model flavor, mesh shape, and hardware being optimized.
    """
    return None
