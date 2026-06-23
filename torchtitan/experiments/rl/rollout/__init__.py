# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: `Rollouter` is intentionally NOT re-exported here. It imports `environment`,
# which imports `rollout.types` — re-exporting it from this package `__init__` would make
# that a circular import. Import it from the submodule: `rollout.rollouter import Rollouter`.
from torchtitan.experiments.rl.rollout.types import (
    Rollout,
    RolloutGroup,
    RolloutStatus,
    RolloutTurn,
)
from torchtitan.experiments.rl.rollout.utils import prepare_rollout_metrics

__all__ = [
    "Rollout",
    "RolloutGroup",
    "RolloutStatus",
    "RolloutTurn",
    "prepare_rollout_metrics",
]
