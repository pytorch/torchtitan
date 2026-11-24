# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What a rollout worker adds to a rollout group it scored."""

from __future__ import annotations

from torchtitan.rl.controller.metrics import compute_rollout_metrics
from torchtitan.rl.rollout.recorder import RolloutSampleEncoder
from torchtitan.rl.rollout.types import RolloutGroup


def finish_rollout_group(
    group: RolloutGroup, *, is_validation: bool, encoder: RolloutSampleEncoder
) -> RolloutGroup:
    """Add the group's rollout metrics to `group.metrics` and encode its recorder lines into
    `group.recorder_lines`, in the rollout worker, where the group is scored; return `group`.

    Validation metrics are named `validation/*` (`compute_rollout_metrics(prefix="validation")`),
    and the lines carry `is_validation`.
    """
    group.metrics.extend(
        compute_rollout_metrics(
            prefix="validation" if is_validation else "rollout",
            rollouts=group.rollouts,
        )
    )
    group.recorder_lines = encoder.encode(
        is_validation=is_validation, rollout_groups=[group]
    )
    return group
