# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What a rollout worker sends back for a finished rollout group: its summary, which the batcher
decides and plans from, and the group packed into one buffer (`packed_group.py`), which the
controller keeps until the group's step and the trainer's DP replica leaders unpack."""

from __future__ import annotations

from dataclasses import dataclass, replace

from torchtitan.rl.controller.metrics import compute_rollout_metrics
from torchtitan.rl.rollout.packed_group import pack_rollout_group, PackedRolloutGroup
from torchtitan.rl.rollout.recorder import RolloutSampleEncoder
from torchtitan.rl.rollout.types import (
    check_prompt_prefixes,
    RolloutGroup,
    RolloutGroupSummary,
)


@dataclass(frozen=True, slots=True)
class FinishedRolloutGroup:
    """A finished rollout group, as the rollout worker returns it: its summary for the batcher,
    and the whole group packed into one buffer for the controller to keep and forward
    (`pack_rollout_group`).

    For training, the worker has already logged and recorded the full group, so the group keeps
    only what training needs (see `slim_rollout_group`). For validation, it is the full group and
    the controller unpacks and records it.
    """

    summary: RolloutGroupSummary

    packed: PackedRolloutGroup

    recorder_lines: str = ""
    """The group's `RolloutSampleRecorder` JSONL lines, encoded from the full group; "" for
    validation."""

    @classmethod
    def from_group(
        cls, group: RolloutGroup, *, recorder_lines: str = ""
    ) -> FinishedRolloutGroup:
        return cls(*pack_rollout_group(group), recorder_lines=recorder_lines)

    @property
    def group_id(self) -> int:
        return self.summary.group.group_id


def slim_rollout_group(group: RolloutGroup) -> RolloutGroup:
    """Return a copy of `group` without the fields only logging and the recorder read.

    Drops each rollout's `reward_breakdown` and `logs` and each turn's messages, `env_rewards`
    and `metrics`. It names only what it drops, so every other field survives, including
    per-token arrays that a feature adds to `RolloutTurn` (or a subclass of it).

    Example:
        group.rollouts[0].turns[0].prompt_messages   # [{"role": "user", "content": "sort..."}]
        slim_rollout_group(group).rollouts[0].turns[0].prompt_messages   # []
    """
    return replace(
        group,
        rollouts=[
            replace(
                rollout,
                reward_breakdown={},
                logs={},
                turns=[
                    replace(
                        turn,
                        prompt_messages=[],
                        completion_message=None,
                        env_messages=[],
                        env_rewards={},
                        metrics=[],
                    )
                    for turn in rollout.turns
                ],
            )
            for rollout in group.rollouts
        ],
    )


def finish_rollout_group(
    group: RolloutGroup, *, for_training: bool, encoder: RolloutSampleEncoder
) -> FinishedRolloutGroup:
    """Turn a scored group into what the rollout worker returns.

    For training: check its rollouts' prompt prefixes (`check_prompt_prefixes`, the one check
    that reads token values, so it runs where the arrays are), compute the group's rollout
    metrics and recorder lines from the full group, then slim it and pack it. Validation groups
    are packed as is: the controller unpacks them, computes their metrics and records them.
    """
    # TODO: finish validation groups here too (encode them with `is_validation=True`; the
    # controller then only prefixes and renames their metrics), and drop `for_training` and
    # `RolloutSampleRecorder.record`.
    if not for_training:
        return FinishedRolloutGroup.from_group(group)
    for rollout in group.rollouts:
        check_prompt_prefixes(rollout)
    group.metrics.extend(
        compute_rollout_metrics(prefix="rollout", rollouts=group.rollouts)
    )
    recorder_lines = encoder.encode(is_validation=False, rollout_groups=[group])
    return FinishedRolloutGroup.from_group(
        slim_rollout_group(group), recorder_lines=recorder_lines
    )
