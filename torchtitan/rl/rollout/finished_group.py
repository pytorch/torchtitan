# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What a rollout worker sends back for a finished rollout group."""

from __future__ import annotations

from dataclasses import dataclass, replace

from torchtitan.rl.controller.metrics import compute_rollout_metrics
from torchtitan.rl.rollout.recorder import RolloutSampleEncoder
from torchtitan.rl.rollout.types import RolloutGroup


@dataclass(frozen=True, slots=True)
class FinishedRolloutGroup:
    """A finished rollout group, as the rollout worker returns it.

    For training, the worker has already logged and recorded the full group, so `group` keeps
    only what training needs (see `slim_rollout_group`). For validation, `group` is the full
    group and the controller records it.
    """

    group: RolloutGroup

    recorder_lines: str
    """The group's `RolloutSampleRecorder` JSONL lines, encoded from the full group; "" for
    validation."""


def slim_rollout_group(group: RolloutGroup) -> RolloutGroup:
    """Return a copy of `group` without the fields only logging and the recorder read, with
    one `bytes` object per image file.

    Drops each rollout's `reward_breakdown` and `logs` and each turn's messages, `env_rewards`
    and `metrics`. It names only what it drops, so every other field survives, including
    per-token arrays that a feature adds to `RolloutTurn` (or a subclass of it).

    The envs read each image file once per rollout, so a group of G rollouts holds G equal
    copies. The copy points every image with the same `hash_value` at the first one's `data`;
    pickle writes an object it has seen as a reference, so the group pickles each file once.

    Example:
        group.rollouts[0].turns[0].prompt_messages   # [{"role": "user", "content": "sort..."}]
        slim_rollout_group(group).rollouts[0].turns[0].prompt_messages   # []
    """
    data_by_hash: dict[str, bytes] = {}
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
                        prompt_delta_images=tuple(
                            replace(
                                image,
                                data=data_by_hash.setdefault(
                                    image.hash_value, image.data
                                ),
                            )
                            for image in turn.prompt_delta_images
                        ),
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

    For training: compute the group's rollout metrics and recorder lines from the full group,
    then slim it. Validation groups are returned as is: the controller computes their metrics
    and records them.
    """
    # TODO: finish validation groups here too (encode them with `is_validation=True`; the
    # controller then only prefixes and renames their metrics), and drop `for_training` and
    # `RolloutSampleRecorder.record`.
    if not for_training:
        return FinishedRolloutGroup(group=group, recorder_lines="")
    group.metrics.extend(
        compute_rollout_metrics(prefix="rollout", rollouts=group.rollouts)
    )
    recorder_lines = encoder.encode(is_validation=False, rollout_groups=[group])
    return FinishedRolloutGroup(
        group=slim_rollout_group(group), recorder_lines=recorder_lines
    )
