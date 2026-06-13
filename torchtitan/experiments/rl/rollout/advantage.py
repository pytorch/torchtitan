# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Advantage estimators: a rollout post-processing step run by the ``Rollouter``.

After a group is scored (rewards filled), the estimator turns those rewards into a
per-rollout ``advantage`` so the trainer/loss just *consumes* it and stays agnostic to
the RL algorithm's advantage definition (GRPO vs. RLOO vs. ...). Add a new estimator by
subclassing :class:`AdvantageEstimator`.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from torchtitan.config import Configurable
from torchtitan.experiments.rl.rollout.types import RolloutGroup


class AdvantageEstimator(Configurable):
    """Compute per-rollout advantages from one scored group, in place.

    Runs after scoring and before training; sets ``rollout.advantage`` on every sibling
    in the group. Subclass and implement :meth:`__call__` for a specific estimator.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __call__(self, group: RolloutGroup) -> None:
        """Set ``rollout.advantage`` in place for every rollout in ``group``."""
        raise NotImplementedError


class GRPOAdvantage(AdvantageEstimator):
    """Group-relative advantage (GRPO): ``A_i = (r_i - mean(r)) / denom``.

    ``denom = std(r) + eps`` when ``std_normalize`` (standard GRPO), else ``1.0``
    (Dr.GRPO mean-baseline). A zero-std group already has advantage 0 (reward == mean),
    so the eps only avoids a 0/0.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(AdvantageEstimator.Config):
        std_normalize: bool = False
        """Divide the centered advantage by the group reward std (+eps) — standard GRPO.
        False (default) = mean-baseline only (Dr.GRPO)."""

    def __init__(self, config: Config) -> None:
        self.std_normalize = config.std_normalize

    def __call__(self, group: RolloutGroup) -> None:
        # Runs right after scoring, so every reward is filled; guard None for typing.
        rewards = [r.reward for r in group.rollouts if r.reward is not None]
        if not rewards:
            return
        group_mean = sum(rewards) / len(rewards)
        denom = (statistics.pstdev(rewards) + 1e-6) if self.std_normalize else 1.0
        for rollout in group.rollouts:
            if rollout.reward is not None:
                rollout.advantage = (rollout.reward - group_mean) / denom
