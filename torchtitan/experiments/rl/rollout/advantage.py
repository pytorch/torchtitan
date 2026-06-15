# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Advantage estimator: a post-scoring step run by the ``Rollouter``.

After a group is scored, this turns the group's rewards into per-rollout advantages
(in group order) that the trainer/loss consumes directly.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from torchtitan.config import Configurable
from torchtitan.experiments.rl.rollout.types import RolloutGroup


class GRPOAdvantage(Configurable):
    """Group-relative advantage: ``A_i = (r_i - mean(r)) / denom``.

    ``denom = std(r) + eps`` when ``should_std_normalize`` (standard GRPO), else
    ``1.0`` (Dr.GRPO mean-baseline).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        should_std_normalize: bool = False
        """Divide the centered advantage by the group reward std (+eps) — standard GRPO.
        False (default) = mean-baseline only (Dr.GRPO)."""

    def __init__(self, config: Config) -> None:
        self.should_std_normalize: bool = config.should_std_normalize

    def __call__(self, group: RolloutGroup) -> list[float]:
        """Return per-rollout advantages, in group order."""
        rewards = [rollout.reward for rollout in group.rollouts]
        group_mean = sum(rewards) / len(rewards)
        denom = (
            (statistics.pstdev(rewards) + 1e-6) if self.should_std_normalize else 1.0
        )
        return [(reward - group_mean) / denom for reward in rewards]
