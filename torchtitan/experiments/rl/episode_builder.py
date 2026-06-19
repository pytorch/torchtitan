# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Converts one generated rollout group into trainable episodes.

RolloutGroup -> data-validity filters -> rollout_to_episodes -> EpisodeBuilderOutput

Policy-version staleness is handled later by `EpisodeBatcher`.
"""

import statistics
from dataclasses import dataclass

from torchtitan.config import Configurable
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.rollout import rollout_to_episodes, RolloutGroup
from torchtitan.experiments.rl.types import Episode


@dataclass(frozen=True, slots=True)
class EpisodeBuilderOutput:
    """Episodes and metrics emitted for one rollout group.

    Example:
        EpisodeBuilderOutput(episodes=[], metrics=[failure_metric])
        # -> failed or filtered group; metrics still reach the trainer logger
    """

    episodes: list[Episode]
    metrics: list[m.Metric]


class EpisodeBuilder(Configurable):
    """Build trainable episodes and rollout-origin metrics from one group.

    Example:
        builder = config.episode_builder.build()
        output = builder.build_episodes(rollout_group=group)
        # output.episodes is empty for failed, untrainable, or zero-std groups.
        # output.metrics still carries the counters for that group.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Episode builder filtering knobs."""

        drop_zero_std_reward_groups: bool = True
        """Drop zero-reward-variance groups;"""

    def __init__(self, config: Config) -> None:
        self.config = config

    def build_episodes(self, *, rollout_group: RolloutGroup) -> EpisodeBuilderOutput:
        """Apply group-level filters and convert surviving rollouts to episodes.

        Example:
            # rollouts=[] from a failed generation
            build_episodes(rollout_group=failed_group)
            # -> EpisodeBuilderOutput(episodes=[], metrics=[failure_metric])
        """
        metrics: list[m.Metric] = list(rollout_group.metrics)

        # Failed generation: empty group carrying its failure metric.
        if not rollout_group.rollouts:
            return EpisodeBuilderOutput(episodes=[], metrics=metrics)

        # Untrainable: any sibling with no completion tokens -> drop the whole group
        if any(
            not any(turn.completion_token_ids for turn in rollout.turns)
            for rollout in rollout_group.rollouts
        ):
            metrics.append(
                m.Metric("episode_builder/num_groups_dropped_untrainable", m.Sum(1.0))
            )
            return EpisodeBuilderOutput(episodes=[], metrics=metrics)

        # Zero-std reward: no learning signal across siblings.
        rewards = [rollout.reward for rollout in rollout_group.rollouts]
        is_zero_std = len(rewards) > 1 and statistics.pstdev(rewards) == 0.0
        metrics.append(
            m.Metric(
                "rollout_reward/group_zero_std_frac",
                m.Mean(1.0 if is_zero_std else 0.0),
            )
        )
        if self.config.drop_zero_std_reward_groups and is_zero_std:
            metrics.append(
                m.Metric("episode_builder/num_groups_dropped_zero_std", m.Sum(1.0))
            )
            return EpisodeBuilderOutput(episodes=[], metrics=metrics)

        # One rollout may branch into multiple trainable episodes.
        episodes: list[Episode] = []
        branches_per_rollout: list[float] = []
        for rollout in rollout_group.rollouts:
            rollout_episodes = rollout_to_episodes(rollout)
            episodes.extend(rollout_episodes)
            branches_per_rollout.append(float(len(rollout_episodes)))

        policy_versions = [episode.policy_version for episode in episodes]
        advantages = [rollout.advantage for rollout in rollout_group.rollouts]
        metrics += [
            m.Metric("episode_builder/num_episodes", m.Sum(float(len(episodes)))),
            m.Metric("advantage", m.SummaryStats.from_list(advantages)),
            m.Metric(
                "rollout/branches_per_rollout", m.Mean.from_list(branches_per_rollout)
            ),
            m.Metric(
                "rollout/branches_per_rollout", m.Max.from_list(branches_per_rollout)
            ),
            m.Metric("rollout/policy_version", m.Min.from_list(policy_versions)),
            m.Metric("rollout/policy_version", m.Max.from_list(policy_versions)),
        ]
        return EpisodeBuilderOutput(episodes=episodes, metrics=metrics)
