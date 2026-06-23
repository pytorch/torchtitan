# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Converts one generated rollout group into trainable training_samples.

RolloutGroup -> data-validity filters -> rollout_to_training_samples -> TrainingSampleBuilderOutput

Policy-version staleness is handled later by `Batcher`.
"""

import statistics
from dataclasses import dataclass

from torchtitan.config import Configurable
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.rollout import Rollout, RolloutGroup
from torchtitan.experiments.rl.types import TrainingSample, RolloutID


@dataclass(frozen=True, slots=True)
class TrainingSampleBuilderOutput:
    """TrainingSamples and metrics emitted for one rollout group.

    Example:
        TrainingSampleBuilderOutput(training_samples=[], metrics=[failure_metric])
        # -> failed or filtered group; metrics still reach the trainer logger
    """

    training_samples: list[TrainingSample]
    metrics: list[m.Metric]


class TrainingSampleBuilder(Configurable):
    """Build trainable training_samples and rollout-origin metrics from one group.

    Example:
        builder = config.training_sample_builder.build()
        output = builder.build_training_samples(rollout_group=group)
        # output.training_samples is empty for failed, untrainable, or zero-std groups.
        # output.metrics still carries the counters for that group.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """TrainingSample builder filtering knobs."""

        drop_zero_std_reward_groups: bool = True
        """Drop zero-reward-variance groups;"""

    def __init__(self, config: Config) -> None:
        self.config = config

    def build_training_samples(self, *, rollout_group: RolloutGroup) -> TrainingSampleBuilderOutput:
        """Apply group-level filters and convert surviving rollouts to training_samples.

        Example:
            # rollouts=[] from a failed generation
            build_training_samples(rollout_group=failed_group)
            # -> TrainingSampleBuilderOutput(training_samples=[], metrics=[failure_metric])
        """
        metrics: list[m.Metric] = list(rollout_group.metrics)

        # Failed generation: empty group carrying its failure metric.
        if not rollout_group.rollouts:
            return TrainingSampleBuilderOutput(training_samples=[], metrics=metrics)

        # Untrainable: any sibling with no completion tokens -> drop the whole group
        if any(
            not any(turn.completion_token_ids for turn in rollout.turns)
            for rollout in rollout_group.rollouts
        ):
            metrics.append(
                m.Metric("training_sample_builder/num_groups_dropped_untrainable", m.Sum(1.0))
            )
            return TrainingSampleBuilderOutput(training_samples=[], metrics=metrics)

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
                m.Metric("training_sample_builder/num_groups_dropped_zero_std", m.Sum(1.0))
            )
            return TrainingSampleBuilderOutput(training_samples=[], metrics=metrics)

        # One rollout may branch into multiple trainable training_samples.
        training_samples: list[TrainingSample] = []
        branches_per_rollout: list[float] = []
        for rollout in rollout_group.rollouts:
            rollout_training_samples = self.rollout_to_training_samples(rollout)
            training_samples.extend(rollout_training_samples)
            branches_per_rollout.append(float(len(rollout_training_samples)))

        min_policy_versions = [
            training_sample.min_policy_version for training_sample in training_samples
        ]
        advantages = [rollout.advantage for rollout in rollout_group.rollouts]
        metrics += [
            m.Metric("training_sample_builder/num_training_samples", m.Sum(float(len(training_samples)))),
            m.Metric("advantage", m.SummaryStats.from_list(advantages)),
            m.Metric(
                "rollout/branches_per_rollout", m.Mean.from_list(branches_per_rollout)
            ),
            m.Metric(
                "rollout/branches_per_rollout", m.Max.from_list(branches_per_rollout)
            ),
            m.Metric("rollout/min_policy_version", m.Min.from_list(min_policy_versions)),
            m.Metric("rollout/min_policy_version", m.Max.from_list(min_policy_versions)),
        ]
        return TrainingSampleBuilderOutput(training_samples=training_samples, metrics=metrics)

    # TODO(async-rl): evaluate renaming TrainingSample -> RolloutSegment once we align the overloaded
    # "sample" names (the dataset/input side also uses "sample").
    def rollout_to_training_samples(self, rollout: Rollout) -> list[TrainingSample]:
        """Pack a scored `Rollout` into training training_samples — usually one, or several where its turns branch.

        Override this to change how a rollout becomes trainable training_samples (the main extension point of
        this class). Turns that share a growing prefix (each prompt continues the previous prompt +
        completion) are packed into ONE training_sample: prompts and env replies are masked out, completions
        are trained. A new training_sample (branch) opens wherever that prefix breaks (history was edited).

        This branching has three common causes:
        1) The rollout loop purposely edits the history: For example, when compacting a long conversation;
        2) Thinking is removed from history: This is a flag in the `renderer` used in the `Rollouter`. Choose
            `preserve_all_thinking=True` to avoid stripping thinking (default=True);
        3) Some error or change from retokenization: Those can be avoided by doing token-in-token-out (TITO),
            i.e. you give tokens to the generators, and you append (bridge) the tokens received, leaving no room for changes.
            TITO is our default;

        Each packed training_sample records the min/max policy version across its turns: `min_policy_version`
        = the oldest (opening) turn's version (the off-policy filter reads it), `max_policy_version` = the
        newest version any of its turns reached.

        Example (5 turns; the env compacts history before turn 3, so the prefix breaks -> 2 training_samples).
        P = prompt; C = completion; E = env reply
            turn0 (v5): prompt=[P1]              completion=[C1]
            turn1 (v6): prompt=[P1,C1,E1]        completion=[C2]
            turn2 (v6): prompt=[P1,C1,E1,C2,E2]  completion=[C3]
            turn3 (v6): prompt=[P2]              completion=[C4]   # history compacted -> prefix breaks
            turn4 (v7): prompt=[P2,C4,E4]        completion=[C5]
            # -> training_sample 0: token_ids=[P1,C1,E1,C2,E2,C3], loss_mask=[0,1,0,1,0,1]
            #              rollout_id.turn_id=0, min_policy_version=5, max_policy_version=6
            #    training_sample 1: token_ids=[P2,C4,E4,C5],        loss_mask=[0,1,0,1]
            #              rollout_id.turn_id=3, min_policy_version=6, max_policy_version=7
        """
        rollout_advantage = rollout.advantage
        if rollout_advantage is None:
            raise ValueError(
                f"rollout {rollout.group_id}/rollout={rollout.rollout_id} has no advantage; the Rollouter "
                "must fill it (via its advantage estimator) before training_samples are built."
            )
        training_samples: list[TrainingSample] = []

        # Used to check if [P1, C1] is prefix of [P1,C1,E1] in the docstring example
        prev_prompt_and_completion: list[int] = []

        # Skip if no completion (nothing to train on). This happens when the prompt is too long.
        # TODO: This seems to happen on the very first turn. Check if we can prefilter it.
        for turn_idx, rollout_turn in enumerate(rollout.turns):
            if not rollout_turn.completion_token_ids:
                if turn_idx != len(rollout.turns) - 1:
                    raise ValueError(
                        f"rollout {rollout.group_id}/rollout={rollout.rollout_id}: "
                        f"non-final turn {turn_idx} has no completion"
                    )
                continue

            prompt = rollout_turn.prompt_token_ids
            # True when this prompt continues the previous one (prefix-preserving);
            # False when the env edited history -> open a new training_sample (branch).
            extends_prev = (
                prompt[: len(prev_prompt_and_completion)] == prev_prompt_and_completion
            )
            if not training_samples or not extends_prev:
                # Start a new training_sample; its rollout_id.turn_id marks the turn the segment begins at.
                training_samples.append(
                    TrainingSample(
                        min_policy_version=rollout_turn.min_policy_version,
                        max_policy_version=rollout_turn.max_policy_version,
                        rollout_id=RolloutID(
                            group_id=rollout.group_id,
                            rollout_id=rollout.rollout_id,
                            turn_id=turn_idx,
                        ),
                        token_ids=[],
                        loss_mask=[],
                        logprobs=[],
                        advantage=[],
                    )
                )
                # The whole prompt opens this training_sample.
                prefix_len = 0
            else:
                # Only the env-reply suffix is new.
                prefix_len = len(prev_prompt_and_completion)

            # Untrained prefix delta (the prompt or the new env reply), then the trained completion.
            training_sample = training_samples[-1]
            prompt_delta = prompt[prefix_len:]
            num_delta = len(prompt_delta)
            num_completion = len(rollout_turn.completion_token_ids)
            training_sample.token_ids += prompt_delta
            training_sample.loss_mask += [False] * num_delta
            training_sample.logprobs += [0.0] * num_delta
            training_sample.advantage += [0.0] * num_delta
            # Widen the segment's version span to include this turn (the min stays the opening turn's).
            training_sample.max_policy_version = max(
                training_sample.max_policy_version, rollout_turn.max_policy_version
            )
            training_sample.token_ids += rollout_turn.completion_token_ids
            training_sample.loss_mask += [True] * num_completion
            training_sample.logprobs += rollout_turn.completion_logprobs
            training_sample.advantage += [rollout_advantage] * num_completion

            prev_prompt_and_completion = prompt + rollout_turn.completion_token_ids

        return training_samples
