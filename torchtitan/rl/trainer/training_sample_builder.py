# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Converts one generated rollout group into trainable training_samples.

RolloutGroup -> group filters -> rollout_to_training_samples -> sample filters -> TrainingSampleGroup
"""

import statistics
from dataclasses import dataclass, field

import numpy as np
from renderers import PlaceholderRange

from torchtitan.config import Configurable
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout import Rollout, RolloutGroup
from torchtitan.rl.rollout.types import PromptImage, RolloutTurnID


@dataclass(kw_only=True, slots=True)
class TrainingSample:
    """A trainable token sequence from a rollout.

    Per-token fields are numpy arrays, like `RolloutTurn`'s:
    - not lists: a full GC walks every entry of a list, but doesn't track numpy arrays;
    - not tensors: the turns are already numpy, so a sample is built with no conversion; the batcher
      converts once, into the trainer's tensors.

    Example:
        # Turn 0: prompt P0 -> assistant A0 -> env reply E0
        # Turn 1: prompt [P0, A0, E0] -> assistant A1
        TrainingSample(
            rollout_id=RolloutTurnID(group_id=3, rollout_id=1, turn_id=1),
            min_policy_version=7,
            max_policy_version=9,            # weights updated during rollout
            token_ids=np.array(P0 + A0 + E0 + A1),
            loss_mask=np.array([False]*len(P0) + [True]*len(A0) + [False]*len(E0) + [True]*len(A1)),
            logprobs=np.array([0.0]*len(P0) + logprobs_A0 + [0.0]*len(E0) + logprobs_A1, dtype=np.float32),
            advantage=np.array([0.0]*len(P0) + [adv]*len(A0) + [0.0]*len(E0) + [adv]*len(A1), dtype=np.float32),
        )
    """

    min_policy_version: int
    """Oldest policy version among this branch's trained turns."""
    max_policy_version: int
    """Newest policy version among this branch's trained turns."""
    rollout_id: RolloutTurnID
    """This sample identifier."""
    token_ids: np.ndarray  # int64
    """[L] packed prompt + completions + env replies."""
    loss_mask: np.ndarray  # bool
    """[L] True on assistant tokens to train."""
    logprobs: np.ndarray  # float32
    """[L] generator logprobs; 0.0 where loss_mask is False."""
    advantage: np.ndarray  # float32
    """[L] advantage on assistant tokens, 0.0 elsewhere."""
    images: tuple[PromptImage, ...] = ()
    """Every image in `token_ids`: the last turn's prompt images (images only appear in
    prompts, and that prompt is a prefix of `token_ids`)."""
    image_placeholders: tuple[PlaceholderRange, ...] = ()
    """Where each of `images` sits in `token_ids` (see `Prompt.image_placeholders`)."""


@dataclass(frozen=True, slots=True)
class TrainingSampleGroup:
    """The training samples + metrics built from one rollout group.

    Example:
        TrainingSampleGroup(group_id=3, training_samples=[], metrics=[failure_metric])
        # -> failed / filtered / zero-std group; metrics still reach the trainer logger
    """

    group_id: int
    training_samples: list[TrainingSample]
    metrics: list[m.Metric]


def _has_valid_loss_token(training_sample: TrainingSample) -> bool:
    """Return whether the shifted sample has a token usable by the RL loss."""
    loss_mask, logprobs = training_sample.loss_mask[1:], training_sample.logprobs[1:]
    return bool((loss_mask & np.isfinite(logprobs)).any())


@dataclass(kw_only=True, slots=True)
class _SampleInProgress:
    """A training_sample being built: per field, a list of chunks that `to_training_sample` concatenates
    once, e.g. token_ids = [prompt0, completion0, prompt_delta1, completion1]."""

    rollout_id: RolloutTurnID
    min_policy_version: int
    max_policy_version: int
    token_ids: list[np.ndarray] = field(default_factory=list)
    loss_mask: list[np.ndarray] = field(default_factory=list)
    logprobs: list[np.ndarray] = field(default_factory=list)
    advantage: list[np.ndarray] = field(default_factory=list)
    images: tuple[PromptImage, ...] = ()
    image_placeholders: tuple[PlaceholderRange, ...] = ()

    def to_training_sample(self) -> TrainingSample:
        return TrainingSample(
            rollout_id=self.rollout_id,
            min_policy_version=self.min_policy_version,
            max_policy_version=self.max_policy_version,
            token_ids=np.concatenate(self.token_ids, dtype=np.int64),
            loss_mask=np.concatenate(self.loss_mask),
            logprobs=np.concatenate(self.logprobs, dtype=np.float32),
            advantage=np.concatenate(self.advantage),
            images=self.images,
            image_placeholders=self.image_placeholders,
        )


class TrainingSampleBuilder(Configurable):
    """Build trainable samples and metrics the trainer consumes from rollout groups.

    Example:
        builder = config.async_loop.batcher.training_sample_builder.build()
        output: TrainingSampleGroup = builder.build_from_group(rollout_group=rollout_group)
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """TrainingSample builder filtering knobs."""

        drop_zero_std_reward_groups: bool = True
        """Drop zero-reward-variance groups;"""

    def __init__(self, config: Config) -> None:
        self.config = config

    def build_from_group(self, *, rollout_group: RolloutGroup) -> TrainingSampleGroup:
        """Apply data-validity filters and convert surviving rollouts to training_samples.

        Example:
            # rollouts=[] from a failed generation
            build_from_group(rollout_group=failed_group)
            # -> TrainingSampleGroup(group_id=..., training_samples=[], metrics=[failure_metric])
        """
        metrics: list[m.Metric] = list(rollout_group.metrics)
        # A rollout with reward None wasn't scored (see `BaseRolloutWorker.make_scored_group`): skip
        # it, not its group. A group left with 10 of 16 trains those 10, centered on their 10 rewards.
        rollouts = [
            rollout for rollout in rollout_group.rollouts if rollout.reward is not None
        ]

        # TODO(async-rl): generalize these group-level filters into an ordered, user-pluggable filter
        # list (e.g. drop groups with all-0 / all-1 advantage), run in sequence on each RolloutGroup.
        # Failed generation, or no rollout scored: no samples, only the metrics.
        if not rollouts:
            return TrainingSampleGroup(
                group_id=rollout_group.group_id, training_samples=[], metrics=metrics
            )

        # Untrainable: any sibling with no completion tokens -> drop the whole group
        if any(
            not any(len(turn.completion_token_ids) > 0 for turn in rollout.turns)
            for rollout in rollouts
        ):
            metrics.append(
                m.Metric(
                    "training_sample_builder/num_groups_dropped_untrainable", m.Sum(1.0)
                )
            )
            return TrainingSampleGroup(
                group_id=rollout_group.group_id, training_samples=[], metrics=metrics
            )

        # Zero-std reward: no learning signal across siblings. A group of N > 1 left with one scored
        # rollout counts too: its advantage is 0.
        rewards = [rollout.reward for rollout in rollouts]
        is_zero_std = (
            len(rollout_group.rollouts) > 1 and statistics.pstdev(rewards) == 0.0
        )
        metrics.append(
            m.Metric(
                "rollout_reward/group_zero_std_frac",
                m.Mean(1.0 if is_zero_std else 0.0),
            )
        )
        if self.config.drop_zero_std_reward_groups and is_zero_std:
            metrics.append(
                m.Metric(
                    "training_sample_builder/num_groups_dropped_zero_std", m.Sum(1.0)
                )
            )
            return TrainingSampleGroup(
                group_id=rollout_group.group_id, training_samples=[], metrics=metrics
            )

        # One rollout may branch into multiple trainable training_samples.
        training_samples: list[TrainingSample] = []
        unfiltered_training_samples: list[TrainingSample] = []
        branches_per_rollout: list[float] = []
        num_samples_dropped_no_valid_tokens = 0
        num_rollouts_dropped_empty_completion = 0
        for rollout in rollouts:
            # A later turn with no completion (e.g. a generate fn that returned no tokens): drop the
            # rollout, so `rollout_to_training_samples` doesn't raise and stop the run.
            if any(
                len(rollout_turn.completion_token_ids) == 0
                for rollout_turn in rollout.turns[1:]
            ):
                num_rollouts_dropped_empty_completion += 1
                continue

            rollout_training_samples = self.rollout_to_training_samples(rollout)
            unfiltered_training_samples.extend(rollout_training_samples)
            kept = [
                training_sample
                for training_sample in rollout_training_samples
                if _has_valid_loss_token(training_sample)
            ]
            num_samples_dropped_no_valid_tokens += len(rollout_training_samples) - len(
                kept
            )
            training_samples.extend(kept)
            branches_per_rollout.append(float(len(rollout_training_samples)))

        if num_rollouts_dropped_empty_completion:
            metrics.append(
                m.Metric(
                    "training_sample_builder/num_rollouts_dropped_empty_completion",
                    m.Sum(float(num_rollouts_dropped_empty_completion)),
                )
            )
        if num_samples_dropped_no_valid_tokens:
            metrics.append(
                m.Metric(
                    "training_sample_builder/num_samples_dropped_no_valid_tokens",
                    m.Sum(float(num_samples_dropped_no_valid_tokens)),
                )
            )

        min_policy_versions = [
            training_sample.min_policy_version
            for training_sample in unfiltered_training_samples
        ]
        max_policy_versions = [
            training_sample.max_policy_version
            for training_sample in unfiltered_training_samples
        ]
        advantages = [rollout.advantage for rollout in rollouts]
        metrics += [
            m.Metric(
                "training_sample_builder/num_training_samples",
                m.Sum(float(len(training_samples))),
            ),
            m.Metric("advantage", m.SummaryStats.from_list(advantages)),
            m.Metric(
                "rollout/branches_per_rollout", m.Mean.from_list(branches_per_rollout)
            ),
            m.Metric(
                "rollout/branches_per_rollout", m.Max.from_list(branches_per_rollout)
            ),
            m.Metric(
                "rollout/min_policy_version", m.Min.from_list(min_policy_versions)
            ),
            m.Metric(
                "rollout/max_policy_version", m.Max.from_list(max_policy_versions)
            ),
        ]
        return TrainingSampleGroup(
            group_id=rollout_group.group_id,
            training_samples=training_samples,
            metrics=metrics,
        )

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
        = the oldest version any of its turns depends on (the policy-age metrics and the max_offpolicy_steps
        check read it), `max_policy_version` = the newest version any of its turns reached.

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
        # The training_sample that later turns may still extend; None until the first trained turn
        sample_in_progress: _SampleInProgress | None = None
        # Length of the previous turn's prompt + completion
        prev_len = 0
        # The previous turn's prompt images and where they sit (images appear only in prompts)
        prev_images: tuple[PromptImage, ...] = ()
        prev_image_placeholders: tuple[PlaceholderRange, ...] = ()

        # Skip if no completion (nothing to train on). This happens when the prompt is too
        # long in the first turn, before any generation. We keep these rollouts for debugging.
        # TODO(async-rl): confirm this signal actually reaches the rollout_recorder.
        for turn_idx, rollout_turn in enumerate(rollout.turns):
            if len(rollout_turn.completion_token_ids) == 0:
                if len(rollout.turns) == 1:

                    continue

                # empty rollouts should only happen when initial prompt is too long
                raise ValueError(
                    f"rollout {rollout.group_id}/rollout={rollout.rollout_id}: "
                    f"non-final turn {turn_idx} has no completion"
                )

            prompt_prefix_len = rollout_turn.prompt_prefix_len
            prompt_delta_token_ids = rollout_turn.prompt_delta_token_ids

            # Too many: e.g. prev = [0, 1, 2, 3] has 4 tokens; a turn can't reuse 6.
            if prompt_prefix_len > prev_len:
                raise ValueError(
                    f"rollout {rollout.group_id}/rollout={rollout.rollout_id}: turn {turn_idx} "
                    f"reuses {prompt_prefix_len} tokens, but the previous turn has only "
                    f"{prev_len}; build it with `split_prompt`"
                )
            # Delta images sit in the delta, at full-prompt positions; an offset inside the
            # delta (e.g. 1 instead of prefix + 1) would put image embeddings on other tokens.
            for placeholder in rollout_turn.prompt_delta_image_placeholders:
                if not (
                    prompt_prefix_len
                    <= placeholder.offset
                    <= rollout_turn.num_prompt_tokens - placeholder.length
                ):
                    raise ValueError(
                        f"rollout {rollout.group_id}/rollout={rollout.rollout_id}: turn "
                        f"{turn_idx} has an image at tokens [{placeholder.offset}, "
                        f"{placeholder.offset + placeholder.length}), outside its prompt delta "
                        f"[{prompt_prefix_len}, {rollout_turn.num_prompt_tokens}); image "
                        "placeholders are positions in the full prompt"
                    )

            # The full prompt's images: the previous prompt's images in the shared prefix, then
            # the turn's own. `split_prompt` ends the prefix at the first changed image, so an
            # image swapped in place (equal placeholder tokens) shortens the prefix too.
            num_prefix_images = sum(
                placeholder.offset < prompt_prefix_len
                for placeholder in prev_image_placeholders
            )
            images = prev_images[:num_prefix_images] + rollout_turn.prompt_delta_images
            image_placeholders = (
                prev_image_placeholders[:num_prefix_images]
                + rollout_turn.prompt_delta_image_placeholders
            )

            # True when this prompt continues the previous one (prefix-preserving);
            # False when the env edited history -> open a new training_sample (branch).
            extends_prev = prompt_prefix_len == prev_len
            if sample_in_progress is not None and not extends_prev:
                training_samples.append(sample_in_progress.to_training_sample())
                sample_in_progress = None

            if sample_in_progress is None:
                # Each turn appended to the last training_sample, so its token_ids are exactly the
                # previous turn's prompt + completion.
                prev_prompt_and_completion = (
                    training_samples[-1].token_ids
                    if training_samples
                    else np.empty(0, dtype=np.int64)
                )

                # Too few: the first new token equals prev's token at the same position, so the turn
                # could reuse at least one more. E.g. prev = [0, 1, 2, 3], stored (0, [0, 1, 9, 3]):
                # 0 == prev[0]; split_prompt gives (2, [9, 3]). Catches a full prompt stored as the delta.
                # Not when a changed image starts the delta: `split_prompt` ends the prefix at
                # it, and its placeholder tokens equal the previous image's.
                could_reuse_one_more = (
                    prompt_prefix_len < prev_len
                    and len(prompt_delta_token_ids) > 0
                    and prompt_delta_token_ids[0]
                    == prev_prompt_and_completion[prompt_prefix_len]
                    and not any(
                        placeholder.offset == prompt_prefix_len
                        for placeholder in rollout_turn.prompt_delta_image_placeholders
                    )
                )
                if could_reuse_one_more:
                    raise ValueError(
                        f"rollout {rollout.group_id}/rollout={rollout.rollout_id}: turn {turn_idx} "
                        f"reuses {prompt_prefix_len} tokens but could reuse more; build it with "
                        "`split_prompt`"
                    )

                # Start a new training_sample; its RolloutTurnID marks the turn the segment begins at.
                sample_in_progress = _SampleInProgress(
                    min_policy_version=rollout_turn.min_policy_version,
                    max_policy_version=rollout_turn.max_policy_version,
                    rollout_id=RolloutTurnID(
                        group_id=rollout.group_id,
                        rollout_id=rollout.rollout_id,
                        turn_id=turn_idx,
                    ),
                )
                # A new training_sample needs the full prompt: the shared prefix + the delta.
                prompt_delta = np.concatenate(
                    [
                        prev_prompt_and_completion[:prompt_prefix_len],
                        prompt_delta_token_ids,
                    ]
                )
            else:
                prompt_delta = prompt_delta_token_ids

            # Append this turn's chunks: prompt delta (untrained) + completion (trained).
            num_delta = len(prompt_delta)
            num_completion = len(rollout_turn.completion_token_ids)
            sample_in_progress.token_ids += [
                prompt_delta,
                rollout_turn.completion_token_ids,
            ]
            sample_in_progress.loss_mask += [
                np.zeros(num_delta, dtype=bool),
                np.ones(num_completion, dtype=bool),
            ]
            sample_in_progress.logprobs += [
                np.zeros(num_delta, dtype=np.float32),
                rollout_turn.completion_logprobs,
            ]
            sample_in_progress.advantage += [
                np.zeros(num_delta, dtype=np.float32),
                np.full(num_completion, rollout_advantage, dtype=np.float32),
            ]
            sample_in_progress.images = images
            sample_in_progress.image_placeholders = image_placeholders
            sample_in_progress.min_policy_version = min(
                sample_in_progress.min_policy_version, rollout_turn.min_policy_version
            )
            sample_in_progress.max_policy_version = max(
                sample_in_progress.max_policy_version, rollout_turn.max_policy_version
            )

            prev_len = rollout_turn.num_prompt_tokens + num_completion
            prev_images, prev_image_placeholders = images, image_placeholders

        if sample_in_progress is not None:
            training_samples.append(sample_in_progress.to_training_sample())
        return training_samples
