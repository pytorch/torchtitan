# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Converts one generated rollout group into trainable training_samples, in two halves.

    decide:       group + its turn lengths -> group filters -> rollout_to_training_sample_specs
                  -> sample filters -> TrainingSampleSpecGroup                       (lengths only)
    materialize:  specs + their rollout group -> one TrainingSample per spec, in
                  spec order                                                         (the arrays)

`rollout_to_training_sample_specs` decides from lengths which turns form each sample and checks
the turns. `materialize` builds each planned sample from its turns' arrays, laying the tokens out
again (`_build_training_sample`); `MicrobatchPacker.pack_planned` checks that the two agree.
"""

import statistics
from dataclasses import dataclass, field

import numpy as np
from renderers import PlaceholderRange

from torchtitan.config import Configurable
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout import Rollout, RolloutGroup
from torchtitan.rl.rollout.types import PromptImage, RolloutTurnID, RolloutTurnLengths


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
class TrainingSampleSpec:
    """What the batcher plans a `TrainingSample` with, computed from lengths without its arrays.

    `TrainingSampleBuilder.materialize` builds the sample a spec describes from the spec's turns,
    so the plan (bins, global token counts) made from specs holds for the samples; the batcher
    checks each packed microbatch's token counts against its specs.

    Example:
        # The single-turn rollout prompt [1, 2] -> completion [3, 4, 5], logprob of 4 NaN:
        TrainingSampleSpec(
            rollout_id=RolloutTurnID(group_id=3, rollout_id=1, turn_id=0),
            last_turn_id=0,
            num_tokens=5,                 # len(token_ids)
            loss_token_counts=(2,),       # shifted completion positions with finite logprobs
            routing_token_counts=(4,),    # shifted positions
            num_response_tokens=3,        # shifted completion positions
            ...
        )
    """

    rollout_id: RolloutTurnID
    """The sample's `TrainingSample.rollout_id`: its rollout and the turn it begins at."""
    last_turn_id: int
    """The last turn in the sample (inclusive)."""
    num_tokens: int
    """`len(TrainingSample.token_ids)`."""
    loss_token_counts: tuple[int, ...]
    """[1 + num_mtp_layers] loss tokens at each MTP depth, as `get_mtp_token_counts` counts them
    for this sample in a packed microbatch."""
    routing_token_counts: tuple[int, ...]
    """[1 + num_mtp_layers] non-padding tokens at each MTP depth, as `get_mtp_token_counts`
    counts them."""
    num_response_tokens: int
    """Shifted completion positions (`TrainingSample.loss_mask[1:].sum()`), finite or not."""
    min_policy_version: int
    max_policy_version: int
    images: tuple[PromptImage, ...] = ()
    """`TrainingSample.images`."""
    image_placeholders: tuple[PlaceholderRange, ...] = ()
    """`TrainingSample.image_placeholders`."""


@dataclass(frozen=True, slots=True)
class TrainingSampleSpecGroup:
    """The training sample specs + metrics decided for one rollout group.

    Example:
        TrainingSampleSpecGroup(group_id=3, training_sample_specs=[], metrics=[failure_metric])
        # -> failed / filtered / zero-std group; metrics still reach the trainer logger
    """

    group_id: int
    training_sample_specs: list[TrainingSampleSpec]
    metrics: list[m.Metric]


@dataclass(kw_only=True, slots=True)
class _SpecInProgress:
    """A training sample spec being built from turn lengths."""

    rollout_id: RolloutTurnID
    min_policy_version: int
    max_policy_version: int
    last_turn_id: int
    num_tokens: int = 0
    # Per turn: where its completion starts in the sample's tokens, and the turn's lengths.
    completions: list[tuple[int, RolloutTurnLengths]] = field(default_factory=list)
    images: tuple[PromptImage, ...] = ()
    image_placeholders: tuple[PlaceholderRange, ...] = ()

    def to_training_sample_spec(self, *, num_mtp_layers: int) -> TrainingSampleSpec:
        # Labels are token_ids[1:], so token index i is shifted position i - 1. At MTP depth d,
        # `get_mtp_token_counts` counts a loss token at each shifted position j >= d whose label is
        # a completion token with a finite logprob (token index >= d + 1), and a routed token at
        # each shifted position j < n - d, for n = num_tokens - 1 shifted positions.
        loss_token_counts = []
        for depth in range(num_mtp_layers + 1):
            num_loss_tokens = 0
            for start, lengths in self.completions:
                first = max(start, depth + 1)
                end = start + lengths.num_completion_tokens
                num_loss_tokens += max(end - first, 0) - sum(
                    start + offset >= first
                    for offset in lengths.completion_nonfinite_logprob_offsets
                )
            loss_token_counts.append(num_loss_tokens)
        num_shifted_tokens = self.num_tokens - 1
        return TrainingSampleSpec(
            rollout_id=self.rollout_id,
            last_turn_id=self.last_turn_id,
            num_tokens=self.num_tokens,
            loss_token_counts=tuple(loss_token_counts),
            routing_token_counts=tuple(
                max(num_shifted_tokens - depth, 0)
                for depth in range(num_mtp_layers + 1)
            ),
            # A completion token at index 0 (an empty first prompt) is no label.
            num_response_tokens=sum(
                lengths.num_completion_tokens - (start == 0)
                for start, lengths in self.completions
            ),
            min_policy_version=self.min_policy_version,
            max_policy_version=self.max_policy_version,
            images=self.images,
            image_placeholders=self.image_placeholders,
        )


class TrainingSampleBuilder(Configurable):
    """Build trainable samples and metrics the trainer consumes from rollout groups, in two halves:
    `decide` filters a group and plans its samples from lengths (`TrainingSampleSpec`s), and
    `materialize` builds the samples of chosen specs from the groups' arrays.

    Example:
        builder = config.async_loop.batcher.training_sample_builder.build()
        spec_group = builder.decide(
            rollout_group=rollout_group,
            turn_lengths=tuple(map(rollout_turn_lengths, rollout_group.rollouts)),
            num_mtp_layers=0,
        )
        samples = builder.materialize(
            training_sample_specs=spec_group.training_sample_specs,
            rollout_group=rollout_group,
        )

    To change which turns form a sample, override `rollout_to_training_sample_specs`.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """TrainingSample builder filtering knobs."""

        drop_zero_std_reward_groups: bool = True
        """Drop zero-reward-variance groups;"""

    def __init__(self, config: Config) -> None:
        self.config = config

    def decide(
        self,
        *,
        rollout_group: RolloutGroup,
        turn_lengths: tuple[tuple[RolloutTurnLengths, ...], ...],
        num_mtp_layers: int,
    ) -> TrainingSampleSpecGroup:
        """Apply data-validity filters and plan the surviving rollouts' training samples, from
        the group's lengths and rewards.

        It reads no array of `rollout_group`, so the step can be planned without them: a filter
        added here reads lengths from `turn_lengths`, not from a turn's arrays.

        Args:
            rollout_group: The scored group; its ids, rewards, advantages, policy versions,
                prompt prefix lengths and images are read, not its arrays.
            turn_lengths: [rollout][turn] -> the turn's lengths (`rollout_turn_lengths` of each
                rollout).
            num_mtp_layers: MTP depth of the token counts.

        Example:
            # rollouts=[] from a failed generation
            decide(rollout_group=failed_group, turn_lengths=(), num_mtp_layers=0)
            # -> TrainingSampleSpecGroup(group_id=..., training_sample_specs=[], metrics=[failure_metric])
        """
        metrics: list[m.Metric] = list(rollout_group.metrics)

        # TODO(async-rl): generalize these group-level filters into an ordered, user-pluggable filter
        # list (e.g. drop groups with all-0 / all-1 advantage), run in sequence on each RolloutGroup.
        # Failed generation: empty group carrying its failure metric.
        if not rollout_group.rollouts:
            return TrainingSampleSpecGroup(
                group_id=rollout_group.group_id,
                training_sample_specs=[],
                metrics=metrics,
            )

        # Untrainable: any sibling with no completion tokens -> drop the whole group
        if any(
            not any(lengths.num_completion_tokens > 0 for lengths in rollout_lengths)
            for rollout_lengths in turn_lengths
        ):
            metrics.append(
                m.Metric(
                    "training_sample_builder/num_groups_dropped_untrainable", m.Sum(1.0)
                )
            )
            return TrainingSampleSpecGroup(
                group_id=rollout_group.group_id,
                training_sample_specs=[],
                metrics=metrics,
            )

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
                m.Metric(
                    "training_sample_builder/num_groups_dropped_zero_std", m.Sum(1.0)
                )
            )
            return TrainingSampleSpecGroup(
                group_id=rollout_group.group_id,
                training_sample_specs=[],
                metrics=metrics,
            )

        # One rollout may branch into multiple trainable training_samples.
        training_sample_specs: list[TrainingSampleSpec] = []
        unfiltered_training_sample_specs: list[TrainingSampleSpec] = []
        branches_per_rollout: list[float] = []
        num_samples_dropped_no_valid_tokens = 0
        num_rollouts_dropped_empty_completion = 0
        for rollout, rollout_lengths in zip(
            rollout_group.rollouts, turn_lengths, strict=True
        ):
            # A later turn with no completion (e.g. a request aborted before its first token): drop the
            # rollout, so `rollout_to_training_sample_specs` doesn't raise and stop the run.
            if any(
                lengths.num_completion_tokens == 0 for lengths in rollout_lengths[1:]
            ):
                num_rollouts_dropped_empty_completion += 1
                continue

            rollout_training_sample_specs = self.rollout_to_training_sample_specs(
                rollout, rollout_lengths, num_mtp_layers=num_mtp_layers
            )
            unfiltered_training_sample_specs.extend(rollout_training_sample_specs)
            # Keep samples with a token usable by the RL loss.
            kept = [
                spec
                for spec in rollout_training_sample_specs
                if spec.loss_token_counts[0] > 0
            ]
            num_samples_dropped_no_valid_tokens += len(
                rollout_training_sample_specs
            ) - len(kept)
            training_sample_specs.extend(kept)
            branches_per_rollout.append(float(len(rollout_training_sample_specs)))

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
            spec.min_policy_version for spec in unfiltered_training_sample_specs
        ]
        max_policy_versions = [
            spec.max_policy_version for spec in unfiltered_training_sample_specs
        ]
        advantages = [rollout.advantage for rollout in rollout_group.rollouts]
        metrics += [
            m.Metric(
                "training_sample_builder/num_training_samples",
                m.Sum(float(len(training_sample_specs))),
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
        return TrainingSampleSpecGroup(
            group_id=rollout_group.group_id,
            training_sample_specs=training_sample_specs,
            metrics=metrics,
        )

    def rollout_to_training_sample_specs(
        self,
        rollout: Rollout,
        turn_lengths: tuple[RolloutTurnLengths, ...],
        *,
        num_mtp_layers: int,
    ) -> list[TrainingSampleSpec]:
        """Plan a scored `Rollout`'s training samples from its turns' lengths: usually one, or
        several where its turns branch. It reads no array of `rollout`.

        Override this to change how a rollout becomes trainable samples (the extension point of
        this class). Turns that share a growing prefix (each prompt continues the previous prompt +
        completion) are packed into ONE sample: prompts and env replies are masked out,
        completions are trained. A new sample (branch) opens wherever that prefix breaks
        (history was edited).

        This branching has three common causes:
        1) The rollout loop purposely edits the history: For example, when compacting a long conversation;
        2) Thinking is removed from history: This is a flag in the `renderer` used in the `Rollouter`. Choose
            `preserve_all_thinking=True` to avoid stripping thinking (default=True);
        3) Some error or change from retokenization: Those can be avoided by doing token-in-token-out (TITO),
            i.e. you give tokens to the generators, and you append (bridge) the tokens received, leaving no room for changes.
            TITO is our default;

        Each sample records the min/max policy version across its turns: `min_policy_version` = the
        oldest version any of its turns depends on (the policy-age metrics and the
        max_offpolicy_steps check read it), `max_policy_version` = the newest version any of its
        turns reached.

        Example (5 turns; the env compacts history before turn 3, so the prefix breaks -> 2 samples).
        P = prompt; C = completion; E = env reply
            turn0 (v5): prompt=[P1]              completion=[C1]
            turn1 (v6): prompt=[P1,C1,E1]        completion=[C2]
            turn2 (v6): prompt=[P1,C1,E1,C2,E2]  completion=[C3]
            turn3 (v6): prompt=[P2]              completion=[C4]   # history compacted -> prefix breaks
            turn4 (v7): prompt=[P2,C4,E4]        completion=[C5]
            # -> spec 0: turns 0-2, token_ids=[P1,C1,E1,C2,E2,C3], loss_mask=[0,1,0,1,0,1]
            #            rollout_id.turn_id=0, min_policy_version=5, max_policy_version=6
            #    spec 1: turns 3-4, token_ids=[P2,C4,E4,C5],        loss_mask=[0,1,0,1]
            #            rollout_id.turn_id=3, min_policy_version=6, max_policy_version=7

        The check that reads token values (a prefix that could reuse one more token) runs
        before, on every rollout of the group (`check_prompt_prefixes`).

        Args:
            rollout: The scored rollout; reads its ids, advantage, policy versions, prompt prefix
                lengths and images, not its arrays.
            turn_lengths: Each turn's lengths (`rollout_turn_lengths(rollout)`).
            num_mtp_layers: MTP depth of the token counts.
        """
        if rollout.advantage is None:
            raise ValueError(
                f"rollout {rollout.group_id}/rollout={rollout.rollout_id} has no advantage; the Rollouter "
                "must fill it (via its advantage estimator) before training_samples are built."
            )
        training_sample_specs: list[TrainingSampleSpec] = []
        spec_in_progress: _SpecInProgress | None = None
        prev_len = 0
        prev_images: tuple[PromptImage, ...] = ()
        prev_image_placeholders: tuple[PlaceholderRange, ...] = ()
        # Skip if no completion (nothing to train on). This happens when the prompt is too
        # long in the first turn, before any generation. We keep these rollouts for debugging.
        # TODO(async-rl): confirm this signal actually reaches the rollout_recorder.
        for turn_idx, (rollout_turn, lengths) in enumerate(
            zip(rollout.turns, turn_lengths, strict=True)
        ):
            if lengths.num_completion_tokens == 0:
                if len(rollout.turns) == 1:
                    continue
                # empty rollouts should only happen when initial prompt is too long
                raise ValueError(
                    f"rollout {rollout.group_id}/rollout={rollout.rollout_id}: "
                    f"non-final turn {turn_idx} has no completion"
                )

            prompt_prefix_len = rollout_turn.prompt_prefix_len
            num_prompt_tokens = prompt_prefix_len + lengths.num_prompt_delta_tokens
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
                    <= num_prompt_tokens - placeholder.length
                ):
                    raise ValueError(
                        f"rollout {rollout.group_id}/rollout={rollout.rollout_id}: turn "
                        f"{turn_idx} has an image at tokens [{placeholder.offset}, "
                        f"{placeholder.offset + placeholder.length}), outside its prompt delta "
                        f"[{prompt_prefix_len}, {num_prompt_tokens}); image "
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
            # False when the env edited history -> open a new sample (branch).
            extends_prev = prompt_prefix_len == prev_len
            if spec_in_progress is not None and not extends_prev:
                training_sample_specs.append(
                    spec_in_progress.to_training_sample_spec(
                        num_mtp_layers=num_mtp_layers
                    )
                )
                spec_in_progress = None

            if spec_in_progress is None:
                spec_in_progress = _SpecInProgress(
                    rollout_id=RolloutTurnID(
                        group_id=rollout.group_id,
                        rollout_id=rollout.rollout_id,
                        turn_id=turn_idx,
                    ),
                    min_policy_version=rollout_turn.min_policy_version,
                    max_policy_version=rollout_turn.max_policy_version,
                    last_turn_id=turn_idx,
                )
                # A new sample starts with the full prompt.
                spec_in_progress.num_tokens += num_prompt_tokens
            else:
                spec_in_progress.num_tokens += lengths.num_prompt_delta_tokens

            spec_in_progress.completions.append((spec_in_progress.num_tokens, lengths))
            spec_in_progress.num_tokens += lengths.num_completion_tokens
            spec_in_progress.last_turn_id = turn_idx
            spec_in_progress.images = images
            spec_in_progress.image_placeholders = image_placeholders
            spec_in_progress.min_policy_version = min(
                spec_in_progress.min_policy_version, rollout_turn.min_policy_version
            )
            spec_in_progress.max_policy_version = max(
                spec_in_progress.max_policy_version, rollout_turn.max_policy_version
            )

            prev_len = num_prompt_tokens + lengths.num_completion_tokens
            prev_images, prev_image_placeholders = images, image_placeholders

        if spec_in_progress is not None:
            training_sample_specs.append(
                spec_in_progress.to_training_sample_spec(num_mtp_layers=num_mtp_layers)
            )
        return training_sample_specs

    def rollout_to_training_samples(
        self,
        rollout: Rollout,
        *,
        training_sample_specs: list[TrainingSampleSpec],
    ) -> list[TrainingSample]:
        """Build the samples `training_sample_specs` describe from a full `rollout`'s arrays, in
        one pass over its turns.

        The pass carries the previous turn's prompt + completion forward, so a sample's prompt
        is that, cut to the shared prefix, then its first turn's delta.

        Args:
            rollout: The full rollout (with arrays).
            training_sample_specs: Specs of this rollout, made by
                `rollout_to_training_sample_specs`, in the order to return their samples.
        """
        if not training_sample_specs:
            return []
        specs_by_first_turn_id: dict[int, list[tuple[int, TrainingSampleSpec]]] = {}
        for position, spec in enumerate(training_sample_specs):
            specs_by_first_turn_id.setdefault(spec.rollout_id.turn_id, []).append(
                (position, spec)
            )
        last_turn_id = max(spec.last_turn_id for spec in training_sample_specs)
        samples_by_position: dict[int, TrainingSample] = {}
        # (position, spec, its first turn's shared prefix) until the spec's last turn.
        in_progress: list[tuple[int, TrainingSampleSpec, list[np.ndarray]]] = []
        # The previous turn's prompt + completion, as chunks; joined only where the prefix breaks.
        previous: list[np.ndarray] = []
        previous_len = 0
        for turn_id, turn in enumerate(rollout.turns[: last_turn_id + 1]):
            prompt_prefix_len = turn.prompt_prefix_len
            delta = turn.prompt_delta_token_ids
            if prompt_prefix_len < previous_len:
                previous_token_ids = (
                    previous[0] if len(previous) == 1 else np.concatenate(previous)
                )
                previous = [previous_token_ids[:prompt_prefix_len]]
            in_progress += [
                (position, spec, list(previous))
                for position, spec in specs_by_first_turn_id.get(turn_id, ())
            ]
            previous += [delta, turn.completion_token_ids]
            previous_len = (
                prompt_prefix_len + len(delta) + len(turn.completion_token_ids)
            )
            for entry in in_progress:
                position, spec, prefix = entry
                if spec.last_turn_id != turn_id:
                    continue
                sample = self._build_training_sample(spec, rollout, prefix=prefix)
                samples_by_position[position] = sample
                # A sample's turns continue one another, so its tokens are this turn's
                # prompt + completion, in one array.
                previous = [sample.token_ids]
            in_progress = [
                entry for entry in in_progress if entry[1].last_turn_id != turn_id
            ]
        return [
            samples_by_position[position]
            for position in range(len(training_sample_specs))
        ]

    def materialize(
        self,
        *,
        training_sample_specs: list[TrainingSampleSpec],
        rollout_group: RolloutGroup,
    ) -> list[TrainingSample]:
        """Build the training samples `training_sample_specs` describe, in their order, with one
        `rollout_to_training_samples` pass per rollout.

        Args:
            training_sample_specs: Specs `decide` made from `rollout_group`, in the order to
                return their samples.
            rollout_group: The full group (with arrays) the specs came from.
        """
        positions_by_rollout_id: dict[int, list[int]] = {}
        for position, spec in enumerate(training_sample_specs):
            positions_by_rollout_id.setdefault(spec.rollout_id.rollout_id, []).append(
                position
            )
        samples_by_position: dict[int, TrainingSample] = {}
        for rollout in rollout_group.rollouts:
            positions = positions_by_rollout_id.get(rollout.rollout_id)
            if positions is None:
                continue
            samples = self.rollout_to_training_samples(
                rollout,
                training_sample_specs=[
                    training_sample_specs[position] for position in positions
                ],
            )
            samples_by_position.update(zip(positions, samples, strict=True))
        return [
            samples_by_position[position]
            for position in range(len(training_sample_specs))
        ]

    def _build_training_sample(
        self,
        spec: TrainingSampleSpec,
        rollout: Rollout,
        *,
        prefix: list[np.ndarray],
    ) -> TrainingSample:
        """The sample `spec` describes, from turns `spec.rollout_id.turn_id` to
        `spec.last_turn_id` of `rollout`: `prefix` (the first turn's shared prefix of the
        previous turn's prompt + completion) and the first turn's delta and completion, then each
        later turn's delta and completion. Prompts and env replies are masked out, completions
        trained.
        """
        rollout_advantage = rollout.advantage
        assert rollout_advantage is not None  # checked when the spec was made
        first_turn_id = spec.rollout_id.turn_id
        num_prefix_tokens = rollout.turns[first_turn_id].prompt_prefix_len
        token_ids, loss_mask, logprobs, advantage = list(prefix), [], [], []
        for turn in rollout.turns[first_turn_id : spec.last_turn_id + 1]:
            num_prompt = num_prefix_tokens + len(turn.prompt_delta_token_ids)
            num_prefix_tokens = 0
            num_completion = len(turn.completion_token_ids)
            token_ids += [turn.prompt_delta_token_ids, turn.completion_token_ids]
            loss_mask += [
                np.zeros(num_prompt, dtype=bool),
                np.ones(num_completion, dtype=bool),
            ]
            logprobs += [
                np.zeros(num_prompt, dtype=np.float32),
                turn.completion_logprobs,
            ]
            advantage += [
                np.zeros(num_prompt, dtype=np.float32),
                np.full(num_completion, rollout_advantage, dtype=np.float32),
            ]
        return TrainingSample(
            rollout_id=spec.rollout_id,
            min_policy_version=spec.min_policy_version,
            max_policy_version=spec.max_policy_version,
            token_ids=np.concatenate(token_ids, dtype=np.int64),
            loss_mask=np.concatenate(loss_mask),
            logprobs=np.concatenate(logprobs, dtype=np.float32),
            advantage=np.concatenate(advantage),
            images=spec.images,
            image_placeholders=spec.image_placeholders,
        )
