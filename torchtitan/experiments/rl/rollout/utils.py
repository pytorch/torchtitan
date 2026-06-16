# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import defaultdict

from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.rollout.types import Rollout
from torchtitan.experiments.rl.types import Episode, RolloutID


# TODO: evaluate renaming Episode -> TrainingSample once we align overloaded "sample" names. E.g. Dataset also uses it.
def rollout_to_episodes(rollout: Rollout) -> list[Episode]:
    """Pack a scored `Rollout` into training episodes — usually one, or several where its turns branch.

    Turns that share a growing prefix (each prompt continues the previous prompt + completion) are
    packed into ONE episode: prompts and env replies are masked out, completions are trained. A new
    episode (branch) opens wherever that prefix breaks (history was edited).

    This branching has three common causes:
    1) The rollout loop purposely edits the history: For example, when compacting a long conversation;
    2) Thinking is removed from history: This is a flag in the `renderer` used in the `Rollouter`. Choose
        `preserve_all_thinking=True` to avoid stripping thinking (default=True);
    3) Some error or change from retokenization: Those can be avoided by doing token-in-token-out (TITO),
        i.e. you give tokens to the generators, and you append (bridge) the tokens received, leaving no room for changes.
        TITO is our default;

    Each turn's `version_intervals` (the policy version it was sampled at) shift to where the turn's
    completion lands in the packed sequence, so a multi-turn episode records the version per turn.

    Example (5 turns; the env compacts history before turn 3, so the prefix breaks -> 2 episodes).
    P = prompt; C = completion; E = env reply
        turn0 (v5): prompt=[P1]              completion=[C1]
        turn1 (v6): prompt=[P1,C1,E1]        completion=[C2]
        turn2 (v6): prompt=[P1,C1,E1,C2,E2]  completion=[C3]
        turn3 (v6): prompt=[P2]              completion=[C4]   # history compacted -> prefix breaks
        turn4 (v7): prompt=[P2,C4,E4]        completion=[C5]
        # -> episode 0: token_ids=[P1,C1,E1,C2,E2,C3], loss_mask=[0,1,0,1,0,1]   rollout_id.turn_id=0
        #              version_intervals=[(1,5),(3,6),(5,6)]   # each completion at its sampling version
        #    episode 1: token_ids=[P2,C4,E4,C5],        loss_mask=[0,1,0,1]       rollout_id.turn_id=3
        #              version_intervals=[(1,6),(3,7)]
    """
    rollout_advantage = rollout.advantage
    episodes: list[Episode] = []

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
        # False when the env edited history -> open a new episode (branch).
        extends_prev = (
            prompt[: len(prev_prompt_and_completion)] == prev_prompt_and_completion
        )
        if not episodes or not extends_prev:
            # Start a new episode; its rollout_id.turn_id marks the turn the segment begins at.
            episodes.append(
                Episode(
                    policy_version=rollout_turn.policy_version,
                    rollout_id=RolloutID(
                        group_id=rollout.group_id,
                        rollout_id=rollout.rollout_id,
                        turn_id=turn_idx,
                    ),
                    token_ids=[],
                    loss_mask=[],
                    logprobs=[],
                    advantage=[],
                    version_intervals=[],
                )
            )
            # The whole prompt opens this episode.
            prefix_len = 0
        else:
            # Only the env-reply suffix is new.
            prefix_len = len(prev_prompt_and_completion)

        # Untrained prefix delta (the prompt or the new env reply), then the trained completion.
        episode = episodes[-1]
        prompt_delta = prompt[prefix_len:]
        num_delta = len(prompt_delta)
        num_completion = len(rollout_turn.completion_token_ids)
        episode.token_ids += prompt_delta
        episode.loss_mask += [False] * num_delta
        episode.logprobs += [0.0] * num_delta
        episode.advantage += [0.0] * num_delta
        # Shift this turn's version boundaries to where its completion lands in the packed
        # sequence, so a packed multi-turn episode records the version each turn was sampled at.
        completion_offset = len(episode.token_ids)
        episode.version_intervals += [
            (completion_offset + start, version)
            for start, version in rollout_turn.version_intervals
        ]
        episode.token_ids += rollout_turn.completion_token_ids
        episode.loss_mask += [True] * num_completion
        episode.logprobs += rollout_turn.completion_logprobs
        episode.advantage += [rollout_advantage] * num_completion

        prev_prompt_and_completion = prompt + rollout_turn.completion_token_ids

    return episodes


def prepare_rollout_metrics(prefix: str, rollouts: list[Rollout]) -> list[m.Metric]:
    """Build rollout-derived metrics (lengths, truncation, reward breakdown).

    Args:
        prefix: Metric namespace (e.g. `"rollout"` or `"validation"`).
        rollouts: Rollouts to summarize.
    """
    # Lengths, truncation, reward
    # TODO: adapt for multi-turn rollouts
    completion_lens = [
        len(rollout_turn.completion_token_ids)
        for rollout in rollouts
        for rollout_turn in rollout.turns
    ]
    prompt_lens = [
        len(rollout.turns[0].prompt_token_ids) for rollout in rollouts if rollout.turns
    ]
    total_lens = [
        len(rollout.turns[-1].prompt_token_ids)
        + len(rollout.turns[-1].completion_token_ids)
        for rollout in rollouts
        if rollout.turns
    ]

    truncated = [float(rollout.status.is_truncated()) for rollout in rollouts]
    rewards = [rollout.reward for rollout in rollouts if rollout.reward is not None]
    num_turns = [float(len(rollout.turns)) for rollout in rollouts]

    out: list[m.Metric] = [
        m.Metric(f"{prefix}/output_tokens", m.Mean.from_list(completion_lens)),
        m.Metric(f"{prefix}/output_tokens", m.Std.from_list(completion_lens)),
        m.Metric(f"{prefix}/output_tokens", m.Max.from_list(completion_lens)),
        m.Metric(f"{prefix}/response_length", m.Mean.from_list(completion_lens)),
        m.Metric(f"{prefix}/response_length", m.Max.from_list(completion_lens)),
        m.Metric(f"{prefix}/prompt_length", m.Mean.from_list(prompt_lens)),
        m.Metric(f"{prefix}/prompt_length", m.Max.from_list(prompt_lens)),
        m.Metric(f"{prefix}/total_length", m.Mean.from_list(total_lens)),
        m.Metric(f"{prefix}/total_length", m.Max.from_list(total_lens)),
        m.Metric(f"{prefix}/num_turns", m.Mean.from_list(num_turns)),
        m.Metric(f"{prefix}/num_turns", m.Max.from_list(num_turns)),
        m.Metric(f"{prefix}/truncation_rate", m.Mean.from_list(truncated)),
        m.Metric(f"{prefix}_reward", m.SummaryStats.from_list(rewards)),
    ]

    # Per-component reward breakdown
    values_by_name: dict[str, list[float]] = defaultdict(list)
    for rollout in rollouts:
        for name, value in rollout.reward_breakdown.items():
            values_by_name[name].append(float(value))
    out.extend(
        m.Metric(f"{prefix}_reward/component/{name}", m.Mean.from_list(values))
        for name, values in sorted(values_by_name.items())
    )
    return out
