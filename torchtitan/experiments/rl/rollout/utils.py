# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import defaultdict

from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.rollout.types import Rollout
from torchtitan.experiments.rl.types import Episode


def rollout_to_episodes(rollout: Rollout) -> list[Episode]:
    """Transforms a `Rollout` into training episodes.

    NOTE: Each rollout may become more than one `Episode`. If `prompt_token_ids` at turn N is not a
    prefix of turn N+1, indicating history branching, we branch it into two different episodes:
    one up to turn N, and another one from N+1 history forward. Prefix check is possible because
    every RolloutTurn.prompt_token_ids carries all history up to that turn.

    Example (3 turns):
        P = prompt; C = completion; E = env reply
        turn0: prompt=[P]             completion=[C1]   #  P  -> mask 0,  C1 -> mask 1
        turn1: prompt=[P,C1,E1]       completion=[C2]   #  E1 -> mask 0,  C2 -> mask 1
        turn2: prompt=[P,C1,E1,C2,E2] completion=[D3]   #  E2 -> mask 0,  C3 -> mask 1
        # -> [Episode(token_ids=[P,C1,E1,C2,E2,C3], loss_mask=[0,1,0,1,0,1],
        #                    logprobs=[0,l1,0,l2,0,l3], advantage=...)]
    """
    rollout_advantage = rollout.advantage if rollout.advantage is not None else 0.0
    episodes: list[Episode] = []
    prev_prompt_and_completion: list[int] = []

    for rollout_turn in rollout.turns:
        # Prompt-only turns (e.g. the prompt went over budget) have nothing to train on.
        if not rollout_turn.completion_token_ids:
            continue

        # Open a new episode on the first turn, or wherever the growing prefix breaks (history
        # was edited / re-tokenized); otherwise extend the current episode.
        prompt = rollout_turn.prompt_token_ids
        extends_prev = (
            prompt[: len(prev_prompt_and_completion)] == prev_prompt_and_completion
        )
        if not episodes or not extends_prev:
            episodes.append(
                Episode(
                    # TODO(async): carry per-token version_intervals in episode
                    policy_version=rollout_turn.policy_version,
                    sample_id=(
                        rollout.sample_id
                        if not episodes
                        else f"{rollout.sample_id}/branch={len(episodes)}"
                    ),
                    token_ids=[],
                    loss_mask=[],
                    logprobs=[],
                    advantage=[],
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
        episode.token_ids += prompt_delta
        episode.loss_mask += [False] * len(prompt_delta)
        episode.logprobs += [0.0] * len(prompt_delta)
        episode.advantage += [0.0] * len(prompt_delta)
        episode.token_ids += rollout_turn.completion_token_ids
        episode.loss_mask += [True] * len(rollout_turn.completion_token_ids)
        episode.logprobs += rollout_turn.completion_logprobs
        episode.advantage += [rollout_advantage] * len(
            rollout_turn.completion_token_ids
        )

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
    completion_lens = [len(t.completion_token_ids) for r in rollouts for t in r.turns]
    prompt_lens = [len(r.turns[0].prompt_token_ids) for r in rollouts if r.turns]
    total_lens = [
        len(r.turns[-1].prompt_token_ids) + len(r.turns[-1].completion_token_ids)
        for r in rollouts
        if r.turns
    ]

    truncated = [float(r.status.is_truncated()) for r in rollouts]
    rewards = [r.reward for r in rollouts if r.reward is not None]
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
