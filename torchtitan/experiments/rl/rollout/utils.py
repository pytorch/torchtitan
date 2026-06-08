# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import defaultdict

from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.rollout.types import Rollout, RolloutTurn
from torchtitan.experiments.rl.types import Episode


def completion_text(turn: RolloutTurn) -> str:
    """The turn's completion message text, or `""`."""
    msg = turn.completion_message
    return (msg.get("content") or "") if msg else ""


def last_completion_text(rollout: Rollout) -> str:
    """The completion text of the rollout's last turn, or `""`."""
    return completion_text(rollout.turns[-1]) if rollout.turns else ""


def rollout_to_episode(rollout: Rollout) -> Episode:
    """Flatten a scored `Rollout` into ONE training `Episode`.

    Trains on the rollout's last turn that produced a completion — whose prompt already holds
    the full conversation history — with the rollout's trajectory reward/advantage. A single-turn
    rollout yields the obvious `(prompt, completion)` pair.

    Example:

        rollout_to_episode(rollout)
        # -> Episode(prompt_token_ids=last_turn_prompt, completion_token_ids=last_turn_completion,
        #            reward=1.0, advantage=0.3, ...)
    """
    # TODO(prefix-matching): a multi-turn rollout's turns share a growing prefix, so the whole
    #   trajectory should pack into ONE sequence (loss-masking each assistant turn) rather than
    #   training only the last turn. Branch into multiple Episodes only where a turn's prompt
    #   diverges from the previous turn's prompt+completion (e.g. the env edited/compacted
    #   history), since those turns no longer share a prefix.
    trainable_turns = [
        rollout_turn
        for rollout_turn in rollout.turns
        if rollout_turn.completion_token_ids
    ]
    last_turn = trainable_turns[-1]
    advantage = rollout.advantage if rollout.advantage is not None else 0.0
    return Episode(
        policy_version=last_turn.policy_version,
        sample_id=rollout.sample_id,
        prompt_token_ids=last_turn.prompt_token_ids,
        completion_text=completion_text(last_turn),
        completion_token_ids=last_turn.completion_token_ids,
        completion_logprobs=last_turn.completion_logprobs,
        reward=rollout.reward,
        advantage=advantage,
    )


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

    out: list[m.Metric] = [
        m.Metric(f"{prefix}/response_length", m.Mean.from_list(completion_lens)),
        m.Metric(f"{prefix}/response_length", m.Max.from_list(completion_lens)),
        m.Metric(f"{prefix}/prompt_length", m.Mean.from_list(prompt_lens)),
        m.Metric(f"{prefix}/prompt_length", m.Max.from_list(prompt_lens)),
        m.Metric(f"{prefix}/total_length", m.Mean.from_list(total_lens)),
        m.Metric(f"{prefix}/total_length", m.Max.from_list(total_lens)),
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
