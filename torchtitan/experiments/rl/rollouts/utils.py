# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import defaultdict

from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.rollouts.types import Rollout
from torchtitan.experiments.rl.types import Episode


def last_assistant_text(rollout: Rollout) -> str:
    """Return the assistant message text from the last turn, or `""`."""
    if not rollout.turns:
        return ""
    for msg in rollout.turns[-1].response_messages:
        if msg.get("role") == "assistant":
            return msg.get("content") or ""
    return ""


def rollout_to_episode(rollout: Rollout, *, text: str = "") -> Episode:
    """Flatten a single-turn `Rollout` into the batcher's `Episode`.

    Args:
        rollout: A finished `Rollout`. Must have exactly one turn.
        text: Decoded assistant text (typically `last_assistant_text(rollout)`).

    Returns:
        `Episode` containing prompt tokens, response tokens, sampling
        logprobs, reward, and advantage for training.
    """
    # TODO: support multi-turn rollout flattening.
    if len(rollout.turns) != 1:
        raise ValueError(
            f"rollout_to_episode expects exactly one turn; got {len(rollout.turns)}."
        )
    turn = rollout.turns[0]
    return Episode(
        policy_version=turn.policy_version,
        prompt_idx=rollout.sample_idx,
        prompt_token_ids=turn.prompt_token_ids,
        text=text,
        token_ids=turn.response_token_ids,
        token_logprobs=turn.response_logprobs,
        reward=rollout.reward,
        advantage=rollout.advantage if rollout.advantage is not None else 0.0,
    )


def prepare_rollout_metrics(prefix: str, rollouts: list[Rollout]) -> list[m.Metric]:
    """Build rollout-derived metrics for one collection round.

    Args:
        prefix: Metric namespace (e.g. `"rollout"` or `"validation"`).
        rollouts: Rollouts to summarize.

    Returns:
        Metrics for response length, prompt length, total length, truncation
        rate, reward summary, and per-component reward means. Reward-component
        metrics nest under `{prefix}/reward/component/<name>`.
    """
    # Lengths, truncation, reward
    response_lens = [len(t.response_token_ids) for r in rollouts for t in r.turns]
    prompt_lens = [len(r.turns[0].prompt_token_ids) for r in rollouts if r.turns]
    # Per-rollout totals (handles multi-turn correctly; single-turn collapses
    # to the same value as prompt_lens[i] + response_lens[i]).
    total_lens = [
        len(r.turns[0].prompt_token_ids)
        + sum(len(t.response_token_ids) for t in r.turns)
        for r in rollouts
        if r.turns
    ]
    truncated = [float(r.status.is_truncated()) for r in rollouts]
    rewards = [r.reward for r in rollouts if r.reward is not None]

    out: list[m.Metric] = [
        m.Metric(f"{prefix}/response_length", m.Mean.from_list(response_lens)),
        m.Metric(f"{prefix}/response_length", m.Max.from_list(response_lens)),
        m.Metric(f"{prefix}/prompt_length", m.Mean.from_list(prompt_lens)),
        m.Metric(f"{prefix}/prompt_length", m.Max.from_list(prompt_lens)),
        m.Metric(f"{prefix}/total_length", m.Max.from_list(total_lens)),
        m.Metric(f"{prefix}/truncation_rate", m.Mean.from_list(truncated)),
        m.Metric(f"{prefix}/reward", m.SummaryStats.from_list(rewards)),
    ]

    # Per-component reward breakdown
    values_by_name: dict[str, list[float]] = defaultdict(list)
    for r in rollouts:
        for name, value in r.reward_components.items():
            values_by_name[name].append(float(value))
    out.extend(
        m.Metric(f"{prefix}/reward/component/{name}", m.Mean.from_list(values))
        for name, values in sorted(values_by_name.items())
    )
    return out
