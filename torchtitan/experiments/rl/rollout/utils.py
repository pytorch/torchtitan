# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import defaultdict

from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.rollout.types import Rollout


def prepare_rollout_metrics(prefix: str, rollouts: list[Rollout]) -> list[m.Metric]:
    """Build rollout-derived metrics: lengths, truncation, reward breakdown, and each turn's
    per-generation metrics (the latter keep their own generator-side keys, unprefixed).

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

    # Per-generation turn metrics (latencies, output tokens) measured by the generator.
    # They carry their own keys (e.g. "generator/..."), so they ride through unprefixed.
    out.extend(
        metric
        for rollout in rollouts
        for rollout_turn in rollout.turns
        for metric in rollout_turn.metrics
    )
    return out
