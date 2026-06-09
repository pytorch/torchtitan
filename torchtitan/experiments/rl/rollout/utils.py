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


def last_completion_text(rollout: Rollout) -> str:
    """Return the completion message text from the last turn, or `""`."""
    if not rollout.turns:
        return ""
    msg = rollout.turns[-1].completion_message
    return (msg.get("content") or "") if msg else ""


def rollout_to_episode(rollout: Rollout) -> Episode:
    """Flatten a scored single-turn `Rollout` into an `Episode`, a class
    that holds only the information needed for training.
    """
    # TODO: support multi-turn rollout flattening.
    # TODO(branching): when a turn's prompt history diverges from the previous turn's
    #       (e.g. the env edited/compacted history), the turns no longer share a prefix
    #       and must be split into separate training sequences instead of one flat episode.
    # TODO: rename Episode -> TrainingSample / rollout_to_episode ->
    #       rollout_to_training_sample (consistent with TrainingBatch).
    if len(rollout.turns) != 1:
        raise ValueError(
            f"rollout_to_episode expects exactly one turn; got {len(rollout.turns)}."
        )
    turn = rollout.turns[0]
    return Episode(
        policy_version=turn.policy_version,
        sample_id=rollout.sample_id,
        prompt_token_ids=turn.prompt_token_ids,
        completion_text=last_completion_text(rollout),
        completion_token_ids=turn.completion_token_ids,
        completion_logprobs=turn.completion_logprobs,
        reward=rollout.reward,
        advantage=rollout.advantage if rollout.advantage is not None else 0.0,
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
