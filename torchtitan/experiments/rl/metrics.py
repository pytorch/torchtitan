# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rollout-side metric builders and rollout sample loggers.

Lives outside :mod:`grpo` so the controller stays focused on actor
orchestration. Three complementary surfaces:

- :func:`aggregate_rewards` + :func:`format_validation` — produce the
  short mean/component summary the validation log prints.
- :func:`build_rollout_metrics` + :func:`build_replay_metrics` —
  emit typed ``m.Metric`` records for the per-step MetricsProcessor
  log (rollout length stats, reward / advantage SummaryStats,
  group_std, zero_std fraction, per-component reward breakdown).
- :func:`log_first_sample` + :func:`dump_rollouts_jsonl` — surface
  representative rollouts to stdout and a JSONL file for an inspector
  subagent (gated on ``config.log_samples``).
"""

from __future__ import annotations

import json
import logging
import os
import statistics
from collections import defaultdict
from collections.abc import Sequence

from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.types import (
    ReplaySample,
    RolloutOutput,
    RolloutStatus,
)

logger = logging.getLogger(__name__)

__all__ = [
    "aggregate_rewards",
    "build_replay_metrics",
    "build_rollout_metrics",
    "dump_rollouts_jsonl",
    "format_validation",
    "log_first_sample",
]


def build_rollout_metrics(
    rollouts: Sequence[RolloutOutput],
    *,
    prefix: str = "rollout",
) -> list[m.Metric]:
    """Length, truncation, status, and per-reward-component records.

    Every emitted key starts with ``prefix/`` so the same helper can
    be used for both training-time rollouts (``prefix="rollout"``) and
    validation rollouts (``prefix="validation"``) without colliding
    with the unprefixed ``reward`` SummaryStats emitted by
    :func:`build_replay_metrics`.

    Lengths are computed over the whole rollout (sum across turns) so
    multi-turn samples are reported correctly. Truncation rate uses
    the rollout-level status.
    """
    if not rollouts:
        return []
    response_lens: list[int] = []
    prompt_lens: list[int] = []
    total_lens: list[int] = []
    truncated: list[float] = []
    errored: list[float] = []
    completed: list[float] = []
    num_turns: list[int] = []
    component_values: dict[str, list[float]] = defaultdict(list)
    rewards: list[float] = []
    for r in rollouts:
        r_resp = sum(len(t.response_token_ids) for t in r.turns)
        r_prompt = r.turns[0].prompt_token_ids.__len__() if r.turns else 0
        response_lens.append(r_resp)
        prompt_lens.append(r_prompt)
        total_lens.append(r_prompt + r_resp)
        num_turns.append(len(r.turns))
        truncated.append(1.0 if r.status == RolloutStatus.TRUNCATED else 0.0)
        errored.append(1.0 if r.status == RolloutStatus.ERROR else 0.0)
        completed.append(1.0 if r.status == RolloutStatus.COMPLETED else 0.0)
        if r.reward is not None:
            rewards.append(float(r.reward))
        for name, value in r.reward_components.items():
            component_values[name].append(float(value))

    records: list[m.Metric] = [
        m.Metric(f"{prefix}/response_length", m.Mean.from_list(response_lens)),
        m.Metric(f"{prefix}/response_length", m.Max.from_list(response_lens)),
        m.Metric(f"{prefix}/prompt_length", m.Mean.from_list(prompt_lens)),
        m.Metric(f"{prefix}/prompt_length", m.Max.from_list(prompt_lens)),
        m.Metric(f"{prefix}/total_length", m.Max.from_list(total_lens)),
        m.Metric(f"{prefix}/num_turns", m.Mean.from_list(num_turns)),
        m.Metric(f"{prefix}/num_turns", m.Max.from_list(num_turns)),
        m.Metric(f"{prefix}/truncated_fraction", m.Mean.from_list(truncated)),
        m.Metric(f"{prefix}/error_fraction", m.Mean.from_list(errored)),
        m.Metric(f"{prefix}/completed_fraction", m.Mean.from_list(completed)),
    ]
    # All keys prefixed so the same builder can be re-used for both
    # rollouts (prefix="rollout") and validation (prefix="validation")
    # without colliding with replay's unprefixed ``reward`` SummaryStats.
    if rewards:
        records.append(
            m.Metric(f"{prefix}/reward", m.SummaryStats.from_list(rewards))
        )
    for name, values in sorted(component_values.items()):
        records.append(
            m.Metric(f"{prefix}/component/{name}", m.Mean.from_list(values))
        )
    return records


def build_replay_metrics(samples: Sequence[ReplaySample]) -> list[m.Metric]:
    """Advantage + reward stats + per-group reward std + policy_version range.

    Group-std and zero-std-fraction quantify how often a group's
    rollouts collapse to the same reward (= no learning signal).
    """
    if not samples:
        return []
    by_group: dict[str, list[float]] = defaultdict(list)
    rewards: list[float] = []
    advantages: list[float] = []
    policy_versions: list[int] = []
    for s in samples:
        rewards.append(float(s.reward))
        advantages.append(float(s.advantage))
        policy_versions.append(int(s.policy_version))
        by_group[s.group_id].append(float(s.reward))
    group_stds = [
        statistics.pstdev(r) if len(r) > 1 else 0.0 for r in by_group.values()
    ]
    zero_std_frac = (
        sum(1 for s in group_stds if s == 0.0) / len(group_stds)
        if group_stds
        else 0.0
    )
    records: list[m.Metric] = [
        m.Metric("reward", m.SummaryStats.from_list(rewards)),
        m.Metric("advantage", m.SummaryStats.from_list(advantages)),
        m.Metric("reward/group_std", m.Mean.from_list(group_stds)),
        m.Metric("reward/group_std", m.Max.from_list(group_stds)),
        m.Metric("reward/zero_std_frac", m.NoReduce(zero_std_frac)),
        m.Metric("replay/policy_version", m.Min.from_list(policy_versions)),
        m.Metric("replay/policy_version", m.Max.from_list(policy_versions)),
    ]
    return records


def aggregate_rewards(
    rollouts: Sequence[RolloutOutput],
) -> dict[str, float | dict[str, float]]:
    """Mean reward + per-component mean across non-ERROR rollouts.

    Returns a dict with ``mean_reward``, ``components``, ``total``, and
    ``fraction_truncated``. ERROR-status rollouts (reward is ``None``)
    are excluded from the numerator but still counted in ``total`` so
    the denominator captures the fail rate honestly.
    """
    scored = [r for r in rollouts if r.reward is not None]
    rewards = [float(r.reward) for r in scored]  # type: ignore[arg-type]
    components: dict[str, list[float]] = defaultdict(list)
    for r in scored:
        for k, v in r.reward_components.items():
            components[k].append(float(v))
    total = len(rollouts) or 1
    return {
        "mean_reward": sum(rewards) / total,
        "components": {k: sum(v) / total for k, v in components.items()},
        "total": len(rollouts),
        "fraction_truncated": (
            sum(1 for r in rollouts if r.status == RolloutStatus.TRUNCATED) / total
        ),
    }


def format_validation(result: dict) -> str:
    """Short line: ``mean_reward=+0.300 (correctness=+0.250, format=+0.50) | truncated=10%``."""
    comp_str = ", ".join(f"{k}={v:+.3f}" for k, v in result["components"].items())
    return (
        f"mean_reward={result['mean_reward']:+.3f} "
        f"({comp_str}) | truncated={result['fraction_truncated']:.0%}"
    )


def log_first_sample(rollouts: Sequence[RolloutOutput]) -> None:
    """Log the first rollout's last assistant message for quick sanity."""
    if not rollouts:
        return
    r = rollouts[0]
    if not r.turns or not r.turns[-1].response_messages:
        return
    msg = r.turns[-1].response_messages[0]
    content = str(msg.get("content") or "")[:200].replace("\n", " ")
    reward_str = "n/a" if r.reward is None else f"{r.reward:+.3f}"
    logger.info(
        "  [%s/%d] reward=%s  A: %s",
        r.group_id,
        r.sample_idx,
        reward_str,
        content,
    )


def dump_rollouts_jsonl(
    rollouts: Sequence[RolloutOutput],
    *,
    dump_folder: str,
    train_step: int,
) -> None:
    """Append one JSON line per rollout to ``{dump_folder}/rollouts.jsonl``.

    Best-effort: a write failure logs but doesn't crash the rollout loop.
    Schema is shallow on purpose — full messages + reward + status, no
    token ids — so an inspector subagent can read the file directly.
    """
    path = os.path.join(dump_folder, "rollouts.jsonl")
    try:
        os.makedirs(dump_folder, exist_ok=True)
        with open(path, "a") as f:
            for r in rollouts:
                turns = [
                    {
                        "prompt_messages": t.prompt_messages,
                        "response_messages": t.response_messages,
                        "policy_version": t.policy_version,
                        "num_response_tokens": len(t.response_token_ids),
                    }
                    for t in r.turns
                ]
                f.write(
                    json.dumps(
                        {
                            "train_step": train_step,
                            "group_id": r.group_id,
                            "sample_idx": r.sample_idx,
                            "status": str(r.status),
                            "reward": r.reward,
                            "reward_components": r.reward_components,
                            "turns": turns,
                        }
                    )
                    + "\n"
                )
    except OSError as exc:
        logger.warning("rollouts.jsonl write failed: %s", exc)
