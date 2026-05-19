# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rollout-side metric builders and rollout sample loggers.

Lives outside :mod:`grpo` so the controller stays focused on actor
orchestration. Two complementary surfaces:

- :func:`aggregate_rewards` + :func:`format_validation` — produce the
  short mean/component summary the validation log prints.
- :func:`log_first_sample` + :func:`dump_rollouts_jsonl` — surface
  representative rollouts to stdout and a JSONL file for an inspector
  subagent (gated on ``config.log_samples``).
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from collections.abc import Sequence

from torchtitan.experiments.rl.types import RolloutOutput, RolloutStatus

logger = logging.getLogger(__name__)

__all__ = [
    "aggregate_rewards",
    "dump_rollouts_jsonl",
    "format_validation",
    "log_first_sample",
]


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
