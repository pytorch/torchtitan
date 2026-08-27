# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Recorder that saves produced rollouts to disk (JSONL) for inspection and debugging."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from torchtitan.config import Configurable
from torchtitan.experiments.rl.rollout import Rollout, RolloutGroup, RolloutTurn
from torchtitan.observability import structured_logger as sl

# TODO(recorders): if a second recorder appears (e.g. an training_sample recorder), generalize to a list of
# recorders behind one interface, instead of bespoke per-type classes.


def _json_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class KeepExtremeRewardsFilter(Configurable):
    """Keep the k highest- and k lowest-reward rollouts of each group (unscored skipped).

    Example:

        # a group whose rollouts have rewards [0.9, 0.9, 0.2, None, -0.3, -1, -1]
        KeepExtremeRewardsFilter.Config(k=2).build()([group])
        # -> rollouts with rewards [-1, -1, 0.9, 0.9]   (None skipped)
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        k: int = 1
        """Rollouts to keep from each end (highest and lowest) of every group."""

    def __init__(self, config: Config) -> None:
        self._k = config.k

    @sl.log_trace_span("keep_extreme_rewards")
    def __call__(self, groups: list[RolloutGroup]) -> list[Rollout]:
        picked: list[Rollout] = []
        for group in groups:
            ranked = sorted(
                (rollout for rollout in group.rollouts if rollout.reward is not None),
                key=lambda rollout: rollout.reward,
            )
            k = self._k
            # small group (<= 2k): keep all, avoiding a highest/lowest overlap; else both ends.
            picked.extend(ranked if len(ranked) <= 2 * k else ranked[:k] + ranked[-k:])
        return picked


class RolloutSampleRecorder(Configurable):
    """Append rollouts to a JSONL file for debugging and inspection.

    The filter selects which rollouts to record each step (default: highest + lowest reward per group);
    raw token / logprob arrays are recorded only when `Config.log_tensors` / `log_logprobs` opt in.

    Example:

        recorder = RolloutSampleRecorder.Config().build(dump_dir="outputs/rl")
        recorder.record(is_validation=False, rollout_groups=groups)
        # -> outputs/rl/rollout_samples.jsonl, one JSON line per recorded rollout
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        filter: KeepExtremeRewardsFilter.Config = field(
            default_factory=KeepExtremeRewardsFilter.Config
        )
        """Selects which rollouts to record each step."""
        filename: str = "rollout_samples.jsonl"
        """JSONL filename written under `dump_dir`."""
        log_tensors: bool = False
        """Also record the raw prompt/completion token-id arrays per turn."""
        log_logprobs: bool = False
        """Also record the raw per-token completion logprob arrays per turn."""

    def __init__(self, config: Config, *, dump_dir: str) -> None:
        self._filter = config.filter.build()
        self._log_tensors = config.log_tensors
        self._log_logprobs = config.log_logprobs
        self._path = Path(dump_dir) / config.filename
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @sl.log_trace_span("rollout_record")
    def record(
        self,
        *,
        is_validation: bool,
        rollout_groups: list[RolloutGroup],
    ) -> None:
        """Append the filtered rollouts to the JSONL, one JSON object per rollout."""
        lines = []
        for rollout in self._filter(rollout_groups):
            record = {
                "is_validation": is_validation,
                **self._encode_rollout(rollout),
            }
            # Rollout messages can hold objects json.dumps can't serialize
            # natively, e.g. tool-calling results
            lines.append(json.dumps(record, default=_json_default) + "\n")
        with self._path.open("a", encoding="utf-8") as file:
            file.write("".join(lines))

    def _encode_rollout(self, rollout: Rollout) -> dict:
        """One JSON object per rollout: its scalar fields + each turn (see `_encode_turn`)."""
        return {
            "group_id": rollout.group_id,
            "rollout_id": rollout.rollout_id,
            "status": rollout.status,
            "reward": rollout.reward,
            "reward_breakdown": rollout.reward_breakdown,
            "advantage": rollout.advantage,
            "turns": [self._encode_turn(turn) for turn in rollout.turns],
        }

    def _encode_turn(self, turn: RolloutTurn) -> dict:
        """One JSON object per turn. The large token / logprob arrays are opt-in
        (`log_tensors` / `log_logprobs`)."""
        encoded = {
            "turn_id": turn.rollout_id.turn_id,
            "min_policy_version": turn.min_policy_version,
            "max_policy_version": turn.max_policy_version,
            "prompt_messages": turn.prompt_messages,
            "completion_message": turn.completion_message,
            "env_messages": turn.env_messages,
            "env_rewards": turn.env_rewards,
        }
        if self._log_tensors:
            encoded["prompt_token_ids"] = turn.prompt_token_ids
            encoded["completion_token_ids"] = turn.completion_token_ids
        if self._log_logprobs:
            encoded["completion_logprobs"] = turn.completion_logprobs
        return encoded
