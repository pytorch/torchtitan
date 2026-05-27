# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from renderers import Message

from torchtitan.experiments.rl.types import Episode


class RolloutStatus(StrEnum):
    """Per-rollout terminal status.

    Absorbs vLLM ``finish_reason`` and renderer/wrapper failure modes
    into one categorical axis. Trainer-facing code reads
    ``is_error()`` to filter, ``is_truncated()`` for truncation metrics.
    """

    COMPLETED = "completed"
    TRUNCATED_LENGTH = "truncated_length"
    TRUNCATED_OVERFLOW = "truncated_overflow"
    ERROR_PARSE = "error_parse"
    ERROR_TIMEOUT = "error_timeout"
    ERROR_ABORT = "error_abort"

    def is_truncated(self) -> bool:
        return self.value.startswith("truncated")

    def is_error(self) -> bool:
        return self.value.startswith("error")


@dataclass(kw_only=True, slots=True)
class RolloutTurn:
    """One generator call + the env response to that call.

    Shape legend:
        T_p: prompt token length for this turn.
        T_r: response token length for this turn.
        M_r: response message count (assistant + env messages this turn).
    """

    prompt_token_ids: list[int]
    response_token_ids: list[int]
    response_logprobs: list[float]
    policy_version: int
    response_messages: list[Message] = field(default_factory=list)


@dataclass(kw_only=True, slots=True)
class Rollout:
    """A complete rollout: ordered turns + terminal state + reward + identity.

    Args:
        group_id: Stable ID for the prompt group used for advantage centering.
        sample_idx: Sibling index within the group.
        turns: Ordered rollout turns. Shape ``[K]``.
        status: Rollout-level terminal status.
        reward: Final scalar reward, filled by the rubric.
        reward_components: Decomposed reward metrics, filled by the rubric.
        advantage: GRPO advantage (reward - group_mean), filled by the controller.
    """

    group_id: str | None = None
    sample_idx: int | None = None
    turns: list[RolloutTurn] = field(default_factory=list)
    status: RolloutStatus = RolloutStatus.COMPLETED
    reward: float | None = None
    reward_components: dict[str, float] = field(default_factory=dict)
    advantage: float | None = None


def validate_rollout_output(output: Rollout) -> None:
    """Raise when rollout token/logprob shapes are inconsistent."""
    for turn_idx, turn in enumerate(output.turns):
        if len(turn.response_token_ids) != len(turn.response_logprobs):
            raise ValueError(
                f"turn {turn_idx}: response_token_ids has "
                f"{len(turn.response_token_ids)} entries but response_logprobs "
                f"has {len(turn.response_logprobs)}"
            )


def last_assistant_text(output: Rollout) -> str:
    """Return the assistant message text from the last turn, or ``""``."""
    if not output.turns:
        return ""
    for msg in output.turns[-1].response_messages:
        if msg.get("role") == "assistant":
            return msg.get("content") or ""
    return ""


def rollout_output_to_episode(output: Rollout, *, text: str = "") -> Episode:
    """Flatten a single-turn ``Rollout`` into the batcher's ``Episode``.

    Multi-turn flattening is the ReplayBuffer PR's job.

    Args:
        output: A finished ``Rollout``. Must have exactly one turn.
        text: Decoded assistant text (typically ``last_assistant_text(output)``).
    """
    if len(output.turns) != 1:
        raise ValueError(
            f"rollout_output_to_episode expects single-turn rollouts; "
            f"got {len(output.turns)} turns. Multi-turn ships with ReplayBuffer."
        )
    if output.reward is None:
        raise ValueError("rollout_output_to_episode requires reward to be set")
    turn = output.turns[0]
    return Episode(
        policy_version=turn.policy_version,
        prompt_idx=output.sample_idx if output.sample_idx is not None else 0,
        prompt_token_ids=turn.prompt_token_ids,
        text=text,
        token_ids=turn.response_token_ids,
        token_logprobs=turn.response_logprobs,
        reward=output.reward,
        advantage=output.advantage if output.advantage is not None else 0.0,
    )


@dataclass(frozen=True, kw_only=True, slots=True)
class DatasetOutput:
    """One row from a task's dataset, ready for env construction.

    Args:
        env_name: Task identifier used for multi-task dispatch.
            For single-task tasks this is a constant.
        env_input: Task-specific payload consumed by ``Task.make_envs``.
            Each task defines its own typed payload; multi-task tasks
            downcast by ``env_name``.
    """

    env_name: str
    env_input: object
