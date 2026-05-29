# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from renderers import Message


_TRUNCATED = frozenset({"truncated_length", "truncated_overflow"})
_ERROR = frozenset({"error_parse", "error_timeout", "error_abort"})


class RolloutStatus(StrEnum):
    """Per-rollout terminal status.

    Absorbs vLLM `finish_reason` and renderer/wrapper failure modes
    into one categorical axis. Trainer-facing code reads `is_error()`
    to filter, `is_truncated()` for truncation metrics.
    """

    COMPLETED = "completed"
    TRUNCATED_LENGTH = "truncated_length"
    TRUNCATED_OVERFLOW = "truncated_overflow"
    ERROR_PARSE = "error_parse"
    ERROR_TIMEOUT = "error_timeout"
    ERROR_ABORT = "error_abort"

    def is_truncated(self) -> bool:
        return self.value in _TRUNCATED

    def is_error(self) -> bool:
        return self.value in _ERROR


@dataclass(kw_only=True, slots=True)
class RolloutTurn:
    """One generator completion + the env response to that completion."""

    prompt_token_ids: list[int]  # [L_prompt]
    """Tokens the generator saw as prompt for this turn."""

    response_token_ids: list[int]  # [L_response]
    """Tokens the generator produced (response only)."""

    response_logprobs: list[float]  # [L_response]
    """Per-token logprobs from the generator policy."""

    policy_version: int
    """Trainer policy version when this response was sampled."""

    response_messages: list[Message] = field(default_factory=list)  # [M_response]
    """Parsed assistant + env-appended messages (assistant first)."""

    reward_components: dict[str, float] = field(default_factory=dict)
    """Optional per-turn component reward, provided by the env and used by the rubric.
    `None` until scored."""


@dataclass(kw_only=True, slots=True)
class Rollout:
    """A complete rollout: ordered turns + terminal state + reward + identifier."""

    group_id: str
    """ID for the prompt group used for advantage centering."""

    sample_idx: int
    """Sample index within the group (0..group_size-1)."""

    turns: list[RolloutTurn] = field(default_factory=list)  # [K_turns]
    """Ordered rollout turns."""

    status: RolloutStatus = RolloutStatus.COMPLETED
    """Rollout-level terminal status."""

    reward: float | None = None
    """Final weighted reward, filled by the rubric."""

    reward_components: dict[str, float] = field(default_factory=dict)
    """Decomposed rewards, filled by the rubric."""

    # TODO: make it per token
    advantage: float | None = None
    """Advantage for this sample."""


@dataclass(frozen=True, kw_only=True, slots=True)
class DatasetOutput:
    """One row from a dataset, ready for env construction."""

    task: str
    """Task identifier the controller uses to dispatch this row to the right
    `Task` in `RLTrainer`."""

    env_input: object
    """Task-specific payload consumed by `Task.make_envs`. Each task defines
    its own typed payload."""
