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
        return self.value in _TRUNCATED

    def is_error(self) -> bool:
        return self.value in _ERROR


@dataclass(kw_only=True, slots=True)
class RolloutTurn:
    """One generator call + the env response to that call."""

    prompt_token_ids: list[int]  # [L_prompt]
    """Token ids the generator saw as prompt for this turn."""

    response_token_ids: list[int]  # [L_response]
    """Token ids the generator produced (response only)."""

    response_logprobs: list[float]  # [L_response]
    """Per-token logprobs from the sampling policy."""

    policy_version: int
    """Trainer policy version when this response was sampled."""

    response_messages: list[Message] = field(default_factory=list)  # [M_response]
    """Parsed assistant + env-appended messages (assistant first)."""


@dataclass(kw_only=True, slots=True)
class Rollout:
    """A complete rollout: ordered turns + terminal state + reward + identity."""

    group_id: str
    """Stable ID for the prompt group used for advantage centering."""

    sample_idx: int
    """Sibling index within the group (0..group_size-1)."""

    turns: list[RolloutTurn] = field(default_factory=list)  # [K]
    """Ordered rollout turns."""

    status: RolloutStatus = RolloutStatus.COMPLETED
    """Rollout-level terminal status."""

    reward: float | None = None
    """Final scalar reward, filled by the rubric."""

    reward_components: dict[str, float] = field(default_factory=dict)
    """Decomposed reward metrics, filled by the rubric."""

    advantage: float | None = None
    """GRPO advantage (reward - group_mean), filled by the controller."""


@dataclass(frozen=True, kw_only=True, slots=True)
class DatasetOutput:
    """One row from a task's dataset, ready for env construction."""

    env_name: str
    """Task identifier used for multi-task dispatch.
    For single-task tasks this is a constant."""

    env_input: object
    """Task-specific payload consumed by ``Task.make_envs``.
    Each task defines its own typed payload; multi-task tasks
    downcast by ``env_name``."""
