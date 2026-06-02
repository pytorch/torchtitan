# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from renderers import Message


_TRUNCATED = frozenset({"truncated_length", "truncated_prompt_too_long"})
_ERROR = frozenset({"error_parse", "error_timeout", "error_abort", "error"})


class RolloutStatus(StrEnum):
    """Per-rollout status."""

    ONGOING = "ongoing"
    COMPLETED = "completed"
    TRUNCATED_LENGTH = "truncated_length"
    TRUNCATED_PROMPT_TOO_LONG = "truncated_prompt_too_long"
    ERROR_PARSE = "error_parse"
    ERROR_TIMEOUT = "error_timeout"
    ERROR_ABORT = "error_abort"
    ERROR = "error"

    def is_truncated(self) -> bool:
        return self.value in _TRUNCATED

    def is_error(self) -> bool:
        return self.value in _ERROR

    def is_terminal(self) -> bool:
        return self is not RolloutStatus.ONGOING


@dataclass(kw_only=True, slots=True)
class RolloutTurn:
    """One generator completion + the env response to that completion."""

    # TODO: add a `logs` field (raw prompt/response text, finish_reason, timings)
    # so a turn can be dumped and inspected without re-deriving from tokens.

    # Fields needed for training
    prompt_token_ids: list[int]  # [L_prompt]
    """Tokenized conversation up to this turn, used to generate the assistant response."""

    assistant_token_ids: list[int]  # [L_response]
    """Tokens the assistant produced this turn."""

    assistant_logprobs: list[float]  # [L_response]
    """Per-token logprobs from the generator policy for the assistant tokens."""

    # Filtering / debugging
    policy_version: int
    """Trainer policy version when this response was sampled."""

    # Logging
    prompt_messages: list[Message] = field(default_factory=list)  # [M_prompt]
    """Full conversation up to this turn, used to generate the assistant response."""

    assistant_message: Message | None = None
    """The assistant's message (generator output, parsed)."""

    env_messages: list[Message] = field(default_factory=list)  # [M_env]
    """The env's reply messages this turn (tool / user)."""

    # For rubrics
    env_rewards: dict[str, float] = field(default_factory=dict)
    """Optional per-turn reward signals the env attached; the rubric decides how to use them."""


@dataclass(kw_only=True, slots=True)
class Rollout:
    """A complete rollout: ordered turns + terminal state + reward + identifier."""

    # TODO: add a `logs` field (per-turn debug records / event trace) to make a
    # full rollout reconstructable for debugging.

    group_id: str
    """Prompt-group ID; siblings share it for advantage centering."""

    sample_idx: int
    """Sample index within the group (0..group_size-1)."""

    turns: list[RolloutTurn] = field(default_factory=list)  # [K_turns]
    """Ordered rollout turns."""

    status: RolloutStatus = RolloutStatus.COMPLETED
    """Rollout-level terminal status."""

    reward: float | None = None
    """Final weighted reward, filled by the rubric."""

    reward_breakdown: dict[str, float] = field(default_factory=dict)
    """Raw per-reward-function values, filled by the rubric."""

    # TODO: make it per token
    advantage: float | None = None
    """Advantage for this sample."""


@dataclass(kw_only=True, slots=True)
class RolloutGroup:
    group_id: str
    """Prompt-group ID; siblings share it for advantage centering."""

    env_input: object
    """The env input (dataset payload) shared by the group; passed to the rubric."""

    rollouts: list[Rollout]  # [group_size]
    """Sibling rollouts sampled from the group's shared prompt."""
