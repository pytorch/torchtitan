# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, TYPE_CHECKING

from renderers import Message

from torchtitan.rl.observability import metrics as m

if TYPE_CHECKING:
    # Type-only: importing the generator module here would pull in vLLM at import time.
    from torchtitan.rl.generator import SamplingConfig


_TRUNCATED = frozenset(
    {"truncated_length", "truncated_prompt_too_long", "truncated_max_turns"}
)
_ERROR = frozenset({"error_parse", "error_timeout", "error_abort", "error"})


@dataclass(frozen=True, slots=True)
class RolloutTurnID:
    """A turn's id: (group, sibling rollout, turn index); renders to the generator request_id.

    Example:

        RolloutTurnID(group_id=5, rollout_id=2, turn_id=0).to_string()
        # -> "group=5/rollout=2/turn=0"
        RolloutTurnID(group_id=5, rollout_id=2, turn_id=0).to_string(include_turn=False)
        # -> "group=5/rollout=2"
    """

    group_id: int
    """Globally-unique GRPO group id; siblings share it (the sticky-routing key, sans turn)."""
    rollout_id: int
    """Sibling index within the group (0..group_size-1)."""
    turn_id: int
    """Turn index within the rollout; for a TrainingSample, the turn where begins.
    This is not 0 when a single rollout is split into multiple training samples."""

    def to_string(self, *, include_turn: bool = True) -> str:
        base = f"group={self.group_id}/rollout={self.rollout_id}"
        return f"{base}/turn={self.turn_id}" if include_turn else base


@dataclass(kw_only=True, slots=True)
class Completion:
    """A single generated sequence from the generator.

    Example:

        Completion(min_policy_version=7, max_policy_version=7, request_id="r0", token_ids=[12, 9],
                   token_logprobs=[-0.2, -1.1], finish_reason="stop",
                   metrics=[Metric("generator/queue_time_ms", ...)])
    """

    min_policy_version: int
    """Oldest policy version among this turn's decode."""
    max_policy_version: int
    """Newest policy version among this turn's decode."""
    # TODO(async-rl): for exact per-token version attribution, switch the engine to
    #   RequestOutputKind.CUMULATIVE and record (start_token, version) boundaries; today we keep only
    #   the per-turn min (min_policy_version) / max (max_policy_version).
    request_id: str
    """Echoes the id the caller passed to `generate`, so callers can validate
    ordered completions or map by id."""
    token_ids: list[int]
    token_logprobs: list[float]
    finish_reason: str | None = None
    """vLLM `CompletionOutput.finish_reason` ("stop" | "length" | "abort")"""

    metrics: list[m.Metric] = field(default_factory=list)
    """Per-generation metrics measured by the generator (latencies); the
    controller attaches them to the rollout turn."""


class GenerateFn(Protocol):
    """Generate one model completion for a prompt.

    The Rollouter calls this once per turn and gets back a `Completion`. It does not need to know
    how or where generation runs. It can be a monarch actor, a router, an http endpoint, etc.
    """

    async def __call__(
        self,
        prompt_token_ids: list[int],
        *,
        request_id: str,
        group_id: int,
        routing_session_id: str | None = None,
        sampling_config: SamplingConfig | None = None,
    ) -> Completion | None:
        """Run one generation.

        Args:
            prompt_token_ids: The tokenized prompt to generate from.
            request_id: Unique per call; identifies the exact turn (e.g. ".../turn=2") in logs.
            group_id: Rollout group this call belongs to. Calls of one group share a prefix
                cache namespace, so siblings and later turns can reuse the group's KV.
            routing_session_id: Optional stable key for the routing session this call
                belongs to. A router may use it for session affinity, routing same-key
                calls to the same generator when possible. `None` means no affinity.
            sampling_config: Optional per-call sampling overrides.

        Returns:
            The generated `Completion`, or `None` if none is produced.
        """


class RolloutStatus(StrEnum):
    """Per-rollout status."""

    ONGOING = "ongoing"
    COMPLETED = "completed"
    TRUNCATED_LENGTH = "truncated_length"
    TRUNCATED_PROMPT_TOO_LONG = "truncated_prompt_too_long"
    TRUNCATED_MAX_TURNS = "truncated_max_turns"
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
    """One turn: the prompt it generated from (tokens as a delta on the previous turn, messages in
    full), the sampled completion and the env's reply. Rubrics score it and
    `rollout_to_training_samples` packs the rollout into training tokens.

    Example:
        # turn 0, prompt [P]:          prompt_prefix_len=0,           prompt_delta_token_ids=P
        # turn 1, prompt [P, C0, E0]:  prompt_prefix_len=len(P + C0), prompt_delta_token_ids=E0
    """

    # TODO: add a `logs` field (raw prompt/response text, finish_reason, timings)
    # so a turn can be dumped and inspected without re-deriving from tokens.

    rollout_id: RolloutTurnID
    """Identifies this turn (group, sibling index, turn index)."""

    # Fields needed for training
    prompt_prefix_len: int
    """Length of the longest prefix the prompt shares with the previous turn's prompt + completion;
    0 on the first turn. Compute it with `split_prompt`; the builder rejects any other value."""

    prompt_delta_token_ids: list[int]  # [num_prompt_tokens - prompt_prefix_len]
    """The rest of the prompt: `prompt == previous_turn_tokens[:prompt_prefix_len] + prompt_delta_token_ids`.
    Storing only these keeps a rollout's size linear in turns."""

    completion_token_ids: list[int]  # [num_completion_tokens]
    """This turn's completion token ids."""

    completion_logprobs: list[float]  # [num_completion_tokens]
    """This turn's completion token logprobs generated by the generator policy."""

    # Filtering
    min_policy_version: int | None = None
    """Oldest policy version this turn was sampled under; `None` if no generation happened."""

    max_policy_version: int | None = None
    """Newest policy version this turn was sampled under; `None` if no generation happened."""

    # Logging
    prompt_messages: list[Message] = field(
        default_factory=list
    )  # [num_prompt_messages]
    """Full conversation up to this turn, not a delta."""

    completion_message: Message | None = None
    """This turn's completion decoded into a message by the renderer (the TokenEnv's parse,
    not the raw generator output)."""

    env_messages: list[Message] = field(default_factory=list)  # [num_env_messages]
    """This turn env's reply messages (tool / user)."""

    # For rubrics
    env_rewards: dict[str, float] = field(default_factory=dict)
    """This turn optional reward signals the env attached; the rubric decides how to use them."""

    metrics: list[m.Metric] = field(default_factory=list)
    """Per-turn metrics produced during rollouts"""

    @property
    def num_prompt_tokens(self) -> int:
        """Length of the full prompt this turn generated from."""
        return self.prompt_prefix_len + len(self.prompt_delta_token_ids)


def split_prompt(
    prompt_token_ids: list[int], previous_token_ids: list[int]
) -> tuple[int, list[int]]:
    """Return `(prompt_prefix_len, prompt_delta_token_ids)`: how many leading tokens the prompt shares
    with the previous turn's prompt + completion, and the rest.

    Example:
        split_prompt([1, 2, 4, 5, 9], previous_token_ids=[1, 2, 4, 5])  # -> (4, [9]): continues
        split_prompt([1, 2, 7], previous_token_ids=[1, 2, 4, 5])        # -> (2, [7]): history rewritten
        split_prompt([1, 2], previous_token_ids=[])                     # -> (0, [1, 2]): first turn
    """
    # Fast path, one C-level list compare: the prompt continues the previous turn (the common case)
    if prompt_token_ids[: len(previous_token_ids)] == previous_token_ids:
        return len(previous_token_ids), prompt_token_ids[len(previous_token_ids) :]
    prefix_len = 0
    for token, previous in zip(prompt_token_ids, previous_token_ids):
        if token != previous:
            break
        prefix_len += 1
    return prefix_len, prompt_token_ids[prefix_len:]


@dataclass(kw_only=True, slots=True)
class Rollout:
    """A complete rollout: ordered turns + terminal state + reward + identifier."""

    group_id: int
    """Prompt-group ID; siblings share it for advantage centering."""

    rollout_id: int
    """Sibling index within the group (0..group_size-1)."""

    turns: list[RolloutTurn] = field(default_factory=list)  # [num_turns]
    """Ordered rollout turns. Each turn stores only the prompt tokens it adds to the previous turn
    (`RolloutTurn.prompt_delta_token_ids`); training_sample assembly concatenates them."""
    # TODO: a message graph, as in verifiers v1, would also share the first prompt across siblings
    # (<1% of a 120-ply chess group) and let a turn fork from an older turn (subagents, tree search).

    status: RolloutStatus
    """Rollout-level terminal status."""

    reward: float | None = None
    """Final weighted reward, filled by the rubric."""

    reward_breakdown: dict[str, float] = field(default_factory=dict)
    """Raw per-reward-function values, filled by the rubric."""

    # TODO: make it per token
    advantage: float | None = None
    """Advantage for this sample."""

    logs: dict[str, Any] = field(default_factory=dict)
    """Rollouter-specific debug record, e.g. why the rollout failed. Saved with the rollout
    by `RolloutSampleRecorder`; not used for training."""


@dataclass(kw_only=True, slots=True)
class RolloutGroup:
    group_id: int
    """Prompt-group ID; siblings share it for advantage centering."""

    rollouts: list[Rollout]  # [group_size]
    """Sibling rollouts sampled from the group's shared prompt. Empty on a failed generation."""

    metrics: list[m.Metric] = field(default_factory=list)
    """Rollout-origin metrics that ride with this group to the trainer (a failed group carries its failure metric)."""
