# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, TYPE_CHECKING

import numpy as np
from renderers import Message

from torchtitan.rl.observability import metrics as m

if TYPE_CHECKING:
    # Type-only: importing the generator module here would pull in vLLM at import time.
    from torchtitan.rl.generator.types import SamplingConfig


_TRUNCATED = frozenset(
    {"truncated_length", "truncated_prompt_too_long", "truncated_max_turns"}
)
_ERROR = frozenset(
    {"error_parse", "error_timeout", "error_abort", "error", "error_system"}
)


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
    """The id `Generator.generate` received. Over HTTP, that is the caller's id plus the server's
    `/http=<n>` suffix."""
    token_ids: list[int]
    token_logprobs: list[float]
    topk_token_ids: np.ndarray | None = None
    """[num_tokens, k] int32 ids of the generator's k most likely tokens at each position;
    None unless `SamplingConfig.num_topk_logprobs` > 0."""
    topk_logprobs: np.ndarray | None = None
    """[num_tokens, k] float32 generator logprobs of `topk_token_ids`."""
    finish_reason: str | None = None
    """vLLM `CompletionOutput.finish_reason` ("stop" | "length" | "abort")"""

    routed_expert_ids: np.ndarray | None = None
    """Router replay: the experts each forward input went to, ``[num_rows, num_layers, top_k]``,
    uint8 (uint16 above 256 experts). Rows start at `SamplingConfig.routed_experts_prompt_start`;
    the last token was never fed back, so it has no row:

        prompt | completion   p0 p1 p2 | c0 c1    routed_experts_prompt_start=1
        rows                     r1 r2 | r3       # no row for c1
    """

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


class ReleaseSessionFn(Protocol):
    """Tell the generator side that a routing session makes no more calls.

    The Rollouter calls this once per rollout, after its last turn, so a router can drop the
    session's affinity.
    """

    async def __call__(self, *, routing_session_id: str) -> None:
        """Release one routing session.

        Args:
            routing_session_id: The `routing_session_id` the rollout passed to `GenerateFn`.
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
    # The rollout failed outside the policy: its code raised, or Verifiers recorded an infra error
    # (a generator 5xx, a sandbox or env failure). It stays in its group for logs, but gets no
    # reward or advantage and isn't trained. A rollout cut as a straggler (see `gather_rollouts`)
    # gets it too, with `logs["straggler_cut"]`.
    ERROR_SYSTEM = "error_system"

    def is_truncated(self) -> bool:
        return self.value in _TRUNCATED

    def is_error(self) -> bool:
        return self.value in _ERROR

    def is_terminal(self) -> bool:
        return self is not RolloutStatus.ONGOING


@dataclass(kw_only=True, slots=True)
class RolloutTurn:
    """One turn: the prompt fed to the generator + the sampled completion +
    the env's reply, in both token and message space. Rubrics score it and
    `rollout_to_training_samples` packs the rollout into training tokens.

    A turn doesn't store its full prompt. It stores how many leading tokens it copies from the previous
    turn's prompt + completion, then its new tokens (`prompt_messages` stay in full):

        full prompt = (previous prompt + completion)[:prompt_prefix_len] + prompt_delta_token_ids

    Example: the previous turn's prompt is [0, 1, 2] and its completion [3].

        this turn's prompt   prompt_prefix_len  prompt_delta_token_ids
        [0, 1, 2, 3, 4, 5]   4                  [4, 5]                  # continues
        [0, 1, 9, 3]         2                  [9, 3]                  # edited 2 -> 9: 3 is stored again

    Per-token fields are numpy arrays, not lists: a group pickles as raw bytes from the rollout worker
    to the controller, and the GC doesn't track them.
    """

    # TODO: add a `logs` field (raw prompt/response text, finish_reason, timings)
    # so a turn can be dumped and inspected without re-deriving from tokens.

    rollout_id: RolloutTurnID
    """Identifies this turn (group, sibling index, turn index)."""

    # Fields needed for training
    prompt_prefix_len: int
    """Length of the longest prefix shared with the previous turn's prompt + completion; set by `split_prompt`."""

    prompt_delta_token_ids: np.ndarray  # [num_prompt_tokens - prompt_prefix_len] int32
    """The prompt after its first `prompt_prefix_len` tokens."""

    completion_token_ids: np.ndarray  # [num_completion_tokens] int32
    """This turn's completion token ids."""

    completion_logprobs: np.ndarray  # [num_completion_tokens] float32
    """This turn's completion token logprobs generated by the generator policy."""

    routed_expert_ids: np.ndarray | None = None
    """This turn's new rows of `Completion.routed_expert_ids`, a delta like the prompt. They start
    at the prefix's last token, which only this turn's prefill ran:

        previous turn   p0 p1 | c0 c1          rows r0 r1 r2
        this turn       p0 p1 c0 c1 e0 | c2    prompt_prefix_len=4 -> rows r3 r4 (r3: c1)
    """
    completion_topk_token_ids: np.ndarray | None = None  # [num_completion_tokens, k]
    """The generator's top-k int32 token ids at each completion position (`Completion.topk_token_ids`)."""

    completion_topk_logprobs: np.ndarray | None = None  # [num_completion_tokens, k]
    """The generator's float32 logprobs of `completion_topk_token_ids`."""

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
) -> tuple[int, np.ndarray]:
    """Split a prompt into the part shared with the previous turn and the new part.

    Args:
        prompt_token_ids: this turn's full prompt.
        previous_token_ids: the previous turn's prompt + completion; [] on the first turn.

    Returns:
        `(prompt_prefix_len, prompt_delta_token_ids)`: the length of the longest shared prefix, and the
        prompt after it.

    Example:
        split_prompt([0, 1, 2, 3, 4, 5], previous_token_ids=[0, 1, 2, 3])  # -> (4, array([4, 5])): continues
        split_prompt([0, 1, 9, 3], previous_token_ids=[0, 1, 2, 3])        # -> (2, array([9, 3])): history edited
        split_prompt([0, 1, 2], previous_token_ids=[])                     # -> (0, array([0, 1, 2])): first turn
    """
    # Common case: the prompt extends the previous turn. One list compare, no Python loop.
    if prompt_token_ids[: len(previous_token_ids)] == previous_token_ids:
        prefix_len = len(previous_token_ids)
    else:
        prefix_len = 0
        for token, previous in zip(prompt_token_ids, previous_token_ids):
            if token != previous:
                break
            prefix_len += 1
    return prefix_len, np.asarray(prompt_token_ids[prefix_len:], dtype=np.int32)


@dataclass(kw_only=True, slots=True)
class Rollout:
    """A complete rollout: ordered turns + terminal state + reward + identifier."""

    group_id: int
    """Prompt-group ID; siblings share it for advantage centering."""

    rollout_id: int
    """Sibling index within the group (0..group_size-1)."""

    turns: list[RolloutTurn] = field(default_factory=list)  # [num_turns]
    """Ordered rollout turns; each stores its prompt as a delta on the previous turn (see `RolloutTurn`)."""
    # TODO: let a turn be a delta on any earlier turn, not just the previous one, so a turn that resumes
    # an older turn (a retry, or a main agent after a subagent) stores only its new tokens.

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
