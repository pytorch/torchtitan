# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Core types for the RL rollout / training pipeline.

Three layers (top to bottom in trainer-data flow):

1. **Env contract** (``envs/types.py``): the message-level interaction
   between rollout driver and user-supplied environments.
2. **Rollout carrier** (``RolloutTurn``, ``RolloutOutput``, ``RolloutStatus``):
   what one rollout looks like from the controller. Ordered turns +
   terminal status + group identity + final reward.
3. **Trainer carrier** (``ReplaySample``, ``TrainBatch``): per-token
   sequences with ``loss_mask`` ready for the trainer's GRPO loss.

``Completion`` survives at the generator boundary (one sample from
vLLM); everything trainer-facing flows through ``ReplaySample``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, TypeAlias

import torch

if TYPE_CHECKING:
    from renderers import Message


# JSON-only metadata, type-only enforced. Forbids ``Callable`` /
# ``torch.Tensor`` / live handles inside env metadata so a future
# remote env proxy can round-trip ``EnvReset.metadata`` / ``EnvStep.metadata``
# without re-typing the protocol.
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class RolloutStatus(StrEnum):
    """Terminal status for a single turn or a whole rollout."""

    COMPLETED = "completed"
    """Env signalled ``done=True`` on a real stop token."""

    TRUNCATED = "truncated"
    """Generator hit ``max_tokens`` or the env capped turns before completion."""

    ERROR = "error"
    """Env or renderer raised; reward should be treated as undefined."""


# ---------------------------------------------------------------------------
# Generator boundary
# ---------------------------------------------------------------------------


@dataclass(kw_only=True, slots=True)
class Completion:
    """Pure generator artifact — one vLLM sample.

    Used by the legacy single-shot ``generate(tokenized_prompts)`` endpoint.
    The submit-queue path (``generate_tokens(request)``) returns the
    lighter ``GenerateOutput`` shape; both carry the same per-token
    logprobs.
    """

    policy_version: int
    prompt_idx: int
    text: str
    token_ids: list[int]
    token_logprobs: list[float]
    finish_reason: str | None = None
    """vLLM ``CompletionOutput.finish_reason`` — ``"stop" | "length" | "abort"``."""


# ---------------------------------------------------------------------------
# Rollout carriers (consumed by the rollout driver + replay-sample builder)
# ---------------------------------------------------------------------------


@dataclass(kw_only=True, slots=True)
class RolloutTurn:
    """One generator call + the env response to that call.

    ``policy_version`` is stamped at generation time; a rollout that
    spans a weight swap surfaces as turns with different versions, and
    the replay-sample builder takes the conservative ``min`` so the
    buffer's age policy can drop the whole sample.

    ``finish_reason`` mirrors vLLM's ``"stop" | "length"``; the rollout
    driver does NOT branch on it (the ``TokenEnv`` adapter owns the
    length-stop terminal case).
    """

    prompt_token_ids: list[int]  # [prompt_tokens]
    response_token_ids: list[int]  # [response_tokens]
    response_logprobs: list[float]  # [response_tokens]
    prompt_messages: list["Message"] = field(default_factory=list)  # [prompt_messages]
    response_messages: list["Message"] = field(
        default_factory=list
    )  # [response_messages]
    status: RolloutStatus = RolloutStatus.COMPLETED
    reward_components: dict[str, float] = field(default_factory=dict)
    policy_version: int = 0
    finish_reason: str | None = None


@dataclass(kw_only=True, slots=True)
class RolloutOutput:
    """One rollout: ordered turns + terminal reward + group identity.

    Pure rollout artifact — immutable after construction. Advantages
    are computed at collate time from groups of rollouts and live on
    ``ReplaySample``.

    Shape legend:
        K: number of turns.

    Example::

        RolloutOutput(
            group_id="sum_digits/step42/group3",
            sample_idx=0,
            turns=[<RolloutTurn>, <RolloutTurn>],
            status=RolloutStatus.COMPLETED,
            reward=0.85,
            reward_components={"correctness": 1.0, "format": 0.7},
        )
    """

    group_id: str
    sample_idx: int
    turns: list[RolloutTurn] = field(default_factory=list)  # [K]
    status: RolloutStatus = RolloutStatus.COMPLETED
    reward: float | None = None
    reward_components: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, JsonValue] = field(default_factory=dict)


def validate_rollout_output(o: RolloutOutput) -> None:
    """Cheap structural check. Raises ``ValueError`` on misalignment that
    would otherwise surface as NaN/Inf inside the trainer.

    Called by ``do_single_rollout`` at the end of each rollout; consumers
    may call it again before training to defend against in-memory
    corruption.
    """
    for i, t in enumerate(o.turns):
        if len(t.response_token_ids) != len(t.response_logprobs):
            raise ValueError(
                f"turn {i}: response_token_ids [{len(t.response_token_ids)}] != "
                f"response_logprobs [{len(t.response_logprobs)}]"
            )
    if o.turns and o.reward is None:
        raise ValueError(
            f"rollout {o.group_id!r}/{o.sample_idx} has turns but no reward; "
            "terminal EnvStep must stamp ``reward``."
        )


# ---------------------------------------------------------------------------
# Trainer carriers
# ---------------------------------------------------------------------------


@dataclass(kw_only=True, slots=True)
class ReplaySample:
    """One contiguous run-of-tokens the trainer can forward through.

    A single-turn rollout produces one ``ReplaySample``. A multi-turn
    rollout produces one ``ReplaySample`` per contiguous prefix-runnable
    span — see ``rollout_to_replay_samples`` in ``replay.py``. Prefix
    continuity is detected by comparing each turn's ``prompt_tokens``
    against the running buffer; a mismatch flushes the current sample
    and starts a new one.

    ``loss_mask = 1`` exactly on the response (assistant-sampled) tokens.
    Prompt and env-emitted tokens carry ``loss_mask = 0``.
    ``behavior_logprobs`` are the per-token logprobs the sampler reported
    at generation time; they're ``0.0`` on prompt positions (never read
    because of the mask).

    ``advantage`` is scalar (group-mean baseline, computed by
    ``compute_advantages``). It's broadcast to per-token by the trainer at
    collate time and zeroed by ``loss_mask`` inside the loss.

    Shape legend:
        T: total tokens in this sample.

    Example (single-turn, 50 prompt + 30 response)::

        ReplaySample(
            token_ids=[101, ..., 256],
            loss_mask=[0]*50 + [1]*30,
            behavior_logprobs=[0.0]*50 + [-0.42, -1.1, ...],
            advantage=0.27,
            group_id="sum_digits/step42/group3",
            sample_idx=0,
            policy_version=3,
            reward=0.85,
            reward_components={"correctness": 1.0, "format": 0.7},
        )
    """

    token_ids: list[int]  # [T]
    loss_mask: list[int]  # [T] (0/1)
    behavior_logprobs: list[float]  # [T] (0.0 on prompt)
    advantage: float
    group_id: str
    sample_idx: int
    policy_version: int
    reward: float
    reward_components: dict[str, float] = field(default_factory=dict)


@dataclass(kw_only=True, slots=True)
class TrainBatch:
    """Varlen-packed batch consumed by ``PolicyTrainer.forward_backward``.

    All trainer-side fields are pre-shifted to align with the model's
    next-token logprobs (``log_softmax(logits[:, :-1]).gather(token_ids[:, 1:])``,
    which produces ``[1, T_total - 1]``). The shift is a single
    ``[..., 1:]`` slice applied at collate time:

    - ``loss_mask[t]`` = ``unshifted_mask[t + 1]`` (= mask of the
      *predicted* token at shifted position ``t``).
    - ``behavior_logprobs[t]`` = ``unshifted_logprobs[t + 1]``.
    - ``advantages_per_token[t]`` = advantage of the sample containing
      the predicted token at original position ``t + 1``.

    Cross-sample-boundary positions naturally end up with
    ``loss_mask = 0`` because the predicted token (first token of the
    next sample) is a prompt position. The trainer doesn't have to
    branch on boundaries.

    Shape legend:
        B: number of ``ReplaySample`` in this DP-rank batch.
        T_total: ``sum(len(sample.token_ids))`` across the batch.
    """

    token_ids: torch.Tensor  # [1, T_total]
    seq_lens: list[int]  # [B]
    loss_mask: torch.Tensor  # [1, T_total - 1] (shifted)
    behavior_logprobs: torch.Tensor  # [1, T_total - 1] (shifted)
    advantages_per_token: torch.Tensor  # [1, T_total - 1] (shifted)


@dataclass(frozen=True, slots=True)
class OptimStepOutput:
    """Result returned by ``PolicyTrainer.optim_step`` to the controller."""

    policy_version: int
    metrics: dict[str, float]
