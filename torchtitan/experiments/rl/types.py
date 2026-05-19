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
    """One vLLM sample exposed via the standalone ``generate.py`` CLI.

    Not used by the rollout driver; the per-rollout path uses the
    slimmer :class:`actors.generator.GenerateOutput`.
    """

    policy_version: int
    prompt_idx: int
    text: str
    token_ids: list[int]
    token_logprobs: list[float]
    finish_reason: str | None = None


# ---------------------------------------------------------------------------
# Rollout carriers (consumed by the rollout driver + replay-sample builder)
# ---------------------------------------------------------------------------


@dataclass(kw_only=True, slots=True)
class RolloutTurn:
    """One generator call + the env response to that call.

    The terminal ``status`` / ``reward`` / ``reward_components`` live on
    the enclosing :class:`RolloutOutput`; per-turn fields here are
    exactly what the replay builder needs to emit one or more
    :class:`ReplaySample`\\ s.

    ``policy_version`` is the weight version that produced this turn's
    response tokens. ``RolloutOutput.behavior_version`` takes the
    conservative ``min`` across turns when a rollout spans a swap.
    """

    prompt_token_ids: list[int]  # [prompt_tokens]
    response_token_ids: list[int]  # [response_tokens]
    response_logprobs: list[float]  # [response_tokens]
    policy_version: int
    """Weight version that produced ``response_token_ids``."""

    prompt_messages: list["Message"] = field(default_factory=list)  # [prompt_messages]
    response_messages: list["Message"] = field(
        default_factory=list
    )  # [response_messages]


@dataclass(kw_only=True, slots=True)
class RolloutOutput:
    """One rollout: ordered turns + terminal reward + group identity.

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
    """Per-component breakdown of ``reward`` (env-specific keys; e.g.
    ``{"correctness": 1.0, "format": 0.5}`` for SumDigits)."""

    @property
    def behavior_version(self) -> int:
        """Min ``policy_version`` across turns — the conservative stamp
        used by the replay buffer's age policy. Rollouts that span a
        weight swap are treated as stale on the older of the two
        versions.
        """
        return min((t.policy_version for t in self.turns), default=-1)


def validate_rollout_output(o: RolloutOutput) -> None:
    """Cheap structural check; raises ``ValueError`` on shape mismatch.

    Allows a missing ``reward`` only on ``status=ERROR`` (a parse or
    timeout failure inside the adapter); a terminal ``COMPLETED`` /
    ``TRUNCATED`` rollout must stamp it.
    """
    for i, t in enumerate(o.turns):
        if len(t.response_token_ids) != len(t.response_logprobs):
            raise ValueError(
                f"turn {i}: response_token_ids [{len(t.response_token_ids)}] != "
                f"response_logprobs [{len(t.response_logprobs)}]"
            )
    if o.turns and o.reward is None and o.status != RolloutStatus.ERROR:
        raise ValueError(
            f"rollout {o.group_id!r}/{o.sample_idx} has turns but no reward; "
            "terminal EnvStep must stamp ``reward`` (unless status=ERROR)."
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
