# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Environment contract.

The rollout loop talks to environments via three small types and one
``Protocol``. Each env instance is **single-use**: the controller
constructs one per rollout, drives it through ``reset → step → step → …``,
and discards it.

Building envs from dataset rows is the ``EnvBuilder`` / ``EnvDataset``
side: ``EnvDataset.sample_groups`` returns ``EnvExample`` rows, and
``EnvBuilder.make_envs(example, group_size=G)`` instantiates a group of
``G`` sibling envs for GRPO group-mean centering.

Token-level concerns (parse failure, length-stop, context overflow,
step timeout) live in ``envs.token_env.TokenEnv`` — the env author
writes message-level game logic and the adapter wraps it for the
rollout driver.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from renderers import Message, ToolSpec

from torchtitan.experiments.rl.types import JsonValue, RolloutStatus

__all__ = [
    "EnvBuilder",
    "EnvDataset",
    "EnvExample",
    "EnvReset",
    "EnvStep",
    "MessageEnv",
]


# ---------------------------------------------------------------------------
# Env-side protocol
# ---------------------------------------------------------------------------


@dataclass(kw_only=True, slots=True)
class EnvReset:
    """Initial observation from ``MessageEnv.reset``.

    Example::

        EnvReset(messages=[
            {"role": "system", "content": "Solve digit-sum problems."},
            {"role": "user",   "content": "What is the digit sum of [12, 34]?"},
        ])
    """

    messages: list[Message]
    tools: list[ToolSpec] = field(default_factory=list)


@dataclass(kw_only=True, slots=True)
class EnvStep:
    """Env reply to one assistant message.

    Terminal steps stamp ``reward`` and set ``done=True``.
    Non-terminal steps return follow-up ``messages`` to append to the
    running conversation.

    ``status`` is the *env-side* terminal status; ``TokenEnv``
    reconciles it with the generator's ``finish_reason`` (e.g.
    ``"length"``) to produce the per-turn :class:`RolloutStatus`.

    Example (terminal)::

        EnvStep(reward=1.0, reward_components={"correct": 1.0},
                done=True, status=RolloutStatus.COMPLETED)

    Example (non-terminal, with follow-up)::

        EnvStep(messages=[{"role": "user", "content": "and now add: 7, 8"}])
    """

    messages: list[Message] = field(default_factory=list)
    reward: float | None = None
    reward_components: dict[str, float] = field(default_factory=dict)
    done: bool = False
    status: RolloutStatus | None = None


@runtime_checkable
class MessageEnv(Protocol):
    """Canonical env protocol. One instance per rollout (single-use).

    All three methods are ``async`` because real envs (sandboxed Python,
    browser sessions, tool servers, retrievers) need to ``await`` on
    construction or step. Pure-CPU envs implement them as ``async def``
    that just returns synchronously — the overhead is negligible and the
    uniform shape lets the adapter wrap any env in ``asyncio.wait_for``
    for per-step timeouts.

    Single-use: the constructor seeds the task; ``reset()`` returns
    the initial observation. Calling ``reset()`` twice is undefined
    behaviour — start a fresh env instead.
    """

    async def reset(self) -> EnvReset:
        ...

    async def step(self, assistant_message: Message) -> EnvStep:
        ...

    async def close(self) -> None:
        ...


# ---------------------------------------------------------------------------
# Data-plane protocol (dataset + builder)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EnvExample:
    """One row of an RL dataset.

    ``payload`` is the row contents the env builder reads. ``task_id``
    is a stable identifier used as the GRPO group ID
    (``RolloutOutput.group_id``); a deterministic ``f"{task}/{step}/{i}"``
    string is the canonical form.

    ``tags`` are short strings the controller uses to route mixed-task
    rows to the right builder (e.g. ``("sum_digits",)`` /
    ``("alphabet_sort",)``). Single-task runs leave them empty.
    """

    task_id: str
    payload: dict[str, JsonValue]
    tags: tuple[str, ...] = ()


class EnvBuilder(Protocol):
    """Builds a group of single-use envs from one dataset row.

    The same row spawns ``group_size`` sibling envs — same task,
    independent sampler seeds — so GRPO can center advantages across
    the group.

    ``make_envs`` is the only required method. Builders that need to
    amortize per-row setup (e.g. loading a verifier model, fetching a
    sandbox image) can do so before constructing the N envs; cheap
    builders ignore the amortization.
    """

    async def make_envs(
        self, example: EnvExample, *, group_size: int
    ) -> Sequence[MessageEnv]:
        ...


class EnvDataset(Protocol):
    """Lazy iterable of ``EnvExample``s.

    ``sample_groups`` is called once per controller-side ``rollout`` batch
    and returns ``num_groups`` rows. The dataset owns sampling policy
    (shuffle, sequential, weighted-mixture); the controller iterates
    rows and dispatches each through its corresponding builder.

    Deterministic datasets seed on ``(step, group_idx)`` so a re-run
    with the same config produces the same rollouts — invaluable for
    debugging.
    """

    def sample_groups(self, *, step: int, num_groups: int) -> Sequence[EnvExample]:
        ...
