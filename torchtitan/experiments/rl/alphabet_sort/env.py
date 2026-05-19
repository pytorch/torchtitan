# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AlphabetSortEnv — multi-turn name-sorting MessageEnv.

Each episode introduces ``num_turns`` batches of CamelCase names; the
model must respond with the *cumulative* sorted list at each turn,
marking the newly-introduced names with ``// new name!``. Reward is
the sequence-similarity score (Ratcliff/Obershelp) of the model's
extracted XML tag content against the expected sorted prefix,
aggregated across turns per ``power_per_turn``.

Reproduces prime-rl's ``examples/alphabet_sort`` env (in turn ported
from verifiers' ``environments/alphabet_sort/alphabet_sort.py``).
Single deterministic local name pool — no network access required.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from renderers import Message

from torchtitan.config import Configurable
from torchtitan.experiments.rl.alphabet_sort.data import AlphabetSortDataset
from torchtitan.experiments.rl.alphabet_sort.grading import score_completion_text
from torchtitan.experiments.rl.envs.types import (
    EnvExample,
    EnvReset,
    EnvStep,
    MessageEnv,
)
from torchtitan.experiments.rl.types import RolloutStatus

__all__ = ["AlphabetSortBuilder", "AlphabetSortDataset", "AlphabetSortEnv"]


def _sort_key(name: str, *, by_last: bool) -> str:
    """Lowercase last token (by_last) or first token (else) as the sort key."""
    parts = name.split()
    return (parts[-1] if by_last else parts[0]).lower()


def _camelcase(name: str) -> str:
    return name.replace(" ", "")


def _initial_prompt(names: list[str], *, sort_by_last: bool) -> str:
    sort_type = "LAST" if sort_by_last else "FIRST"
    shuffled = ", ".join(_camelcase(n) for n in names)
    return (
        f"Sort these names in alphabetical order by {sort_type} name: {shuffled}\n"
        "\n"
        "Use exactly this format:\n"
        "<alphabetical_sorted>\n"
        "Name1\nName2\n...\n"
        "</alphabetical_sorted>"
    )


def _follow_up_prompt(new_names: list[str], *, sort_by_last: bool) -> str:
    sort_type = "LAST" if sort_by_last else "FIRST"
    shuffled = ", ".join(_camelcase(n) for n in new_names)
    return (
        f"Now sort ALL of these names alphabetically by {sort_type} name: {shuffled}\n"
        "\n"
        "These are in addition to the prior list. Mark any NEW names "
        "(that weren't in the prior list) with `// new name!` at the end.\n"
        "\n"
        "Use exactly this format:\n"
        "<combined_alphabetical_sorted>\n"
        "Name1\nName2 // new name!\n...\n"
        "</combined_alphabetical_sorted>"
    )


class AlphabetSortEnv(MessageEnv):
    """One AlphabetSort episode: introduce names across N turns, score each.

    ``reward`` is stamped on the terminal step only — the per-turn
    similarity is accumulated and aggregated per ``power_per_turn``:

    - ``power_per_turn=True``: ``mean_i(similarity_i ** power)``
    - ``power_per_turn=False``: ``mean_i(similarity_i) ** power``

    Verifiers' default is ``True``; prime-rl's RL run uses ``False``
    (heavily penalizes consistently mediocre output).

    Example::

        env = AlphabetSortEnv(
            names=["Aaron Bell", "Brian Mack", "Caleb Oak"],
            names_per_turn=[2, 1],
            sort_by_last=True,
        )
        reset = await env.reset()
        # reset.messages[-1]["content"] introduces "AaronBell" and "BrianMack".
        ...
    """

    def __init__(
        self,
        *,
        names: Sequence[str],
        names_per_turn: Sequence[int],
        sort_by_last: bool,
        similarity_power: int = 4,
        power_per_turn: bool = False,
    ) -> None:
        if sum(names_per_turn) != len(names):
            raise ValueError(
                f"sum(names_per_turn)={sum(names_per_turn)} must equal "
                f"len(names)={len(names)}"
            )
        self._names = list(names)
        self._names_per_turn = list(names_per_turn)
        self._sort_by_last = sort_by_last
        self._similarity_power = similarity_power
        self._power_per_turn = power_per_turn

        self._turn_idx: int = 0
        self._similarities: list[float] = []  # raw per-turn similarities

    async def reset(self) -> EnvReset:
        first_batch = self._names[: self._names_per_turn[0]]
        return EnvReset(
            messages=[
                {
                    "role": "user",
                    "content": _initial_prompt(
                        first_batch, sort_by_last=self._sort_by_last
                    ),
                }
            ]
        )

    async def step(self, assistant_message: Message) -> EnvStep:
        text = str(assistant_message.get("content") or "")
        cumulative_count = sum(self._names_per_turn[: self._turn_idx + 1])
        expected = sorted(
            (_camelcase(n) for n in self._names[:cumulative_count]),
            key=lambda c: _sort_key(c, by_last=self._sort_by_last),
        )

        # Store the raw similarity; aggregation applies the power once
        # at episode end.
        self._similarities.append(
            score_completion_text(text, expected, turn_idx=self._turn_idx)
        )

        self._turn_idx += 1
        if self._turn_idx >= len(self._names_per_turn):
            reward = self._aggregate_reward()
            return EnvStep(
                reward=reward,
                reward_components={"similarity": reward},
                done=True,
                status=RolloutStatus.COMPLETED,
            )

        next_count = self._names_per_turn[self._turn_idx]
        next_names = self._names[cumulative_count : cumulative_count + next_count]
        return EnvStep(
            messages=[
                {
                    "role": "user",
                    "content": _follow_up_prompt(
                        next_names, sort_by_last=self._sort_by_last
                    ),
                }
            ],
            done=False,
        )

    async def close(self) -> None:
        pass

    def _aggregate_reward(self) -> float:
        if not self._similarities:
            return 0.0
        if self._power_per_turn:
            # Per-turn power then mean: mean_i(sim_i^p).
            return sum(s**self._similarity_power for s in self._similarities) / len(
                self._similarities
            )
        # Holistic: mean of raw similarities, then power.
        return (
            sum(self._similarities) / len(self._similarities)
        ) ** self._similarity_power


class AlphabetSortBuilder(Configurable):
    """Build a group of sibling :class:`AlphabetSortEnv`s from one row."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        similarity_power: int = 4
        """Powers up the Ratcliff/Obershelp ratio. Verifiers default ``4``;
        prime-rl eval uses ``8`` (very strict). Higher = more punitive."""

        power_per_turn: bool = False
        """``True``: ``mean(sim^p)``. ``False``: ``mean(sim)^p`` (prime-rl run)."""

    def __init__(self, config: Config) -> None:
        self.config = config

    async def make_envs(
        self, example: EnvExample, *, group_size: int
    ) -> Sequence[MessageEnv]:
        payload = example.payload
        return [
            AlphabetSortEnv(
                names=payload["names"],
                names_per_turn=payload["names_per_turn"],
                sort_by_last=payload["sort_by_last"],
                similarity_power=self.config.similarity_power,
                power_per_turn=self.config.power_per_turn,
            )
            for _ in range(group_size)
        ]
