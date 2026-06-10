# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from renderers import Message

from torchtitan.experiments.rl.environment import (
    MessageEnv,
    MessageEnvInitOutput,
    MessageEnvStepOutput,
)
from torchtitan.experiments.rl.examples.alphabet_sort.data import AlphabetSortSample


class AlphabetSortEnv(MessageEnv):
    """Runs one AlphabetSort sample as a chat: ask the model to sort a list of names, turn by turn.

    `init` asks the model to sort the first batch of names. Each later `step` introduces a new
    batch of names and asks the model to re-sort the whole list so far, marking the new ones. When
    no batches are left, `step` ends the rollout (so a single-turn sample ends at the first step).

    Example (a 2-turn sample):

        init:  sort [MarcChardin, AnaChardin]
        step:  add [BobBeck], then re-sort all three and mark BobBeck as new
        step:  no names left -> end the rollout
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MessageEnv.Config):
        pass

    def __init__(self, config: Config, *, env_input: AlphabetSortSample) -> None:
        self._env_input = env_input

    async def init(self) -> MessageEnvInitOutput:
        """Ask the model to sort the first batch of names."""
        # init asks for turn 0; step asks for turns 1, 2, ... in order
        self._next_turn = 1
        sort_by = "FIRST" if self._env_input.sort_by_first_name else "LAST"
        prompt = (
            f"Sort these names in alphabetical order by {sort_by} name: "
            f"{', '.join(self._env_input.new_names_per_turn[0])}\n\n"
            "Use exactly this format:\n"
            "<alphabetical_sorted>\n"
            "Name1\n"
            "Name2\n"
            "</alphabetical_sorted>"
        )
        return MessageEnvInitOutput(
            init_prompt_messages=[{"role": "user", "content": prompt}]
        )

    async def step(self, completion_message: Message) -> MessageEnvStepOutput:
        """Introduce the next batch of new names to re-sort, or end the rollout if none are left."""
        if self._next_turn >= len(self._env_input.new_names_per_turn):
            return MessageEnvStepOutput(done=True)
        new_names = self._env_input.new_names_per_turn[self._next_turn]
        self._next_turn += 1
        sort_by = "FIRST" if self._env_input.sort_by_first_name else "LAST"
        prompt = (
            f"Now sort ALL of these names alphabetically by {sort_by} name: "
            f"{', '.join(new_names)}\n\n"
            "These are in addition to the prior list. Mark any NEW names "
            "(that weren't in the prior list) with `// new name!` at the end.\n\n"
            "Use exactly this format:\n"
            "<combined_alphabetical_sorted>\n"
            "Name1\n"
            "Name2 // new name!\n"
            "</combined_alphabetical_sorted>"
        )
        return MessageEnvStepOutput(env_messages=[{"role": "user", "content": prompt}])
