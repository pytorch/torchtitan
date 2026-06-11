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

    Prompts are copied verbatim from PrimeIntellect's `verifiers` alphabet_sort environment (the
    same task): `init` asks the model to sort the first batch into an `<alphabetical_sorted>` block;
    each later `step` adds a batch and asks for the whole list re-sorted into a
    `<combined_alphabetical_sorted>` block, with the names new that round tagged `// new name!`. Each
    turn is self-contained (no upfront multi-round preamble); the first `step` shows the combined
    format with an example, later ones just say "follow the same format". The format example ends
    with a literal `...` so the model doesn't copy a fixed number of rows. When no batches are left,
    `step` ends the rollout (so a single-turn sample ends at the first step).

    Example (a 2-turn sample):

        init:  Sort these names ...: MarcChardin, AnaChardin
        step:  Now sort ALL ...: BobBeck   (re-sort all three, tag BobBeck `// new name!`)
        step:  no names left -> end the rollout
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MessageEnv.Config):
        pass

    def __init__(self, config: Config, *, env_input: AlphabetSortSample) -> None:
        self._env_input = env_input

    async def init(self) -> MessageEnvInitOutput:
        """Ask the model to sort the first batch of names (turn 0)."""
        # init asks for turn 0; step asks for turns 1, 2, ... in order
        self._next_turn = 1
        sort_type = "FIRST" if self._env_input.sort_by_first_name else "LAST"
        names = ", ".join(self._env_input.new_names_per_turn[0])
        prompt = (
            f"Sort these names in alphabetical order by {sort_type} name: {names}\n\n"
            "Use exactly this format:\n"
            "<alphabetical_sorted>\nName1\nName2\n...\n</alphabetical_sorted>"
        )
        return MessageEnvInitOutput(
            init_prompt_messages=[{"role": "user", "content": prompt}]
        )

    async def step(self, completion_message: Message) -> MessageEnvStepOutput:
        """Introduce the next batch of new names to re-sort, or end the rollout if none are left."""
        if self._next_turn >= len(self._env_input.new_names_per_turn):
            return MessageEnvStepOutput(done=True)
        turn = self._next_turn
        self._next_turn += 1
        sort_type = "FIRST" if self._env_input.sort_by_first_name else "LAST"
        names = ", ".join(self._env_input.new_names_per_turn[turn])
        prompt = (
            f"Now sort ALL of these names alphabetically by {sort_type} name: {names}\n\n"
            "These are in addition to the prior list. Mark any NEW names (that weren't in the "
            "prior list) with `// new name!` at the end."
        )
        if turn == 1:
            # The first re-sort shows the combined format with one new name tagged.
            prompt += (
                "\n\nUse exactly this format:\n"
                "<combined_alphabetical_sorted>\nName1\nName2 // new name!\n...\n"
                "</combined_alphabetical_sorted>"
            )
        else:
            prompt += " Follow the same format as before."
        return MessageEnvStepOutput(env_messages=[{"role": "user", "content": prompt}])
