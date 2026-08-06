# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field

from torchtitan.experiments.rl.examples.alphabet_sort.data import AlphabetSortDataset
from torchtitan.experiments.rl.examples.alphabet_sort.env import AlphabetSortEnv
from torchtitan.experiments.rl.examples.alphabet_sort.rubric import RewardAlphabetSort

from torchtitan.experiments.rl.rollout.rollouter import Rollouter

from torchtitan.experiments.rl.rubrics import Rubric


class AlphabetSortRollouter(Rollouter):
    """Wires up the AlphabetSort task: its train/val datasets, env, and reward.

    The model is shown a shuffled list of researcher names and must return them sorted
    alphabetically (by first or last name) inside an `<alphabetical_sorted>` block. Reward is the
    order similarity between the returned list and the correct one, so it rises as the policy
    learns to sort. Multi-turn samples add more names each turn and ask for the whole list
    re-sorted, marking the names new that turn.

    Example (two turns, by first name):

        turn 1 prompt:  Sort by FIRST name: MarcChardin, AnaChardin
        turn 1 model:   <alphabetical_sorted>
                        AnaChardin
                        MarcChardin
                        </alphabetical_sorted>

        turn 2 prompt:  Now sort ALL by FIRST name, marking new names: BobBeck
        turn 2 model:   <combined_alphabetical_sorted>
                        AnaChardin
                        BobBeck // new name!
                        MarcChardin
                        </combined_alphabetical_sorted>          # reward 1.0

    Pure config — `make_env_group`, `get_*_sample`, and `score_group` are inherited from `Rollouter`.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Rollouter.Config):
        train_dataset: AlphabetSortDataset.Config = field(
            default_factory=lambda: AlphabetSortDataset.Config(seed=42)
        )
        validation_dataset: AlphabetSortDataset.Config = field(
            default_factory=lambda: AlphabetSortDataset.Config(seed=99)
        )
        rubric: Rubric.Config = field(
            default_factory=lambda: Rubric.Config(
                reward_fns=[RewardAlphabetSort.Config(weight=1.0)]
            )
        )
        message_env: AlphabetSortEnv.Config = field(
            default_factory=AlphabetSortEnv.Config
        )
