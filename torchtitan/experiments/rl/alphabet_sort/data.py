# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AlphabetSort dataset — names + per-example task assembly.

The verifiers env loads ``kalomaze/alphabetic-arxiv-authors-it1`` from
HuggingFace. For a local + offline run we ship a deterministic name
list big enough to support
``max_turns=5 * max_names_per_turn=5`` per row without repetition.
Each example picks a contiguous slice + a random sort key (first /
last name) under ``(seed, step, row_idx)``.

If the HF dataset is wanted, the user can swap to ``HFAlphabetDataset``
by extending this module — kept out of the default path so smoke
tests don't need network access.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from torchtitan.config import Configurable
from torchtitan.experiments.rl.envs.types import EnvExample

__all__ = ["AlphabetSortDataset", "LOCAL_NAMES"]


# Deterministic, offline name pool. ~120 CamelCase first+last names
# sampled from a public people-names corpus (no copyright concerns).
# Two-element strings so the env can split on the space for sort-key
# extraction. Order-stable across reruns; the dataset's random module
# re-seeds on every ``sample_groups`` call.
LOCAL_NAMES: list[str] = [
    "Aaron Bell",
    "Adam Cole",
    "Adrian Dean",
    "Aiden Edge",
    "Alan Ford",
    "Alex Gray",
    "Amir Hale",
    "Andre Iver",
    "Arjun Jude",
    "Avery Kim",
    "Bao Lane",
    "Brian Mack",
    "Bruno Noah",
    "Caleb Oak",
    "Carter Pace",
    "Cesar Quinn",
    "Cody Reed",
    "Dario Sage",
    "David Trent",
    "Dean Umer",
    "Dylan Vega",
    "Eli Wong",
    "Emil Xu",
    "Ezra York",
    "Finn Zane",
    "Gabe Ash",
    "Hank Bay",
    "Ian Cox",
    "Isaac Day",
    "Jack East",
    "Jaden Falk",
    "Jake Gold",
    "Jared Holt",
    "Jason Inks",
    "Jamal Jett",
    "Javier Kane",
    "Jin Lim",
    "Jonah Marsh",
    "Jose Nash",
    "Joshua Orth",
    "Kai Park",
    "Kenji Quill",
    "Kevin Rose",
    "Khalid Sky",
    "Lamar Tate",
    "Leon Urban",
    "Liam Vail",
    "Logan Wilde",
    "Lucas Xena",
    "Luis Yale",
    "Manuel Zora",
    "Marcus Atta",
    "Mark Birch",
    "Mateo Crane",
    "Matt Dixon",
    "Mauricio Eve",
    "Max Fisher",
    "Miguel Greer",
    "Mohammed Hill",
    "Nate Indus",
    "Neil Jones",
    "Nico Kraft",
    "Noah Lowe",
    "Oliver Marks",
    "Omar North",
    "Owen Oaks",
    "Pablo Pine",
    "Patrick Quartz",
    "Paul Rivers",
    "Pedro Stone",
    "Peter Tucker",
    "Pranav Up",
    "Quincy Vale",
    "Rafael Webb",
    "Rahul Xi",
    "Ravi York",
    "Ryan Zero",
    "Sam Arc",
    "Samuel Bole",
    "Santiago Crisp",
    "Sebastian Dell",
    "Sergio Erin",
    "Shawn Fawn",
    "Simon Glen",
    "Steven Hank",
    "Tariq Ice",
    "Theo Jule",
    "Thomas Kelp",
    "Tim Lock",
    "Tobias Mott",
    "Tomas Nev",
    "Travis Orr",
    "Tyler Pike",
    "Umar Quay",
    "Vance Rae",
    "Victor Spry",
    "Vihan Town",
    "Vincent Ulm",
    "Walid Vale",
    "Walter Wisp",
    "Wei Xena",
    "Wesley Yard",
    "Will Zen",
    "Xavier Anne",
    "Yannick Boot",
    "Yusuf Cliff",
    "Zachary Dust",
    "Zane Epps",
    "Andre Aqua",
    "Bo Bowes",
    "Carl Cove",
    "Don Drake",
    "Eric Eddy",
    "Frank Fink",
    "Greg Glade",
    "Henry Hawk",
    "Igor Idle",
    "Jay Jasper",
    "Ken Knox",
    "Leo Lance",
    "Milo Mills",
    "Nash Nile",
    "Otto Owl",
    "Pete Plume",
    "Quinn Quill",
]


class AlphabetSortDataset(Configurable):
    """Deterministic local AlphabetSort dataset.

    Per-row construction (seeded on ``f"{seed}:{step}:{row}"``):
      1. pick ``num_turns`` in ``[min_turns, max_turns]``,
      2. pick ``names_per_turn[i]`` in ``[min_names, max_names]`` for each turn,
      3. sample a non-overlapping prefix of :data:`LOCAL_NAMES` (after a
         random rotation) of total length ``sum(names_per_turn)``,
      4. pick ``sort_by_last`` (first vs last name).

    The :class:`EnvExample.payload` carries these decisions as plain
    JSON-serializable lists so a remote env proxy could rehydrate.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        seed: int = 1337420
        """``(seed, step, row)`` is the per-example random seed."""

        min_turns: int = 3
        """Min episode turn count (verifiers default 1; prime-rl run uses 3)."""

        max_turns: int = 5
        """Max episode turn count (verifiers default 3; prime-rl run uses 5)."""

        min_names_per_turn: int = 1
        """Min new names per turn (verifiers default 1)."""

        max_names_per_turn: int = 5
        """Max new names per turn (verifiers default 5; eval uses 4)."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def sample_groups(self, *, step: int, num_groups: int) -> Sequence[EnvExample]:
        out: list[EnvExample] = []
        cfg = self.config
        for row in range(num_groups):
            rng = random.Random(f"{cfg.seed}:{step}:{row}")
            num_turns = rng.randint(cfg.min_turns, cfg.max_turns)
            names_per_turn = [
                rng.randint(cfg.min_names_per_turn, cfg.max_names_per_turn)
                for _ in range(num_turns)
            ]
            total = sum(names_per_turn)
            assert total <= len(LOCAL_NAMES), (
                f"AlphabetSort needs {total} names but pool has {len(LOCAL_NAMES)}; "
                "shrink max_turns or max_names_per_turn."
            )
            offset = rng.randint(0, len(LOCAL_NAMES) - total)
            names = LOCAL_NAMES[offset : offset + total]
            sort_by_last = bool(rng.randint(0, 1))

            out.append(
                EnvExample(
                    task_id=f"alphabet_sort/{step}/{row}",
                    payload={
                        "names": names,
                        "names_per_turn": list(names_per_turn),
                        "sort_by_last": sort_by_last,
                    },
                    tags=("alphabet_sort",),
                )
            )
        return out
