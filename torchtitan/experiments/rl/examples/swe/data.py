# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""R2E-Gym dataset for the SWE coding-agent example.

One sample carries everything needed to run AND grade an R2E-Gym instance: the
problem statement, the per-instance container image (deps + repo baked in), and
the hidden test spec used at grade time. The jsonl is produced by slime's
``r2e_to_slime.py`` (R2E-Gym HF rows -> one row per line); see ``README.md``.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass

from torchtitan.config import Configurable


@dataclass(frozen=True, kw_only=True, slots=True)
class R2EGymSample:
    """One R2E-Gym instance: problem + work image + hidden-test grading spec."""

    instance_id: str
    """Stable id, e.g. ``orange3-2d9617bd0c``."""

    image: str
    """Per-instance container image (repo + deps baked in), pinned by digest/tag."""

    problem_statement: str
    """The issue text shown to the agent."""

    test_file_names: tuple[str, ...]
    """Hidden test file paths (relative to the repo root), injected at grade time."""

    test_file_codes: tuple[str, ...]
    """Source of each hidden test file, aligned with ``test_file_names``."""

    expected_output_json: str
    """JSON mapping ``test_name -> PASSED|FAILED|SKIPPED`` that a correct fix
    reproduces. The grading source of truth; never exposed to the agent."""


class R2EGymDataset(Configurable):
    """Endless, seeded stream of ``R2EGymSample`` read from a jsonl file.

    Row order is shuffled with ``seed`` and reshuffled on each wrap, so a run sees
    a fresh permutation every epoch. Set ``shuffle=False`` for validation so each
    pass draws the same held-out samples in the same order.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        data_path: str
        """Path to the R2E-Gym jsonl (one instance per line). Required."""

        seed: int = 42
        """Seed for the row-order shuffle."""

        shuffle: bool = True
        """Shuffle row order (reshuffling on wrap). False = deterministic order."""

    def __init__(self, config: Config) -> None:
        if not config.data_path:
            raise ValueError("R2EGymDataset.Config.data_path must be set")
        self._samples = _load_jsonl(config.data_path)
        if not self._samples:
            raise ValueError(f"no rows found in {config.data_path}")
        self._rng = random.Random(config.seed)
        self._shuffle = config.shuffle
        self._order = list(range(len(self._samples)))
        if self._shuffle:
            self._rng.shuffle(self._order)
        self._pos = 0

    def __iter__(self) -> Iterator[R2EGymSample]:
        return self

    def __next__(self) -> R2EGymSample:
        if self._pos >= len(self._order):
            if self._shuffle:
                self._rng.shuffle(self._order)
            self._pos = 0
        idx = self._order[self._pos]
        self._pos += 1
        return self._samples[idx]

    def state_dict(self) -> dict:
        """Snapshot RNG + position so a run can resume mid-stream."""
        return {
            "rng_state": self._rng.getstate(),
            "order": list(self._order),
            "pos": self._pos,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._rng.setstate(state_dict["rng_state"])
        self._order = list(state_dict["order"])
        self._pos = state_dict["pos"]


def _load_jsonl(path: str) -> list[R2EGymSample]:
    samples: list[R2EGymSample] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(_row_to_sample(json.loads(line)))
    return samples


def _row_to_sample(row: dict) -> R2EGymSample:
    """Convert one slime jsonl row into an ``R2EGymSample``.

    Row shape (from ``r2e_to_slime.py``)::

        {"prompt", "label", "metadata": {"instance_id", "image", "workdir",
         "problem_statement", "r2e": {"test_file_names", "test_file_codes",
         "expected_output_json"}}}
    """
    meta = row["metadata"]
    r2e = meta["r2e"]
    return R2EGymSample(
        instance_id=meta["instance_id"],
        image=meta["image"],
        problem_statement=meta["problem_statement"],
        test_file_names=tuple(r2e["test_file_names"]),
        test_file_codes=tuple(r2e["test_file_codes"]),
        expected_output_json=r2e["expected_output_json"],
    )
