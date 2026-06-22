# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""R2E-Gym (SWE) dataset for the coding-agent RL example.

Reads a JSONL produced from R2E-Gym (see THUDM/slime
``examples/coding_agent_rl/local_smoke/r2e_to_slime.py``). Each row::

    {
      "prompt": <problem_statement>,
      "label": <instance_id>,
      "metadata": {
        "instance_id", "image" (docker.io/...), "workdir" "/testbed",
        "problem_statement",
        "r2e": {"test_file_names", "test_file_codes", "expected_output_json"},
        "pre_commands": <optional list[str]|str>
      }
    }

The dataset is an endless, seeded stream of frozen ``SWER2ESample``s, mirroring
``SearchR1Dataset``.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass, field

from torchtitan.config import Configurable


@dataclass(frozen=True, kw_only=True, slots=True)
class SWER2ESample:
    """One R2E-Gym SWE task: a containerized repo, an issue, and hidden tests."""

    instance_id: str
    """Stable task id (e.g. ``orange3-4467fb9e92``)."""

    image: str
    """Docker image with the repo + interpreter (e.g. ``docker.io/namanjain12/...``)."""

    workdir: str
    """Repo path inside the sandbox (R2E default ``/testbed``)."""

    problem_statement: str
    """The issue body the agent must resolve."""

    r2e: dict = field(default_factory=dict)
    """Grading payload: ``test_file_names``, ``test_file_codes``, ``expected_output_json``."""

    pre_commands: list[str] | str | None = None
    """Optional baseline-alignment commands run before the agent / eval."""


class SWER2EDataset(Configurable):
    """Endless, seeded stream of R2E-Gym SWE samples loaded from a JSONL file."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        data_path: str = ""
        """Path to the R2E JSONL file (required)."""

        seed: int = 42
        """Seed for the row-order shuffle."""

        shuffle: bool = True
        """Shuffle row order (reshuffling on each wrap). Set False for validation."""

    def __init__(self, config: Config) -> None:
        if not config.data_path:
            raise ValueError("SWER2EDataset.Config.data_path is required")
        samples: list[SWER2ESample] = []
        with open(config.data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                md = row.get("metadata") or {}
                instance_id = (
                    md.get("instance_id")
                    or (row.get("label") if isinstance(row.get("label"), str) else None)
                    or "unknown"
                )
                image = md.get("image")
                workdir = md.get("workdir")
                if not image or not workdir:
                    raise ValueError(
                        f"row {instance_id!r} missing image/workdir in metadata"
                    )
                samples.append(
                    SWER2ESample(
                        instance_id=instance_id,
                        image=image,
                        workdir=workdir,
                        problem_statement=md.get("problem_statement")
                        or _coerce_prompt(row.get("prompt")),
                        r2e=md.get("r2e") or {},
                        pre_commands=md.get("pre_commands"),
                    )
                )
        if not samples:
            raise ValueError(f"no rows found in {config.data_path}")
        self._samples = samples

        self._rng = random.Random(config.seed)
        self._shuffle = config.shuffle
        self._order = list(range(len(self._samples)))
        if self._shuffle:
            self._rng.shuffle(self._order)
        self._pos = 0

    def __iter__(self) -> Iterator[SWER2ESample]:
        return self

    def __next__(self) -> SWER2ESample:
        if self._pos >= len(self._order):
            if self._shuffle:
                self._rng.shuffle(self._order)
            self._pos = 0
        idx = self._order[self._pos]
        self._pos += 1
        return self._samples[idx]

    def state_dict(self) -> dict:
        return {
            "rng_state": self._rng.getstate(),
            "order": list(self._order),
            "pos": self._pos,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._rng.setstate(state_dict["rng_state"])
        self._order = list(state_dict["order"])
        self._pos = state_dict["pos"]


def _coerce_prompt(prompt) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for m in prompt:
            if isinstance(m, dict) and m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, str):
                    return content
    return ""
