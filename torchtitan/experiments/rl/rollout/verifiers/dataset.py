# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
import random
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from torchtitan.config import Configurable


def _load_taskset_id(taskset_id: str) -> str:
    """Load a dotted local taskset module under a Verifiers-compatible alias."""
    if "." not in taskset_id or "/" in taskset_id:
        return taskset_id

    module = importlib.import_module(taskset_id)
    alias = taskset_id.replace(".", "_").lower()
    existing = sys.modules.get(alias)
    if existing is not None and existing is not module:
        raise ValueError(f"taskset alias {alias!r} is already registered")
    sys.modules[alias] = module
    return alias


@dataclass(frozen=True, kw_only=True, slots=True)
class VerifiersTaskSample:
    """Serialized task data dispatched to a stateless Verifiers EnvServer."""

    task_data: dict[str, Any]


class VerifiersTaskDataset(Configurable):
    """Load a Verifiers taskset into TorchTitan's resumable dataset contract."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        taskset_id: str
        num_tasks: int
        taskset_args: dict[str, Any] = field(default_factory=dict)
        seed: int = 42
        shuffle: bool = True

        def __post_init__(self) -> None:
            if self.num_tasks <= 0:
                raise ValueError("num_tasks must be positive")

    def __init__(self, config: Config) -> None:
        # Verifiers is an example-only dependency. Keep the import local so the
        # rest of TorchTitan RL does not require it.
        from verifiers.v1.utils.loaders import load_taskset, taskset_config_type

        taskset_id = _load_taskset_id(config.taskset_id)
        taskset_config = taskset_config_type(taskset_id).model_validate(
            {"id": taskset_id, **config.taskset_args}
        )
        tasks = list(load_taskset(taskset_config).head(config.num_tasks))
        if len(tasks) != config.num_tasks:
            raise ValueError(
                f"Verifiers taskset {config.taskset_id!r} yielded {len(tasks)} "
                f"tasks, expected {config.num_tasks}"
            )

        self._samples = [
            VerifiersTaskSample(task_data=task.data.model_dump(mode="json"))
            for task in tasks
        ]
        self._rng = random.Random(config.seed)
        self._shuffle = config.shuffle
        self._order = list(range(len(self._samples)))
        self._position = 0
        if self._shuffle:
            self._rng.shuffle(self._order)

    def __iter__(self) -> Iterator[VerifiersTaskSample]:
        return self

    def __next__(self) -> VerifiersTaskSample:
        if self._position == len(self._order):
            self._position = 0
            if self._shuffle:
                self._rng.shuffle(self._order)
        sample = self._samples[self._order[self._position]]
        self._position += 1
        return sample

    def state_dict(self) -> dict:
        return {
            "rng_state": self._rng.getstate(),
            "order": list(self._order),
            "position": self._position,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._rng.setstate(state_dict["rng_state"])
        self._order = list(state_dict["order"])
        self._position = int(state_dict["position"])
