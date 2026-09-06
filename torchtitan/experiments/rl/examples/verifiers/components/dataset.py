# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import grain.python as grain
from verifiers.v1.configs.taskset import TasksetConfig as VerifiersTasksetConfig
from verifiers.v1.utils.loaders import load_taskset

from torchtitan.config import Configurable


def register_local_taskset_alias(taskset_id: str) -> str:
    """Register a dotted local taskset under an importable Verifiers plugin ID."""
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
    """Adapt one Verifiers taskset to a resumable Grain iterator."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        taskset: VerifiersTasksetConfig
        """Typed configuration for the Verifiers taskset to load."""

        num_tasks: int | None = None
        """Optional task cap, required when the taskset is infinite."""

        seed: int = 42
        """Seed used to produce a reproducible task order."""

        shuffle: bool = True
        """Whether to shuffle the materialized task order before repetition."""

        def __post_init__(self) -> None:
            if self.num_tasks is not None and self.num_tasks <= 0:
                raise ValueError("num_tasks must be positive")

        def to_dict(self) -> dict[str, Any]:
            return {
                "taskset": self.taskset.model_dump(mode="json"),
                "num_tasks": self.num_tasks,
                "seed": self.seed,
                "shuffle": self.shuffle,
            }

    def __init__(self, config: Config) -> None:
        taskset_config = config.taskset.model_copy(
            update={"id": register_local_taskset_alias(config.taskset.id)}
        )
        taskset = load_taskset(taskset_config)
        if config.num_tasks is None and taskset.INFINITE:
            raise ValueError(
                f"Verifiers taskset {config.taskset.id!r} is infinite; "
                "num_tasks is required"
            )
        taskset = (
            taskset if config.num_tasks is None else taskset.head(config.num_tasks)
        )
        if config.shuffle:
            taskset = taskset.shuffle(config.seed)
        tasks = list(taskset)
        if not tasks:
            raise ValueError(
                f"Verifiers taskset {config.taskset.id!r} yielded no tasks"
            )
        if config.num_tasks is not None and len(tasks) != config.num_tasks:
            raise ValueError(
                f"Verifiers taskset {config.taskset.id!r} yielded {len(tasks)} "
                f"tasks, expected {config.num_tasks}"
            )

        samples = [
            VerifiersTaskSample(task_data=task.data.model_dump(mode="json"))
            for task in tasks
        ]
        dataset = grain.MapDataset.source(samples)
        self._iterator = iter(dataset.repeat().to_iter_dataset())

    def __iter__(self) -> Iterator[VerifiersTaskSample]:
        return self

    def __next__(self) -> VerifiersTaskSample:
        return next(self._iterator)

    def state_dict(self) -> dict:
        return self._iterator.get_state()

    def load_state_dict(self, state_dict: dict) -> None:
        self._iterator.set_state(state_dict)
