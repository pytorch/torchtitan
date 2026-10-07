# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from typing import Any

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

    verifiers_task_data: dict[str, Any]


class VerifiersTaskSource(Configurable):
    """Materialize one Verifiers taskset as a random-access source."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        verifiers_taskset: VerifiersTasksetConfig
        """Typed configuration for the Verifiers taskset to load."""

        num_tasks: int | None = None
        """Optional task cap, required when the taskset is infinite."""

        def __post_init__(self) -> None:
            if self.num_tasks is not None and self.num_tasks <= 0:
                raise ValueError("num_tasks must be positive")

        def to_dict(self) -> dict[str, Any]:
            return {
                "verifiers_taskset": self.verifiers_taskset.model_dump(mode="json"),
                "num_tasks": self.num_tasks,
            }

    def __init__(self, config: Config) -> None:
        verifiers_taskset_config = config.verifiers_taskset.model_copy(
            update={"id": register_local_taskset_alias(config.verifiers_taskset.id)}
        )
        verifiers_taskset = load_taskset(verifiers_taskset_config)
        if config.num_tasks is None and verifiers_taskset.INFINITE:
            raise ValueError(
                f"Verifiers taskset {config.verifiers_taskset.id!r} is infinite; "
                "num_tasks is required"
            )
        verifiers_taskset = (
            verifiers_taskset
            if config.num_tasks is None
            else verifiers_taskset.head(config.num_tasks)
        )
        tasks = list(verifiers_taskset)
        if not tasks:
            raise ValueError(
                f"Verifiers taskset {config.verifiers_taskset.id!r} yielded no tasks"
            )
        if config.num_tasks is not None and len(tasks) != config.num_tasks:
            raise ValueError(
                f"Verifiers taskset {config.verifiers_taskset.id!r} yielded {len(tasks)} "
                f"tasks, expected {config.num_tasks}"
            )

        self._samples = [
            VerifiersTaskSample(verifiers_task_data=task.data.model_dump(mode="json"))
            for task in tasks
        ]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> VerifiersTaskSample:
        return self._samples[index]
