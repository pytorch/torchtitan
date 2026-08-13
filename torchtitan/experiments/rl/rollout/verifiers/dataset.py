# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from torchtitan.config import Configurable


@dataclass(frozen=True, kw_only=True, slots=True)
class VerifiersTaskSample:
    """Serialized task data dispatched to a stateless Verifiers EnvServer."""

    task_data: dict[str, Any]


class VerifiersTaskDataset(Configurable):
    """Load and schedule tasks for a Verifiers-backed rollout.

    Verifiers 0.3 keeps v1 datasets on the client: TorchTitan loads the taskset and
    sends each task's serialized ``TaskData`` to the EnvServer. The server owns the
    environment, harness, runtime, and scoring behavior, and rebuilds the task from
    that data without loading the dataset itself.

    This adapter implements TorchTitan's dataset iterator contract and controls
    task selection, shuffling, and epoch cycling.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        taskset_id: str
        num_tasks: int
        taskset_args: dict[str, Any] = field(default_factory=dict)
        task_indices: tuple[int, ...] = ()
        seed: int = 42
        shuffle: bool = True

        def __post_init__(self) -> None:
            if self.num_tasks <= 0:
                raise ValueError("num_tasks must be positive")
            if any(index < 0 or index >= self.num_tasks for index in self.task_indices):
                raise ValueError(
                    "task_indices must be in [0, num_tasks); got "
                    f"num_tasks={self.num_tasks}, task_indices={self.task_indices}"
                )

    def __init__(self, config: Config) -> None:
        from verifiers.v1.utils.loaders import load_taskset, taskset_config_type

        taskset_config = taskset_config_type(config.taskset_id).model_validate(
            {"id": config.taskset_id, **config.taskset_args}
        )
        tasks = list(load_taskset(taskset_config).head(config.num_tasks))
        if len(tasks) != config.num_tasks:
            raise ValueError(
                f"Verifiers taskset {config.taskset_id!r} yielded {len(tasks)} "
                f"tasks, expected {config.num_tasks}"
            )
        indices = config.task_indices or tuple(range(config.num_tasks))
        self._samples = [
            VerifiersTaskSample(task_data=tasks[index].data.model_dump(mode="json"))
            for index in indices
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
