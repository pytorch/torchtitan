# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import grain.python as grain
import verifiers.v1 as vf

from torchtitan.rl.rollout.data import RLDataset


@dataclass(frozen=True, kw_only=True, slots=True)
class VerifiersTaskSample:
    """One Verifiers task, and the env that runs it.

    Example:
        VerifiersTaskSample(
            env_config_json='{"id": "", "taskset": {"id": "torchtitan-verifiers-dapo-math", ...}, ...}',
            task_data={"idx": 17, "prompt": [{"role": "user", "content": "..."}], "ground_truth": "42"},
        )
    """

    env_config_json: str
    """The env's config as JSON. Plain data, so a checkpoint of pending samples holds no
    Verifiers class; equal strings mean the same env."""

    # TODO: Consider sending the task's fields as one lazily deserialized object, decoded
    # only where the task runs. The hops before it (the controller, a checkpoint of pending
    # samples) would copy bytes, not a nested dict.
    task_data: dict[str, Any]
    """The task's `TaskData` as JSON-ready data."""


class VerifiersTaskDataset(RLDataset):
    """The tasks of a Verifiers env's taskset, repeated in a resumable order. The env is
    configured once, here: a rollout worker builds it from each sample.

    Example:
        dataset = VerifiersTaskDataset.Config(
            env=vf.SingleAgentEnvConfig(taskset=MyTasksetConfig(id="my-package"), agent=...),
            select=vf.SelectConfig(shuffle=True, seed=42),
        ).build()
        next(dataset)  # VerifiersTaskSample(env_config_json='{"taskset": ...}', task_data={...})
    """

    # TODO: implement this as a SourceConfig and reuse SingleDatasetConfig once
    # Rollouter supplies the core data pipeline's build context and iteration policy.
    # That pipeline also mixes sources: one training stream of two envs, e.g. DAPO math and
    # Terminal-Bench. Each sample already names its env, and a worker serves several.

    @dataclass(kw_only=True, slots=True)
    class Config(RLDataset.Config):
        env: vf.EnvConfig
        """The Verifiers env that runs these tasks, e.g. `vf.SingleAgentEnvConfig(taskset=...,
        agent=...)`. Its taskset gives the tasks."""

        select: vf.SelectConfig = field(default_factory=vf.SelectConfig)
        """Which tasks, in which order, e.g. `vf.SelectConfig(shuffle=True, seed=42, limit=1000)`.
        An infinite taskset needs a `limit`."""

    def __init__(self, config: Config) -> None:
        # Parsed from JSON, as each worker parses it: an env a worker can't load fails here.
        env_config_json = config.env.model_dump_json()
        env_config = vf.resolve_env_config(json.loads(env_config_json))
        taskset = vf.load_taskset(env_config.taskset).select(config.select)
        if not taskset.bounded:
            raise ValueError(
                f"Verifiers taskset {env_config.taskset.id!r} is infinite; set select.limit"
            )
        samples = [
            VerifiersTaskSample(
                env_config_json=env_config_json,
                task_data=task.data.model_dump(mode="json"),
            )
            for task in taskset
        ]
        # TODO: skip tasks whose first prompt never fits. Today each such group runs, then
        # trains nothing. Not done here: the harness builds the first prompt, so the dataset
        # can't measure it.
        if not samples:
            raise ValueError(
                f"Verifiers taskset {env_config.taskset.id!r} has no tasks"
            )
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
