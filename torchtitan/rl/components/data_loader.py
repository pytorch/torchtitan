# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import grain.python as grain
from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.components.data.sources import RandomAccessDataSource
from torchtitan.config import Configurable


class RLDataLoader(Stateful, Configurable):
    """Reserve globally indexed env inputs and track unacknowledged indices.

    This class is not thread-safe. Call reserve, acknowledge, state_dict, and
    load_state_dict from the controller event loop. get is pure and may run in
    a worker thread.
    """

    _STATE_VERSION = 1

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        source: Configurable.Config
        """Builds a finite RandomAccessDataSource of env inputs."""

        seed: int = 42
        shuffle: bool = True

    def __init__(self, config: Config) -> None:
        source = config.source.build()
        if not isinstance(source, RandomAccessDataSource):
            raise ValueError("RL data source must support random access")
        if len(source) == 0:
            raise ValueError("RL data source must not be empty")

        dataset = grain.MapDataset.source(source)
        if config.shuffle:
            dataset = dataset.shuffle(seed=config.seed)
        self._dataset = dataset.repeat()
        self._seed = config.seed
        self._shuffle = config.shuffle
        self._dataset_length = len(source)
        self._next_index = 0
        self._pending: set[int] = set()
        self._replay: deque[int] = deque()

    def reserve(self) -> int:
        if self._replay:
            return self._replay.popleft()
        index = self._next_index
        self._next_index += 1
        self._pending.add(index)
        return index

    def get(self, index: int) -> object:
        return self._dataset[index]

    def acknowledge(self, indices: Iterable[int]) -> None:
        replay = set(self._replay)
        for index in indices:
            if index not in self._pending or index in replay:
                raise RuntimeError(
                    f"cannot acknowledge unreserved or unreplayed index {index}"
                )
            self._pending.remove(index)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self._STATE_VERSION,
            "seed": self._seed,
            "shuffle": self._shuffle,
            "dataset_length": self._dataset_length,
            "next_index": self._next_index,
            "pending_indices": sorted(self._pending),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        expected = {
            "version": self._STATE_VERSION,
            "seed": self._seed,
            "shuffle": self._shuffle,
            "dataset_length": self._dataset_length,
        }
        for key, value in expected.items():
            if state_dict.get(key) != value:
                raise ValueError(
                    f"RL data state {key}={state_dict.get(key)!r} does not match "
                    f"the configured value {value!r}"
                )

        next_index = state_dict["next_index"]
        pending = list(state_dict["pending_indices"])
        if next_index < 0 or any(index < 0 or index >= next_index for index in pending):
            raise ValueError("RL data state contains an invalid index")
        if len(pending) != len(set(pending)):
            raise ValueError("RL data state contains duplicate pending indices")

        self._next_index = next_index
        self._pending = set(pending)
        self._replay = deque(sorted(pending))
        # TODO: Replay finished rollouts from a cache instead of regenerating
        # every pending prompt after resume.
