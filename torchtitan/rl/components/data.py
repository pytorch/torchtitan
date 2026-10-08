# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import grain.python as grain
from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.components.data.sources import RandomAccessDataSource
from torchtitan.config import Configurable


class RLDataset(Configurable, ABC):
    """Base class for datasets that emit RL environment inputs."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass


class RLDataLoader(Stateful, ABC, Configurable):
    """Checkpointable iterator of globally identified RL inputs.

    Subclasses may generate inputs, stream them, or mix sources. They must
    restore every yielded but unacknowledged input after a checkpoint load.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        dataset: RLDataset.Config
        """Builds the dataset consumed by this dataloader."""

    @abstractmethod
    def __iter__(self) -> RLDataLoader:
        ...

    @abstractmethod
    def __next__(self) -> tuple[int, object]:
        ...

    @abstractmethod
    def acknowledge(self, indices: Iterable[int]) -> None:
        ...


class IterableRLDataLoader(RLDataLoader):
    """Checkpointable RL loader backed by a stateful iterable dataset.

    Unacknowledged samples are stored in the checkpoint and replayed before new
    samples after resume. The wrapped dataset remains responsible for its own
    cursor, RNG, and any external source state.
    """

    _STATE_VERSION = 1
    """Increment when the serialized fields or their interpretation changes."""

    @dataclass(kw_only=True, slots=True)
    class Config(RLDataLoader.Config):
        pass

    def __init__(self, config: Config) -> None:
        dataset = config.dataset.build()
        if not isinstance(dataset, Iterable) or not isinstance(dataset, Stateful):
            raise ValueError("RL dataset must be a stateful iterable")
        self._dataset: Stateful = dataset
        self._iterator: Iterator[object] = iter(dataset)
        self._next_index = 0
        self._pending: dict[int, object] = {}
        self._replay: deque[tuple[int, object]] = deque()

    def __iter__(self) -> IterableRLDataLoader:
        return self

    def __next__(self) -> tuple[int, object]:
        if self._replay:
            index, sample = self._replay.popleft()
        else:
            index = self._next_index
            sample = next(self._iterator)
            self._next_index += 1
        self._pending[index] = sample
        return index, sample

    def acknowledge(self, indices: Iterable[int]) -> None:
        for index in indices:
            if index not in self._pending:
                raise RuntimeError(f"cannot acknowledge an unyielded index {index}")
            del self._pending[index]

    def state_dict(self) -> dict[str, Any]:
        replay = dict(self._replay)
        replay.update(self._pending)
        return {
            "version": self._STATE_VERSION,
            "dataset": self._dataset.state_dict(),
            "next_index": self._next_index,
            "pending_samples": sorted(replay.items()),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict.get("version") != self._STATE_VERSION:
            raise ValueError(
                f"unsupported RL data state version {state_dict.get('version')!r}"
            )
        next_index = state_dict["next_index"]
        pending = list(state_dict["pending_samples"])
        pending_indices = [index for index, _ in pending]
        if next_index < 0 or any(
            index < 0 or index >= next_index for index in pending_indices
        ):
            raise ValueError("RL data state contains an invalid index")
        if len(pending_indices) != len(set(pending_indices)):
            raise ValueError("RL data state contains duplicate pending indices")

        self._dataset.load_state_dict(state_dict["dataset"])
        self._iterator = iter(self._dataset)
        self._next_index = next_index
        self._pending = {}
        self._replay = deque(sorted(pending))


class MapStyleRLDataLoader(RLDataLoader):
    """Checkpointable RL input iterator backed by a random-access source.

    This class is not thread-safe. Call next, acknowledge, state_dict, and
    load_state_dict serially from the controller data-input thread.
    """

    _STATE_VERSION = 1
    """Increment when the serialized fields or their interpretation changes."""

    @dataclass(kw_only=True, slots=True)
    class Config(RLDataLoader.Config):
        seed: int = 42
        shuffle: bool = True

    def __init__(self, config: Config) -> None:
        dataset_source = config.dataset.build()
        if not isinstance(dataset_source, RandomAccessDataSource):
            raise ValueError("RL dataset must support random access")
        if len(dataset_source) == 0:
            raise ValueError("RL dataset must not be empty")

        dataset = grain.MapDataset.source(dataset_source)
        if config.shuffle:
            dataset = dataset.shuffle(seed=config.seed)
        self._dataset = dataset.repeat()
        self._seed = config.seed
        self._shuffle = config.shuffle
        self._dataset_length = len(dataset_source)
        self._next_index = 0
        self._pending: set[int] = set()
        self._replay: deque[int] = deque()

    def __iter__(self) -> MapStyleRLDataLoader:
        return self

    def __next__(self) -> tuple[int, object]:
        if self._replay:
            index = self._replay.popleft()
        else:
            index = self._next_index
            self._next_index += 1
        self._pending.add(index)
        return index, self._dataset[index]

    def acknowledge(self, indices: Iterable[int]) -> None:
        for index in indices:
            if index not in self._pending:
                raise RuntimeError(f"cannot acknowledge an unyielded index {index}")
            self._pending.remove(index)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self._STATE_VERSION,
            "seed": self._seed,
            "shuffle": self._shuffle,
            "dataset_length": self._dataset_length,
            "next_index": self._next_index,
            "pending_indices": sorted(self._pending | set(self._replay)),
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
        self._pending = set()
        self._replay = deque(sorted(pending))
        # TODO: Replay finished rollouts from a cache instead of regenerating
        # every pending prompt after resume.
