# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Storage adapters for Grain datasets."""

import glob
import json
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import datasets
import grain.python as grain
from datasets.distributed import split_dataset_by_node

from torchtitan.components.data.types import DatasetIterationPolicy
from torchtitan.config import Configurable


@runtime_checkable
class RandomAccessDataSource(Protocol):
    """Finite data source addressable by integer index."""

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> Any:
        ...


class SourceConfig(Protocol):
    """Builds a random-access or streaming source."""

    def build(
        self,
        *,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> RandomAccessDataSource | grain.IterDataset:
        ...


class IndexedJsonlSource(Configurable):
    """Provides random access to JSONL rows through compact byte offsets."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        patterns: tuple[str, ...]

    def __init__(
        self,
        config: Config,
        *,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> None:
        del dataset_iteration_policy
        self._paths = _file_patterns_to_paths(config.patterns)
        self._path_ids = array("I")
        self._byte_offsets = array("Q")
        # TODO(data-jsonl-sidecar): Startup rescans every JSONL file per rank and
        # worker. Build one validated offset index that all processes can memory-map.
        for path_id, path in enumerate(self._paths):
            with open(path, "rb") as file:
                while True:
                    offset = file.tell()
                    line = file.readline()
                    if not line:
                        break
                    if line.strip():
                        self._path_ids.append(path_id)
                        self._byte_offsets.append(offset)

    def __len__(self) -> int:
        return len(self._byte_offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        path = self._paths[self._path_ids[index]]
        with open(path, "rb") as file:
            file.seek(self._byte_offsets[index])
            return json.loads(file.readline())


class HuggingFaceRandomAccessSource(Configurable):
    """Provides random access to a materialized Hugging Face dataset."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        path: str
        split: str
        name: str | None = None
        revision: str | None = None
        load_dataset_kwargs: dict[str, Any] = field(default_factory=dict)

        def __post_init__(self) -> None:
            duplicated = {"split", "name", "revision", "streaming"} & (
                self.load_dataset_kwargs.keys()
            )
            if duplicated:
                raise ValueError(
                    "first-class Hugging Face fields repeated in kwargs: "
                    f"{sorted(duplicated)}"
                )

    def __init__(
        self,
        config: Config,
        *,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> None:
        del dataset_iteration_policy
        dataset = datasets.load_dataset(
            config.path,
            name=config.name,
            split=config.split,
            revision=config.revision,
            streaming=False,
            **config.load_dataset_kwargs,
        )
        if not isinstance(dataset, datasets.Dataset):
            raise TypeError(
                "random-access Hugging Face source requires one Dataset; "
                f"got {type(dataset).__qualname__}"
            )
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._dataset[index]


class HuggingFaceStreamingSource(Configurable, grain.IterDataset):
    """Provides a DP-sharded Hugging Face stream with cursor checkpointing."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        path: str
        split: str
        name: str | None = None
        revision: str | None = None
        load_dataset_kwargs: dict[str, Any] = field(default_factory=dict)

        def __post_init__(self) -> None:
            duplicated = {"split", "name", "revision", "streaming"} & (
                self.load_dataset_kwargs.keys()
            )
            if duplicated:
                raise ValueError(
                    "first-class Hugging Face fields repeated in kwargs: "
                    f"{sorted(duplicated)}"
                )

    def __init__(
        self,
        config: Config,
        *,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> None:
        super().__init__()
        dataset = datasets.load_dataset(
            config.path,
            name=config.name,
            split=config.split,
            revision=config.revision,
            streaming=True,
            **config.load_dataset_kwargs,
        )
        if not isinstance(dataset, datasets.IterableDataset):
            raise TypeError(
                "streaming Hugging Face source requires one IterableDataset; "
                f"got {type(dataset).__qualname__}"
            )
        if not hasattr(dataset, "state_dict") or not hasattr(
            dataset, "load_state_dict"
        ):
            raise TypeError(
                "Hugging Face streaming source does not support exact resume"
            )
        self._dataset = split_dataset_by_node(
            dataset,
            rank=dataset_iteration_policy.dp_rank,
            world_size=dataset_iteration_policy.dp_world_size,
        )
        self._repeat = dataset_iteration_policy.repeat
        self._shuffle = dataset_iteration_policy.shuffle

    def __iter__(self) -> grain.DatasetIterator:
        return _HuggingFaceCursorIterator(
            self._dataset,
            repeat=self._repeat,
            shuffle=self._shuffle,
        )


def _file_patterns_to_paths(patterns: tuple[str, ...]) -> tuple[str, ...]:
    """Return sorted, unique absolute paths matched by file patterns.

    Every pattern must match at least one file. Duplicate resolved paths are
    rejected so one file cannot be indexed twice.
    """
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"pattern matched no files: {pattern!r}")
        paths.extend(str(Path(match).resolve()) for match in matches)
    if len(paths) != len(set(paths)):
        raise ValueError("patterns resolve to the same file more than once")
    return tuple(paths)


class _HuggingFaceCursorIterator(grain.DatasetIterator):
    """Exposes a Hugging Face streaming cursor to Grain checkpoint recursion."""

    def __init__(
        self,
        dataset: datasets.IterableDataset,
        *,
        repeat: bool,
        shuffle: bool,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._repeat = repeat
        self._shuffle = shuffle
        self._epoch = 0
        self._initial_state = dataset.state_dict()
        self._iterator = iter(dataset)

    def __next__(self) -> dict[str, Any]:
        try:
            return next(self._iterator)
        except StopIteration:
            if not self._repeat:
                raise
            self._epoch += 1
            if self._shuffle:
                self._dataset.set_epoch(self._epoch)
            self._dataset.load_state_dict(self._initial_state)
            self._iterator = iter(self._dataset)
            return next(self._iterator)

    def get_state(self) -> dict[str, Any]:
        return {
            "epoch": self._epoch,
            "hf": self._dataset.state_dict(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._epoch = state["epoch"]
        if self._shuffle:
            self._dataset.set_epoch(self._epoch)
        self._dataset.load_state_dict(state["hf"])
        self._iterator = iter(self._dataset)
