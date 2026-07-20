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

from torchtitan.config import Configurable


@runtime_checkable
class RandomAccessSource(Protocol):
    """A finite source addressable by integer index."""

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> Any:
        ...


class SourceConfig(Protocol):
    """Builds a random-access or streaming source."""

    def build(
        self, *, dp_rank: int, dp_world_size: int
    ) -> RandomAccessSource | grain.IterDataset:
        ...


def _expand_patterns(patterns: tuple[str, ...]) -> tuple[str, ...]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"pattern matched no files: {pattern!r}")
        paths.extend(str(Path(match).resolve()) for match in matches)
    if len(paths) != len(set(paths)):
        raise ValueError("patterns resolve to the same file more than once")
    return tuple(paths)


class IndexedJsonlSource(Configurable):
    """Provides random access to JSONL rows through compact byte offsets."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        patterns: tuple[str, ...]

    def __init__(self, config: Config, *, dp_rank: int, dp_world_size: int) -> None:
        del dp_rank, dp_world_size
        self._paths = _expand_patterns(config.patterns)
        self._path_ids = array("I")
        self._byte_offsets = array("Q")
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
        load_dataset_kwargs: dict[str, Any] = field(default_factory=dict)

    def __init__(self, config: Config, *, dp_rank: int, dp_world_size: int) -> None:
        del dp_rank, dp_world_size
        self._dataset = datasets.load_dataset(
            config.path, streaming=False, **config.load_dataset_kwargs
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._dataset[index]


class _HuggingFaceCursorIterator(grain.DatasetIterator):
    """Exposes a Hugging Face streaming cursor to Grain checkpoint recursion."""

    def __init__(self, dataset: Any) -> None:
        super().__init__()
        self._dataset = dataset
        self._iterator = iter(dataset)

    def __next__(self) -> dict[str, Any]:
        return next(self._iterator)

    def get_state(self) -> dict[str, Any]:
        return {"hf": self._dataset.state_dict()}

    def set_state(self, state: dict[str, Any]) -> None:
        self._dataset.load_state_dict(state["hf"])
        self._iterator = iter(self._dataset)


class HuggingFaceStreamingSource(Configurable, grain.IterDataset):
    """Provides a DP-sharded Hugging Face stream with exact cursor resume."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        path: str
        load_dataset_kwargs: dict[str, Any] = field(default_factory=dict)

    def __init__(self, config: Config, *, dp_rank: int, dp_world_size: int) -> None:
        super().__init__()
        dataset = datasets.load_dataset(
            config.path, streaming=True, **config.load_dataset_kwargs
        )
        if not hasattr(dataset, "state_dict") or not hasattr(
            dataset, "load_state_dict"
        ):
            raise TypeError(
                "Hugging Face streaming source does not support exact resume"
            )
        self._dataset = split_dataset_by_node(
            dataset,
            rank=dp_rank,
            world_size=dp_world_size,
        )

    def __iter__(self) -> grain.DatasetIterator:
        return _HuggingFaceCursorIterator(self._dataset)
