# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import pytest

from torchtitan.rl.components.data import (
    IterableRLDataLoader,
    MapStyleRLDataLoader,
    RLDataLoader,
    RLDataset,
)


class _StatefulDataset(RLDataset):
    @dataclass(kw_only=True, slots=True)
    class Config(RLDataset.Config):
        pass

    def __init__(self, config: Config) -> None:
        del config
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        value = self.position
        self.position += 1
        return value

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state_dict):
        self.position = state_dict["position"]


class _RowsDataset(RLDataset):
    @dataclass(kw_only=True, slots=True)
    class Config(RLDataset.Config):
        rows: tuple[int, ...]

    def __init__(self, config: Config) -> None:
        self._rows = config.rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> int:
        return self._rows[index]


def _loader(*, seed: int = 7, shuffle: bool = True, rows=tuple(range(10))):
    return MapStyleRLDataLoader.Config(
        dataset=_RowsDataset.Config(rows=rows), seed=seed, shuffle=shuffle
    ).build()


def test_next_and_acknowledge() -> None:
    loader = _loader(shuffle=False)
    assert [next(loader) for _ in range(3)] == [(0, 0), (1, 1), (2, 2)]
    loader.acknowledge([0, 2])
    assert loader.state_dict()["pending_indices"] == [1]
    with pytest.raises(RuntimeError):
        loader.acknowledge([0])


def test_state_round_trip_replays_pending_then_continues() -> None:
    loader = _loader()
    assert [next(loader)[0] for _ in range(4)] == [0, 1, 2, 3]
    loader.acknowledge([1, 3])

    restored = _loader()
    restored.load_state_dict(loader.state_dict())
    assert [next(restored)[0] for _ in range(3)] == [0, 2, 4]


@pytest.mark.parametrize(
    "key,value",
    [("version", 2), ("seed", 8), ("shuffle", False), ("dataset_length", 9)],
)
def test_load_rejects_changed_data_identity(key, value) -> None:
    loader = _loader()
    state = loader.state_dict()
    state[key] = value
    with pytest.raises(ValueError, match=key):
        loader.load_state_dict(state)


def test_replay_must_be_yielded_before_acknowledgement() -> None:
    loader = _loader()
    next(loader)
    state = loader.state_dict()
    restored = _loader()
    restored.load_state_dict(state)
    with pytest.raises(RuntimeError):
        restored.acknowledge([0])
    assert next(restored)[0] == 0
    restored.acknowledge([0])


def test_checkpoint_before_replay_preserves_unyielded_indices() -> None:
    loader = _loader()
    next(loader)
    restored = _loader()
    restored.load_state_dict(loader.state_dict())

    restored_again = _loader()
    restored_again.load_state_dict(restored.state_dict())
    assert next(restored_again)[0] == 0


def test_rejects_empty_or_non_random_access_source() -> None:
    with pytest.raises(ValueError, match="empty"):
        _loader(rows=())

    class _StreamingDataset(RLDataset):
        @dataclass(kw_only=True, slots=True)
        class Config(RLDataset.Config):
            pass

        def __init__(self, config: Config) -> None:
            del config

        def __iter__(self):
            return iter((1, 2))

    with pytest.raises(ValueError, match="random access"):
        MapStyleRLDataLoader.Config(dataset=_StreamingDataset.Config()).build()


def test_dataloader_requires_nominal_subclass() -> None:
    class _StreamingLoader:
        def __iter__(self):
            return self

        def __next__(self):
            return 0, "generated"

        def acknowledge(self, indices):
            pass

        def state_dict(self):
            return {}

        def load_state_dict(self, state_dict):
            pass

    assert not isinstance(_StreamingLoader(), RLDataLoader)


def test_iterable_loader_replays_samples_then_continues_dataset() -> None:
    loader = IterableRLDataLoader.Config(dataset=_StatefulDataset.Config()).build()
    assert [next(loader) for _ in range(3)] == [(0, 0), (1, 1), (2, 2)]
    loader.acknowledge([1])

    restored = IterableRLDataLoader.Config(dataset=_StatefulDataset.Config()).build()
    restored.load_state_dict(loader.state_dict())
    assert [next(restored) for _ in range(3)] == [(0, 0), (2, 2), (3, 3)]


def test_iterable_loader_replay_must_be_yielded_before_acknowledgement() -> None:
    loader = IterableRLDataLoader.Config(dataset=_StatefulDataset.Config()).build()
    next(loader)
    restored = IterableRLDataLoader.Config(dataset=_StatefulDataset.Config()).build()
    restored.load_state_dict(loader.state_dict())

    with pytest.raises(RuntimeError):
        restored.acknowledge([0])
    assert next(restored) == (0, 0)
