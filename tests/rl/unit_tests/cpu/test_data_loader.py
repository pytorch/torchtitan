# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import pytest

from torchtitan.rl.components.data_loader import RLDataLoader


@dataclass(frozen=True)
class _RowsConfig:
    rows: tuple[int, ...]

    def build(self):
        return self.rows


def _loader(*, seed: int = 7, shuffle: bool = True, rows=tuple(range(10))):
    return RLDataLoader.Config(
        source=_RowsConfig(rows), seed=seed, shuffle=shuffle
    ).build()


def test_reserve_get_and_acknowledge() -> None:
    loader = _loader(shuffle=False)
    assert [loader.reserve() for _ in range(3)] == [0, 1, 2]
    assert [loader.get(index) for index in range(3)] == [0, 1, 2]
    loader.acknowledge([0, 2])
    assert loader.state_dict()["pending_indices"] == [1]
    with pytest.raises(RuntimeError):
        loader.acknowledge([0])


def test_state_round_trip_replays_pending_then_continues() -> None:
    loader = _loader()
    assert [loader.reserve() for _ in range(4)] == [0, 1, 2, 3]
    loader.acknowledge([1, 3])

    restored = _loader()
    restored.load_state_dict(loader.state_dict())
    assert [restored.reserve() for _ in range(3)] == [0, 2, 4]


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


def test_replay_must_be_reserved_before_acknowledgement() -> None:
    loader = _loader()
    loader.reserve()
    state = loader.state_dict()
    restored = _loader()
    restored.load_state_dict(state)
    with pytest.raises(RuntimeError):
        restored.acknowledge([0])
    assert restored.reserve() == 0
    restored.acknowledge([0])


def test_rejects_empty_or_non_random_access_source() -> None:
    with pytest.raises(ValueError, match="empty"):
        _loader(rows=())

    @dataclass(frozen=True)
    class _StreamingConfig:
        def build(self):
            return iter((1, 2))

    with pytest.raises(ValueError, match="random access"):
        RLDataLoader.Config(source=_StreamingConfig()).build()
