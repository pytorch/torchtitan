# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace

from torchtitan.rl.rollout.data import IterableRLDataLoader, RLDataset
from torchtitan.rl.rollout.rollouter import Rollouter


class _ValidationConfig:
    def __init__(self, values: tuple[int, ...]) -> None:
        self.values = values
        self.num_builds = 0

    def build(self):
        self.num_builds += 1
        return iter(self.values)


def _rollouter(validation_dataset: _ValidationConfig) -> Rollouter:
    rollouter = Rollouter.__new__(Rollouter)
    rollouter._config = SimpleNamespace(validation_dataset=validation_dataset)
    return rollouter


def test_validation_steps_bound_iterable_source() -> None:
    config = _ValidationConfig((1, 2, 3))
    rollouter = _rollouter(config)

    assert rollouter.get_validation_samples(2) == [1, 2]
    assert rollouter.get_validation_samples(5) == [1, 2, 3]
    assert config.num_builds == 2


def test_validation_minus_one_consumes_one_finite_pass() -> None:
    rollouter = _rollouter(_ValidationConfig((1, 2, 3)))

    assert rollouter.get_validation_samples(-1) == [1, 2, 3]


def test_state_dict_waits_for_in_flight_training_sample() -> None:
    advanced = threading.Event()

    class _SlowDataset(RLDataset):
        @dataclass(kw_only=True, slots=True)
        class Config(RLDataset.Config):
            pass

        def __init__(self, config: Config) -> None:
            del config
            self.position = 0

        def __iter__(self):
            return self

        def __next__(self):
            # Advance first, then stay inside __next__ long enough for a checkpoint.
            value = self.position
            self.position += 1
            advanced.set()
            time.sleep(0.1)
            return value

        def state_dict(self):
            return {"position": self.position}

        def load_state_dict(self, state_dict):
            self.position = state_dict["position"]

    config = SimpleNamespace(
        training_dataloader=IterableRLDataLoader.Config(dataset=_SlowDataset.Config())
    )
    rollouter = Rollouter(config)
    reader = threading.Thread(target=rollouter.get_training_sample)
    reader.start()
    # Save state while the reader thread is inside __next__.
    assert advanced.wait(timeout=10)
    state = rollouter.state_dict()
    reader.join(timeout=10)

    restored = Rollouter(config)
    restored.load_state_dict(state)
    assert [restored.get_training_sample() for _ in range(2)] == [(0, 0), (1, 1)]
