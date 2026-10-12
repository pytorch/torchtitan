# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

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
