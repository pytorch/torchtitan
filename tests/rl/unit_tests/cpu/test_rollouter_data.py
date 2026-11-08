# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from torchtitan.rl.rollout.data import RLDataset, ValidationDataLoader
from torchtitan.rl.rollout.rollouter import Rollouter


class _ListDataset(RLDataset):
    """Yields 1, 2, 3, then repeats or stops; iterating it twice continues where it stopped."""

    @dataclass(kw_only=True, slots=True)
    class Config(RLDataset.Config):
        repeat: bool

    def __init__(self, config: Config) -> None:
        samples = (1, 2, 3)
        self._iterator = itertools.cycle(samples) if config.repeat else iter(samples)

    def __iter__(self):
        return self._iterator


def _rollouter(*, repeat: bool, num_prompts: int) -> Rollouter:
    rollouter = Rollouter.__new__(Rollouter)
    rollouter._config = SimpleNamespace(
        validation_dataloader=ValidationDataLoader.Config(
            dataset=_ListDataset.Config(repeat=repeat), num_prompts=num_prompts
        )
    )
    return rollouter


def test_num_prompts_bounds_a_repeating_dataset() -> None:
    rollouter = _rollouter(repeat=True, num_prompts=5)

    # Each pass builds a fresh dataset, so the second pass starts at the first prompt too.
    assert list(rollouter.iter_validation_samples()) == [1, 2, 3, 1, 2]
    assert list(rollouter.iter_validation_samples()) == [1, 2, 3, 1, 2]


def test_dataset_that_ends_early_raises() -> None:
    prompts = _rollouter(repeat=False, num_prompts=4).iter_validation_samples()

    assert [next(prompts) for _ in range(3)] == [1, 2, 3]
    with pytest.raises(
        RuntimeError, match="ended after 3 prompts, but num_prompts is 4"
    ):
        next(prompts)
