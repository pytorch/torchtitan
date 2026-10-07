# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
from dataclasses import fields

import pytest

from torchtitan.config.configs import TrainingConfig
from torchtitan.tools.garbage_collector import GarbageCollector
from torchtitan.training_engine import TrainingEngine


@pytest.mark.parametrize("gc_freq", [0, -1])
def test_non_positive_gc_freq_raises(gc_freq: int) -> None:
    with pytest.raises(ValueError, match="gc_freq"):
        GarbageCollector.Config(gc_freq=gc_freq)


@pytest.mark.parametrize("gc_freq", [1, 50])
def test_positive_gc_freq_is_accepted(gc_freq: int) -> None:
    # build() disables GC and runs one collection. Restore the previous state
    # so this acceptance check does not leak into later tests.
    was_enabled = gc.isenabled()
    try:
        garbage_collector = GarbageCollector.Config(gc_freq=gc_freq).build()
        assert garbage_collector.gc_freq == gc_freq
    finally:
        if was_enabled:
            gc.enable()


def test_training_engine_owns_garbage_collector_defaults() -> None:
    garbage_collector = TrainingEngine.Config().garbage_collector
    assert garbage_collector.gc_freq == 50
    assert garbage_collector.debug is False
    training_fields = {field.name for field in fields(TrainingConfig)}
    assert "gc_freq" not in training_fields
    assert "gc_debug" not in training_fields
