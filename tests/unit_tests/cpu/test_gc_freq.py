# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc

import pytest

from torchtitan.tools.garbage_collector import GarbageCollector


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


def test_freeze_moves_survivors_to_the_permanent_generation() -> None:
    try:
        GarbageCollector.freeze("test")
        assert gc.get_freeze_count() > 0
    finally:
        gc.unfreeze()
