# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc

import pytest

from torchtitan.tools.utils import GarbageCollection


@pytest.mark.parametrize("gc_freq", [0, -1])
def test_non_positive_gc_freq_raises(gc_freq: int) -> None:
    with pytest.raises(ValueError, match="gc_freq"):
        GarbageCollection(gc_freq)


@pytest.mark.parametrize("gc_freq", [1, 1000])
def test_positive_gc_freq_is_accepted(gc_freq: int) -> None:
    # __init__ disables GC and runs one collection. Restore the previous state
    # so this acceptance check does not leak into later tests.
    was_enabled = gc.isenabled()
    try:
        garbage_collection = GarbageCollection(gc_freq)
        assert garbage_collection.gc_freq == gc_freq
    finally:
        if was_enabled:
            gc.enable()
