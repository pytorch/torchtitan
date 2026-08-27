# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.tools.profiler import Profiler


def test_profiler_config_rejects_profile_period_shorter_than_cycle() -> None:
    with pytest.raises(
        ValueError,
        match="profiler.profile_freq must be greater than or equal to profiler_warmup \\+ profiler_active",
    ):
        Profiler.Config(enable_profiling=True, profile_freq=3)


def test_profiler_config_allows_default_schedule() -> None:
    config = Profiler.Config(enable_profiling=True)
    assert config.profile_freq >= config.profiler_warmup + config.profiler_active


def test_profiler_config_does_not_validate_unused_schedule() -> None:
    assert Profiler.Config(enable_profiling=False, profile_freq=0).profile_freq == 0
