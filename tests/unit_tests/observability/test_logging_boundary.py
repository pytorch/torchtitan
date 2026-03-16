# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for logging_boundary.py (EveryNSteps schedule)."""

import pytest

from torchtitan.observability.logging_boundary import EveryNSteps


class TestEveryNSteps:
    def test_every_n_basic(self):
        schedule = EveryNSteps(every_n=5)
        results = [schedule(s) for s in range(16)]
        # Step 0: False (excluded), step 5: True, step 10: True, step 15: True
        assert results == [
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            True,
        ]

    def test_step_0_excluded(self):
        schedule = EveryNSteps(every_n=1)
        assert schedule(0) is False
        assert schedule(1) is True

    def test_additional_steps(self):
        schedule = EveryNSteps(every_n=5, additional_steps={0, 3})
        assert schedule(0) is True  # additional
        assert schedule(3) is True  # additional
        assert schedule(5) is True  # every_n
        assert schedule(1) is False

    def test_zero_every_n_raises(self):
        with pytest.raises(ValueError, match="positive"):
            EveryNSteps(every_n=0)

    def test_negative_every_n_raises(self):
        with pytest.raises(ValueError, match="positive"):
            EveryNSteps(every_n=-1)

    def test_every_1(self):
        schedule = EveryNSteps(every_n=1)
        assert schedule(0) is False
        assert schedule(1) is True
        assert schedule(2) is True
        assert schedule(100) is True
