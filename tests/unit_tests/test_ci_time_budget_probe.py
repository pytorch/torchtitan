# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time


def test_below_p90_time_budget() -> None:
    time.sleep(8)


def test_above_p90_time_budget() -> None:
    time.sleep(12)
