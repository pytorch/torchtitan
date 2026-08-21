# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.components.metrics import MetricsProcessor


@pytest.mark.parametrize("log_freq", [0, -1])
def test_metrics_config_rejects_non_positive_log_frequency(log_freq: int) -> None:
    with pytest.raises(ValueError, match="metrics.log_freq must be greater than 0"):
        MetricsProcessor.Config(log_freq=log_freq)


def test_metrics_config_accepts_positive_log_frequency() -> None:
    assert MetricsProcessor.Config(log_freq=3).log_freq == 3
