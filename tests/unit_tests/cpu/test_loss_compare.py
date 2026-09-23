# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from scripts.loss_compare import assert_metrics_equal


def test_assert_metrics_equal_reports_every_mismatched_metric(capsys) -> None:
    baseline_metrics = {
        "loss": {1: 1.25},
        "grad_norm": {1: 2.5},
    }
    test_metrics = {
        "loss": {1: 1.5},
        "grad_norm": {1: 3.0},
    }

    with pytest.raises(SystemExit):
        assert_metrics_equal(baseline_metrics, test_metrics)

    captured = capsys.readouterr()
    assert "Actual baseline loss values" in captured.out
    assert "1 1.25" in captured.out
    assert "Actual baseline grad_norm values" in captured.out
    assert "1 2.5" in captured.out
