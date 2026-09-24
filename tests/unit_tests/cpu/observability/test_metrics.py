# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from typing import Any, cast

import pytest

from torchtitan.distributed import ParallelDims
from torchtitan.observability.metrics import (
    BaseLogger,
    DeviceMemoryMonitor,
    DeviceMemStats,
    MetricsProcessor,
)


class _MemoryMonitor:
    device_name = "test device"

    def __init__(self) -> None:
        self.num_resets = 0

    def get_peak_stats(self) -> DeviceMemStats:
        return DeviceMemStats(1.0, 2.0, 3.0, 4.0, 0, 0)

    def reset_peak_stats(self) -> None:
        self.num_resets += 1


class _CapturingLogger(BaseLogger):
    def __init__(self) -> None:
        self.metrics: dict[str, Any] | None = None

    def log(self, metrics: dict[str, Any], step: int) -> None:
        del step
        self.metrics = metrics


def _processor(monkeypatch: pytest.MonkeyPatch) -> MetricsProcessor:
    monkeypatch.setattr(
        "torchtitan.observability.metrics.utils.get_peak_flops",
        lambda device_name: 1000,
    )
    processor = MetricsProcessor(
        MetricsProcessor.Config(disable_color_printing=True),
        parallel_dims=cast(ParallelDims, SimpleNamespace(non_data_parallel_size=2)),
        device_memory_monitor=cast(DeviceMemoryMonitor, _MemoryMonitor()),
    )
    processor.logger = _CapturingLogger()
    return processor


def test_training_log_reports_passed_flops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = _processor(monkeypatch)
    processor.ntokens_since_last_log = 20
    processor.data_loading_times.append(0.25)
    processor.step_last_log = 0
    processor.time_last_log = 8.0
    times = iter((10.0, 10.0))
    monkeypatch.setattr(
        "torchtitan.observability.metrics.time.perf_counter", lambda: next(times)
    )

    processor.log(
        step=1,
        global_avg_loss=1.0,
        global_max_loss=2.0,
        grad_norm=3.0,
        num_flops=4000,
    )

    logger = processor.logger
    assert isinstance(logger, _CapturingLogger)
    assert logger.metrics is not None
    assert logger.metrics["tflops"] == 1e-9
    assert logger.metrics["mfu(%)"] == 100.0


def test_validation_log_resets_training_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = _processor(monkeypatch)
    processor.ntokens_since_last_log = 20
    processor.data_loading_times.append(0.25)
    processor.step_last_log = 0
    processor.time_last_log = 8.0
    times = iter((10.0, 10.0))
    monkeypatch.setattr(
        "torchtitan.observability.metrics.time.perf_counter", lambda: next(times)
    )

    processor.log_validation(loss=1.0, step=1)

    assert processor.ntokens_since_last_log == 0
