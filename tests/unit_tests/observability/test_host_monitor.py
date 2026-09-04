# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import sys
import time
from types import SimpleNamespace

import pytest

from torchtitan.components import metrics
from torchtitan.components.metrics import DeviceMemStats, HostMonitor


class _RecordingLogger:
    def __init__(self) -> None:
        self.metrics: dict[str, float] = {}

    def log(self, metrics: dict[str, float], step: int) -> None:
        self.metrics = metrics


class _DeviceMemoryMonitor:
    def get_peak_stats(self) -> DeviceMemStats:
        return DeviceMemStats(1.0, 2.0, 3.0, 4.0, 5, 6)

    def reset_peak_stats(self) -> None:
        pass


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="RSS is read from /proc"
)
def test_reports_resident_memory():
    stats = HostMonitor().get_stats()

    assert stats.rss_gib > 0
    assert math.isfinite(stats.rss_gib)
    # The two come from different kernel accounting, so they are only expected
    # to agree in magnitude. A wider gap than this means ru_maxrss was scaled
    # with the wrong unit -- it is KiB on Linux but bytes on macOS.
    assert 0.1 * stats.rss_gib < stats.max_rss_gib < 10 * stats.rss_gib


def test_build_host_monitor_logs_capacity(caplog):
    with caplog.at_level(logging.INFO):
        monitor = metrics.build_host_monitor()

    assert monitor.host_capacity_gib > 0
    assert "Host capacity:" in caplog.text


def test_reports_no_resident_memory_without_proc(monkeypatch):
    monitor = HostMonitor()
    monkeypatch.setattr(monitor, "page_size", 0)

    stats = monitor.get_stats()

    assert stats.rss_gib < 0
    assert stats.max_rss_gib > 0


def test_cpu_utilization_tracks_busy_work():
    monitor = HostMonitor()
    monitor.get_stats()

    deadline = time.perf_counter() + 0.2
    while time.perf_counter() < deadline:
        pass

    assert monitor.get_stats().cpu_utilization_pct > 0


def test_cpu_utilization_is_measured_per_interval():
    monitor = HostMonitor()

    deadline = time.perf_counter() + 0.2
    while time.perf_counter() < deadline:
        pass
    busy = monitor.get_stats().cpu_utilization_pct

    time.sleep(0.2)
    idle = monitor.get_stats().cpu_utilization_pct

    # The counter resets each call, so an idle interval must not inherit the
    # CPU time burned during the previous one.
    assert idle < busy


def test_metrics_processor_logs_rank_local_host_stats():
    processor = metrics.MetricsProcessor.__new__(metrics.MetricsProcessor)
    recording_logger = _RecordingLogger()
    processor.logger = recording_logger
    processor.parallel_dims = SimpleNamespace(non_data_parallel_size=1)
    processor.device_memory_monitor = _DeviceMemoryMonitor()
    processor.host_monitor = HostMonitor()
    processor.color = metrics.utils.NoColor()
    processor.gpu_peak_flops = 1.0
    processor.ntokens_since_last_log = 1
    processor.data_loading_times = [0.0]
    processor.time_last_log = time.perf_counter() - 1.0
    processor.step_last_log = 0
    processor.num_flops_per_token = 1
    processor.has_quantization = True

    processor.log(step=1, global_avg_loss=1.0, global_max_loss=1.0, grad_norm=1.0)

    assert "cpu/process_rss(GiB)" in recording_logger.metrics
    assert "cpu/process_peak_rss(GiB)" in recording_logger.metrics
    assert "cpu/process_utilization(%)" in recording_logger.metrics
