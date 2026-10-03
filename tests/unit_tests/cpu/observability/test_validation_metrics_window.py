# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

from torchtitan.observability import metrics
from torchtitan.observability.metrics import DeviceMemStats, MetricsProcessor
from torchtitan.tools.utils import NoColor


class _FakeMemoryMonitor:
    def __init__(self) -> None:
        self.resets = 0

    def reset_peak_stats(self) -> None:
        self.resets += 1

    def get_peak_stats(self) -> DeviceMemStats:
        return DeviceMemStats(0, 0.0, 0, 0.0, 0, 0)


def test_validation_is_measured_on_its_own_window(monkeypatch) -> None:
    now = [100.0]
    monkeypatch.setattr(metrics.time, "perf_counter", lambda: now[0])
    logged = {}

    processor = object.__new__(MetricsProcessor)
    processor.parallelism_context = SimpleNamespace(non_data_parallel_size=1)
    processor.color = NoColor()
    processor.logger = SimpleNamespace(log=lambda m, step: logged.update(m))
    processor.device_memory_monitor = _FakeMemoryMonitor()
    # 50 s of training (and checkpointing) since the last training log.
    processor.ntokens_since_last_log = 5000
    processor.data_loading_times = [0.5, 0.5]
    processor.time_last_log = 50.0
    processor.step_last_log = 8

    processor.reset()
    processor.ntokens_since_last_log += 800  # what the validator counts
    now[0] = 104.0
    processor.log_validation(loss=1.0, step=10)

    assert logged["validation_metrics/throughput(tps)"] == 200
    assert processor.data_loading_times == []
    assert processor.device_memory_monitor.resets == 1
    assert processor.step_last_log == 10
