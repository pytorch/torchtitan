# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from types import SimpleNamespace

from torchtitan.observability import metrics
from torchtitan.observability.metrics import (
    BaseLogger,
    DeviceMemStats,
    MetricsProcessor,
)


class _FakeMemoryMonitor:
    device_name = "unused"

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

    class RecordingLogger(BaseLogger):
        @dataclass(kw_only=True, slots=True)
        class Config(BaseLogger.Config):
            pass

        def __init__(self, config, *, log_dir, config_dict, tag):
            self.closed = False

        def log(self, metrics, step):
            logged.update(metrics)

        def close(self):
            self.closed = True

    monkeypatch.setattr("torch.distributed.get_rank", lambda: 0)
    monkeypatch.setattr(metrics.utils, "get_peak_flops", lambda _: 1.0)
    processor = MetricsProcessor.Config(loggers=[RecordingLogger.Config()]).build(
        parallelism_context=SimpleNamespace(pp_enabled=False, non_data_parallel_size=1),
        device_memory_monitor=_FakeMemoryMonitor(),
    )
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
    assert processor.device_memory_monitor.resets == 2
    assert processor.step_last_log == 10

    processor.close()
    assert processor.loggers[0].closed
