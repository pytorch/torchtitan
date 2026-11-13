# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import torch

from torchtitan.observability import metrics
from torchtitan.observability.logging import NoColor
from torchtitan.observability.metrics import DeviceMemStats, MetricsProcessor


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
    monkeypatch.setattr(metrics, "device_type", "cpu")
    logged = {}

    processor = object.__new__(MetricsProcessor)
    processor.parallelism_context = SimpleNamespace(non_data_parallel_size=1)
    processor.color = NoColor()
    processor.loggers = [SimpleNamespace(log=lambda m, step: logged.update(m))]
    processor.device_memory_monitor = _FakeMemoryMonitor()
    # 50 s of training (and checkpointing) since the last training log.
    processor.ntokens_since_last_log = 5000
    processor.data_loading_times = [0.5, 0.5]
    processor._window_start = 50.0
    processor.step_last_log = 8

    processor.reset()
    processor.ntokens_since_last_log += 800  # what the validator counts
    now[0] = 104.0
    processor.log_validation(loss=1.0, step=10)

    assert logged["validation_metrics/throughput(tps)"] == 200
    assert processor.data_loading_times == []
    assert processor.device_memory_monitor.resets == 1
    assert processor.step_last_log == 10


def test_async_log_uses_the_window_at_log_time(monkeypatch) -> None:
    """The next step fetches data before the flush; that must not count."""
    now = [100.0]
    monkeypatch.setattr(metrics.time, "perf_counter", lambda: now[0])
    monkeypatch.setattr(metrics, "device_type", "cpu")
    logged = {}

    processor = object.__new__(MetricsProcessor)
    processor.parallelism_context = SimpleNamespace(non_data_parallel_size=1)
    processor.color = NoColor()
    processor.loggers = [SimpleNamespace(log=lambda m, step: logged.update(m))]
    processor.device_memory_monitor = _FakeMemoryMonitor()
    processor.gpu_peak_flops = None
    processor.has_quantization = False
    processor.ntokens_since_last_log = 0
    processor.data_loading_times = []
    processor.step_last_log = 0
    processor._pending = None

    processor.reset()
    processor.ntokens_since_last_log += 800
    processor.data_loading_times.append(1.0)
    now[0] = 104.0
    # Under PP, ranks without the last stage log a (1,)-shaped loss.
    processor.log_async(
        1,
        torch.tensor([2.5]),
        torch.tensor([3.0]),
        torch.tensor(0.5),
        {"n_tokens_seen": torch.tensor(800)},
        num_flops=0.0,
    )
    assert logged == {}

    # The next step starts a window and fetches its data before the flush.
    processor.reset()
    processor.ntokens_since_last_log += 800
    now[0] = 110.0
    processor.flush()

    assert logged["throughput(tps)"] == 200
    assert logged["time_metrics/data_loading(%)"] == 25
    assert logged["loss_metrics/global_avg_loss"] == 2.5
    assert logged["loss_metrics/global_max_loss"] == 3.0
    assert logged["grad_norm"] == 0.5
    assert logged["n_tokens_seen"] == 800
    assert processor.step_last_log == 1
