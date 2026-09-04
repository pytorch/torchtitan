# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import sys
import time

import pytest
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components import metrics
from torchtitan.components.metrics import HostMonitor, HostStats


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


class TestReduceHostStats(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @property
    def device_type(self):
        return "cpu"

    @with_comms
    def test_reduces_average_and_max_across_all_ranks(self):
        rank = dist.get_rank()
        local_stats = (
            HostStats(1.0, 3.0, 10.0) if rank == 0 else HostStats(3.0, 5.0, 30.0)
        )
        world_mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("world",),
        )

        average, maximum = metrics._reduce_host_stats(local_stats, world_mesh)

        self.assertEqual(average, HostStats(2.0, 4.0, 20.0))
        self.assertEqual(maximum, HostStats(3.0, 5.0, 30.0))
