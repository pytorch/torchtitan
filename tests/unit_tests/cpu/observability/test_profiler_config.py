# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.observability.kernel_spans import KernelSpanRecorder
from torchtitan.observability.profiler import (
    CuspyTraceProfiler,
    KinetoTraceProfiler,
    MemorySnapshotProfiler,
    Profiler,
)


def test_trace_profiler_rejects_freq_shorter_than_cycle() -> None:
    with pytest.raises(
        ValueError,
        match=r"freq \(3\) must be greater than or equal to warmup \+ active",
    ):
        KinetoTraceProfiler.Config(freq=3)


def test_cuspy_trace_profiler_validates_schedule() -> None:
    with pytest.raises(ValueError, match="must be greater than or equal to"):
        CuspyTraceProfiler.Config(freq=3)


def test_trace_profiler_allows_default_schedule() -> None:
    config = KinetoTraceProfiler.Config()
    assert config.freq >= config.warmup + config.active


def test_memory_snapshot_freq_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be greater than zero"):
        MemorySnapshotProfiler.Config(freq=0)


def test_profiler_rejects_kineto_with_kernel_spans() -> None:
    with pytest.raises(ValueError, match="Kineto cannot profile CUDA"):
        Profiler.Config(
            trace=KinetoTraceProfiler.Config(),
            kernel_spans=KernelSpanRecorder.Config(),
        )
