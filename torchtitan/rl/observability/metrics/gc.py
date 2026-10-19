# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import time

from .types import Max, Metric, Sum


class GCTimer:
    """Time this process spends in Python's garbage collector; flush() drains it.

    One per process that logs metrics: the controller flushes once per step, a rollout worker
    once per finished group, the generation server once per completion. `per_generation` adds
    the split by GC generation (the controller's, where full collections used to stall).

    Example:
        gc_timer = GCTimer("controller", per_generation=True)  # registers a gc callback
        ...                                                     # collections run during the step
        gc_timer.flush()
        # -> [Metric("perf/controller/gc_seconds", Sum(1.9)),    # every collection since the last flush
        #     Metric("perf/controller/gc_seconds", Max(1.2)),    # the longest pause
        #     ...,                                               # gen0, gen1: the same 4 keys as below
        #     Metric("controller/gc/gen2/collections", Sum(2)),  # full collections
        #     Metric("controller/gc/gen2/seconds", Sum(1.8)),    # time in them
        #     Metric("controller/gc/gen2/seconds", Max(1.2)),    # the longest one
        #     Metric("controller/gc/gen2/collected", Sum(310))]  # objects they freed
        gc_timer.close()
    """

    def __init__(self, role: str, *, per_generation: bool = False) -> None:
        self._role = role
        self._per_generation = per_generation
        self._start = time.perf_counter()
        self._reset()
        gc.callbacks.append(self._on_gc)

    def _reset(self) -> None:
        # One entry per GC generation: 0, 1, 2.
        self._collections = [0, 0, 0]
        self._total_s = [0.0, 0.0, 0.0]
        self._max_s = [0.0, 0.0, 0.0]
        self._collected = [0, 0, 0]

    def _on_gc(self, phase: str, info: dict) -> None:
        if phase == "start":
            self._start = time.perf_counter()
            return
        # Wall time: if another thread takes the GIL during these callbacks, its run time counts too.
        seconds = time.perf_counter() - self._start
        generation = info["generation"]
        self._collections[generation] += 1
        self._total_s[generation] += seconds
        self._max_s[generation] = max(self._max_s[generation], seconds)
        self._collected[generation] += info["collected"]

    def flush(self) -> list[Metric]:
        """Return the GC metrics since the last flush, then reset."""
        collections, total_s, max_s, collected = (
            self._collections,
            self._total_s,
            self._max_s,
            self._collected,
        )
        self._reset()
        metrics = [
            Metric(f"perf/{self._role}/gc_seconds", Sum(sum(total_s))),
            Metric(f"perf/{self._role}/gc_seconds", Max(max(max_s))),
        ]
        if not self._per_generation:
            return metrics
        # Not under perf/: the console prints every perf/ key.
        for generation in range(3):
            prefix = f"{self._role}/gc/gen{generation}"
            metrics += [
                Metric(f"{prefix}/collections", Sum(collections[generation])),
                Metric(f"{prefix}/seconds", Sum(total_s[generation])),
                Metric(f"{prefix}/seconds", Max(max_s[generation])),
                Metric(f"{prefix}/collected", Sum(collected[generation])),
            ]
        return metrics

    def close(self) -> None:
        """Unregister the gc callback; a second call does nothing."""
        if self._on_gc in gc.callbacks:
            gc.callbacks.remove(self._on_gc)
