# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

from .types import Max, Metric, Sum


class EventLoopLag:
    """Measure how long sync code blocks this process's event loop; flush() drains it.

    A task sleeps `interval_s` and records how late it wakes: a late wake means something held
    the loop (a sync call, a long GC pause). Create it on the running loop and flush it where the
    process logs metrics, as with `GCTimer`.

    Example:
        lag = EventLoopLag("controller")  # needs a running event loop
        ...                                # another task calls time.sleep(2) on this loop
        lag.flush()
        # -> [Metric("perf/controller/event_loop_lag_ms", Max(1950.0)),  # the longest block
        #     Metric("perf/controller/event_loop_lag_ms", Sum(1950.0))]  # blocks >= threshold, summed
        lag.close()

    Args:
        role: Metric key infix, e.g. "controller" -> `perf/controller/event_loop_lag_ms`.
        interval_s: Seconds between checks; a block is undercounted by up to this.
        threshold_s: Smallest lag added to the Sum, so scheduling noise doesn't accumulate.
    """

    def __init__(
        self, role: str, *, interval_s: float = 0.05, threshold_s: float = 0.1
    ) -> None:
        self._key = f"perf/{role}/event_loop_lag_ms"
        self._interval_s = interval_s
        self._threshold_s = threshold_s
        self._reset()
        self._task = asyncio.get_running_loop().create_task(
            self._run(), name=f"{role}_event_loop_lag"
        )

    def _reset(self) -> None:
        self._max_s = 0.0
        self._blocked_s = 0.0

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            due = loop.time() + self._interval_s
            await asyncio.sleep(self._interval_s)
            lag_s = loop.time() - due
            self._max_s = max(self._max_s, lag_s)
            if lag_s >= self._threshold_s:
                self._blocked_s += lag_s

    def flush(self) -> list[Metric]:
        """Return the lag since the last flush, then reset."""
        max_s, blocked_s = self._max_s, self._blocked_s
        self._reset()
        return [
            Metric(self._key, Max(max_s * 1000)),
            Metric(self._key, Sum(blocked_s * 1000)),
        ]

    def close(self) -> None:
        self._task.cancel()
