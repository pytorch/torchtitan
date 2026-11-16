# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

from .types import Max, Mean, Metric


class EventLoopLag:
    """Measure how late this process's event loop runs work that is ready to run.

    A task sleeps `interval_s` and records how late it wakes up. It wakes late when code holds the
    loop without an `await` (a long render, a large JSON parse), or when too much work waits on it.

    Example, inside a running event loop:

        lag = EventLoopLag("rollout_worker")
        ...  # over 10 s, another task calls time.sleep(2) once, which holds the loop for 2 s
        lag.flush()
        # -> [Metric("rollout_worker/event_loop_lag_ms", Max(1980.0)),  # the longest delay
        #     Metric("rollout_worker/event_loop_lag_ms_per_s", Mean(1980.0, count=10.0))]
        #     # about 198 ms late per second
        lag.flush()
        # -> Max and Mean about 0: each flush starts over
        lag.close()

    Args:
        role: The kind of process, used as the key's first part: "rollout_worker" ->
            `rollout_worker/event_loop_lag_ms`.
        interval_s: Seconds between checks. A delay can read up to `interval_s` shorter than it was.
    """

    def __init__(self, role: str, *, interval_s: float = 0.05) -> None:
        self._key = f"{role}/event_loop_lag_ms"
        self._interval_s = interval_s
        self._loop = asyncio.get_running_loop()
        self._reset()
        self._task = self._loop.create_task(self._run(), name=f"{role}_event_loop_lag")

    def _reset(self) -> None:
        self._max_lag_s = 0.0
        self._total_lag_s = 0.0
        self._reset_time = self._loop.time()

    async def _run(self) -> None:
        while True:
            wake_time = self._loop.time() + self._interval_s
            await asyncio.sleep(self._interval_s)
            lag_s = self._loop.time() - wake_time
            self._max_lag_s = max(self._max_lag_s, lag_s)
            self._total_lag_s += lag_s

    def flush(self) -> list[Metric]:
        """Return the lag since the last flush, then reset."""
        max_lag_s, total_lag_s = self._max_lag_s, self._total_lag_s
        elapsed_s = self._loop.time() - self._reset_time
        self._reset()
        return [
            Metric(self._key, Max(1000 * max_lag_s)),
            # count = the seconds covered, so across groups and workers the Mean is ms late per second.
            Metric(f"{self._key}_per_s", Mean(1000 * total_lag_s, count=elapsed_s)),
        ]

    def close(self) -> None:
        self._task.cancel()
