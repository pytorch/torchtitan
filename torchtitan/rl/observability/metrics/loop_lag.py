# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import time

from .types import Max, Mean, Metric

logger = logging.getLogger(__name__)


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


class ThreadPoolLag:
    """Measure how late this process's thread pool starts an `asyncio.to_thread` call.

    A task submits an empty call every `interval_s` and records how late a thread starts it. It
    starts late when every thread is busy (e.g. held by blocking calls), or when other threads hold
    the GIL. High here but not in `EventLoopLag`: too few threads. High in both: the GIL or the loop.

    Example, a pool of 1 thread, inside a running event loop:

        lag = ThreadPoolLag("rollout_worker")
        ...  # over 10 s, another task calls asyncio.to_thread(time.sleep, 2) once
        lag.flush()
        # -> [Metric("rollout_worker/thread_pool_lag_ms", Max(1980.0)),  # the longest delay
        #     Metric("rollout_worker/thread_pool_lag_ms_per_s", Mean(1980.0, count=10.0))]
        #     # about 198 ms per second with no free thread
        lag.close()

    Args:
        role: The kind of process, used as the key's first part: "rollout_worker" ->
            `rollout_worker/thread_pool_lag_ms`.
        interval_s: Seconds between checks.
        warn_after_s: A check that waits this long logs a warning. A stuck pool may stop every
            group, and with it the metrics that groups carry.
    """

    def __init__(
        self, role: str, *, interval_s: float = 0.05, warn_after_s: float = 60.0
    ) -> None:
        self._key = f"{role}/thread_pool_lag_ms"
        self._interval_s = interval_s
        self._warn_after_s = warn_after_s
        # When the waiting check was submitted, or None between checks.
        self._submit_time: float | None = None
        self._reset()
        self._task = asyncio.get_running_loop().create_task(
            self._run(), name=f"{role}_thread_pool_lag"
        )

    def _reset(self) -> None:
        self._max_lag_s = 0.0
        self._total_lag_s = 0.0
        self._reset_time = time.monotonic()

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(self._interval_s)
            self._submit_time = time.monotonic()
            # Submitted now, and the thread reads the start time, so the loop's own lag isn't counted.
            check = loop.run_in_executor(None, time.monotonic)
            done, _ = await asyncio.wait({check}, timeout=self._warn_after_s)
            if not done:
                logger.warning(
                    "No free thread for an asyncio.to_thread call in %.0f s: every thread "
                    "may be held by a blocking call",
                    self._warn_after_s,
                )
            start_time = await check
            self._max_lag_s = max(self._max_lag_s, start_time - self._submit_time)
            # Only the wait since the last flush: a flush during the wait counted the part before it.
            waited_since_flush_s = start_time - max(self._submit_time, self._reset_time)
            self._total_lag_s += max(0.0, waited_since_flush_s)
            self._submit_time = None

    def flush(self) -> list[Metric]:
        """Return the lag since the last flush, then reset. A check still waiting counts its wait so far."""
        now = time.monotonic()
        max_lag_s, total_lag_s = self._max_lag_s, self._total_lag_s
        if self._submit_time is not None:
            max_lag_s = max(max_lag_s, now - self._submit_time)
            total_lag_s += now - max(self._submit_time, self._reset_time)
        elapsed_s = now - self._reset_time
        self._reset()
        return [
            Metric(self._key, Max(1000 * max_lag_s)),
            # count = the seconds covered, so across groups and workers the Mean is ms late per second.
            Metric(f"{self._key}_per_s", Mean(1000 * total_lag_s, count=elapsed_s)),
        ]

    def close(self) -> None:
        self._task.cancel()
