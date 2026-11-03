# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""An event loop whose clock jumps to the next timer once nothing else can run. A test
then waits out at once every shorter timer that a wrong implementation could start."""

import asyncio
import selectors
from typing import Any


class _JumpingSelector(selectors.DefaultSelector):
    """Polls without blocking and, when nothing is ready, moves ``clock`` on by the time
    its loop would have waited for the next timer."""

    def __init__(self, clock: list[float]) -> None:
        super().__init__()
        self._clock = clock

    def select(self, timeout: float | None = None) -> list[Any]:
        events = super().select(0)
        if not events and timeout:
            self._clock[0] += timeout
        return events


class VirtualClockLoop(asyncio.SelectorEventLoop):
    """An event loop whose clock jumps to the next timer once no callback is ready. A
    wait of any length thus ends at once and after every shorter one. Only for code that
    waits on no thread or process, whose replies would not stop the clock."""

    def __init__(self) -> None:
        self._clock = [0.0]
        super().__init__(_JumpingSelector(self._clock))

    def time(self) -> float:
        return self._clock[0]
