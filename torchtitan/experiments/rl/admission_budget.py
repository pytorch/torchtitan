# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pre-generation admission gate: bounds how many rollout groups are in flight ahead of the trainer."""

from __future__ import annotations

import asyncio


class AdmissionBudget:
    """Bounds the rollout groups admitted for generation before the trainer consumes them.

    A worker `acquire()`s a permit before generating a group; the permit is `release()`d when the
    group's episodes leave the buffer (consumed or dropped). So generation runs at most
    `max_active_rollout_groups` groups ahead of training — the born-stale bound at scale. `None`
    leaves it unbounded (no gate).

    Example:

        budget = AdmissionBudget(max_active_rollout_groups=32)
        await budget.acquire()   # blocks once 32 groups are in flight
        # ... generate + add the group to the buffer ...
        budget.release()         # the group's round drained from the buffer
    """

    def __init__(self, max_active_rollout_groups: int | None) -> None:
        if max_active_rollout_groups is not None and max_active_rollout_groups < 1:
            raise ValueError(
                f"max_active_rollout_groups must be >= 1 or None, got {max_active_rollout_groups}"
            )
        self._semaphore = (
            asyncio.Semaphore(max_active_rollout_groups)
            if max_active_rollout_groups is not None
            else None
        )

    async def acquire(self) -> None:
        """Take one group permit, blocking if the budget is exhausted (no-op when unbounded)."""
        if self._semaphore is not None:
            await self._semaphore.acquire()

    def release(self, count: int = 1) -> None:
        """Return `count` group permits (no-op when unbounded)."""
        if self._semaphore is None:
            return
        for _ in range(count):
            self._semaphore.release()
