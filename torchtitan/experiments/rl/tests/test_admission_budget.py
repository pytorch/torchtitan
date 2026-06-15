# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `torchtitan.experiments.rl.admission_budget.AdmissionBudget`."""

from __future__ import annotations

import asyncio

import pytest

from torchtitan.experiments.rl.admission_budget import AdmissionBudget


def test_unbounded_never_blocks():
    async def main():
        budget = AdmissionBudget(None)
        for _ in range(100):
            await budget.acquire()  # never blocks when unbounded
        budget.release(5)  # no-op

    asyncio.run(main())


def test_blocks_when_exhausted_then_unblocks_on_release():
    async def main():
        budget = AdmissionBudget(2)
        await budget.acquire()
        await budget.acquire()  # budget now exhausted
        waiter = asyncio.create_task(budget.acquire())
        await asyncio.sleep(0.01)
        assert not waiter.done()  # blocked: no permits
        budget.release()  # free one permit
        await asyncio.wait_for(waiter, timeout=1.0)  # the waiter is admitted
        assert waiter.done()

    asyncio.run(main())


def test_rejects_invalid_budget():
    with pytest.raises(ValueError):
        AdmissionBudget(0)
