# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit test for the RL Monarch actor base class, over real Monarch."""

import asyncio

import monarch.actor
from monarch.actor import (
    Actor as MonarchActor,
    concurrent_endpoint,
    endpoint,
    this_host,
)

from torchtitan.rl.actors.monarch.actor import Actor


class _Callee(Actor):
    @concurrent_endpoint
    async def reply_after(self, delay_s: float) -> None:
        await asyncio.sleep(delay_s)

    @endpoint
    async def is_alive(self) -> bool:
        return True


class _Caller(MonarchActor):
    def __init__(self, callee) -> None:
        self.callee = callee

    @endpoint
    async def start_call(self, delay_s: float) -> None:
        self.call = asyncio.ensure_future(self.callee.reply_after.call_one(delay_s))


def test_reply_to_a_stopped_caller_does_not_fail_the_callee(monkeypatch) -> None:
    """A generator replying to a rollout worker that already stopped, as at RL shutdown: with
    Monarch's default, the bounced reply fails the generator and the job exits 130."""
    faults = []
    # Record unhandled faults instead of exiting the test process.
    monkeypatch.setattr(monarch.actor, "unhandled_fault_hook", faults.append)

    async def run() -> None:
        callee_procs = this_host().spawn_procs(per_host={"procs": 1})
        caller_procs = this_host().spawn_procs(per_host={"procs": 1})
        try:
            callee = callee_procs.spawn("callee", _Callee)
            caller = caller_procs.spawn("caller", _Caller, callee)
            await caller.start_call.call_one(1.0)
            await caller.stop()
            # Let the callee reply to the stopped caller, and the reply bounce back.
            await asyncio.sleep(2.0)
            assert await callee.is_alive.call_one()
        finally:
            await caller_procs.stop()
            await callee_procs.stop()

    asyncio.run(run())
    assert faults == []
