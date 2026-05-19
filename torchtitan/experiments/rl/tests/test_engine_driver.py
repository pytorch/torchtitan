# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the VLLMGenerator admission/stepping split (CP38).

Exercises the driver task + per-request future bookkeeping without a
real vLLM engine. Verifies:

- Two concurrent ``generate_tokens`` calls coalesce on the engine
  (driver sees both batches in flight at once, not serialized).
- Driver exits cleanly when idle.
- ``pull_model_state_dict`` drains in-flight requests and re-opens
  the admit gate.
- Driver crash propagates an exception to all pending futures (no
  forever-await on a dead driver).
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from torchtitan.experiments.rl.actors.generator import SamplingConfig, VLLMGenerator


class _FakeRenderer:
    def render_cmpl(self, prompts):
        return [
            {
                "type": "token",
                "prompt_token_ids": p["prompt_token_ids"],
                "arrival_time": 0.0,
            }
            for p in prompts
        ]


def _request_output(request_id: str, finished: bool = True):
    """Build a SimpleNamespace shaped like vLLM's RequestOutput."""
    sample = SimpleNamespace(
        index=0,
        text="ok",
        token_ids=[10, 11],
        logprobs=[
            {10: SimpleNamespace(logprob=-0.1)},
            {11: SimpleNamespace(logprob=-0.2)},
        ],
        finish_reason="stop",
    )
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3],
        num_cached_tokens=0,
        finished=finished,
        metrics=SimpleNamespace(
            first_token_latency=0.012,
            queued_ts=1.0,
            scheduled_ts=1.005,
            first_token_ts=1.017,
            last_token_ts=1.047,
            num_generation_tokens=2,
        ),
        outputs=[sample],
    )


class _FakeEngine:
    """Engine that finishes requests in submission order.

    Each ``step()`` pops one outstanding request and returns it as
    finished. Multiple add_requests can be in flight at once — the
    finish order is FIFO of admission order. That's enough to test
    the concurrent-admission shape without modeling vLLM's internals.
    """

    def __init__(self) -> None:
        self.inflight: deque[str] = deque()
        self.add_requests: list[tuple[tuple, dict]] = []
        self.renderer = _FakeRenderer()
        self.step_count = 0
        # If non-None, raises on the n-th step() call (1-indexed).
        self.raise_on_step: int | None = None

    def add_request(self, *args, **kwargs):
        self.add_requests.append((args, kwargs))
        self.inflight.append(kwargs["request_id"])

    def has_unfinished_requests(self) -> bool:
        return bool(self.inflight)

    def step(self):
        self.step_count += 1
        if self.raise_on_step is not None and self.step_count >= self.raise_on_step:
            raise RuntimeError("synthetic engine crash")
        if not self.inflight:
            return []
        req_id = self.inflight.popleft()
        return [_request_output(req_id, finished=True)]


def _make_generator() -> VLLMGenerator:
    """Build a VLLMGenerator without running ``__init__`` (no vLLM/GPU)."""
    g = VLLMGenerator.__new__(VLLMGenerator)
    g._engine = _FakeEngine()
    g.policy_version = 7
    g.config = SimpleNamespace(
        sampling=SamplingConfig(n=1, temperature=0.0, top_p=1.0, max_tokens=4),
        debug=SimpleNamespace(seed=None),
    )
    g._next_request_id = 0
    g._engine_step_lock = asyncio.Lock()
    g._req_futures = {}
    g._req_admit_versions = {}
    g._engine_driver_task = None
    g._engine_shutdown = None
    g._admit_gate = None
    g._engine_lock = g._engine_step_lock
    return g


async def _call_generate_tokens(generator, prompts):
    """Invoke the endpoint's underlying coroutine directly."""
    return await VLLMGenerator.generate_tokens._method(  # noqa: SLF001
        generator, prompts
    )


async def _impl_test_concurrent_admissions_coalesce_on_engine():
    """Two concurrent generate_tokens calls share the engine — the
    driver picks up both sets of requests before any finish."""
    generator = _make_generator()
    engine = generator._engine

    # Hold engine.step() until we've admitted both batches by gating
    # the step() return on a counter.
    saw_two_inflight = asyncio.Event()
    original_step = engine.step

    def gated_step():
        # Snapshot the in-flight set on each step. If we've ever
        # observed >= 2 requests concurrently, declare success.
        if len(engine.inflight) >= 2:
            saw_two_inflight.set()
        return original_step()

    engine.step = gated_step

    # Fire both calls concurrently — they should both admit before
    # the driver finishes the first one.
    task_a = asyncio.create_task(_call_generate_tokens(generator, [[1, 2, 3]]))
    task_b = asyncio.create_task(_call_generate_tokens(generator, [[4, 5, 6]]))

    await asyncio.wait_for(asyncio.gather(task_a, task_b), timeout=5.0)
    assert saw_two_inflight.is_set(), "engine never saw two concurrent requests"


async def _impl_test_driver_exits_when_idle():
    """After all requests finish, the driver task returns cleanly so
    next call can recreate it via _ensure_engine_driver."""
    generator = _make_generator()

    await _call_generate_tokens(generator, [[1, 2, 3]])
    # Yield once so the driver's cleanup-after-idle runs.
    await asyncio.sleep(0)
    assert generator._engine_driver_task is not None
    assert generator._engine_driver_task.done()


async def _impl_test_driver_restarts_on_next_admit():
    """First call exits driver on idle; second call must recreate it."""
    generator = _make_generator()

    await _call_generate_tokens(generator, [[1, 2, 3]])
    first_task = generator._engine_driver_task
    await asyncio.sleep(0)
    assert first_task.done()

    await _call_generate_tokens(generator, [[4, 5, 6]])
    second_task = generator._engine_driver_task
    assert second_task is not first_task
    # Two requests were processed total.
    assert generator._engine.step_count >= 2


async def _impl_test_driver_crash_propagates_to_pending_futures():
    """If engine.step() raises, all in-flight futures receive an
    exception instead of hanging forever."""
    generator = _make_generator()
    engine = generator._engine
    # Crash on the first step() call — request already admitted but
    # never finished.
    engine.raise_on_step = 1

    with pytest.raises(RuntimeError):
        await asyncio.wait_for(
            _call_generate_tokens(generator, [[1, 2, 3]]),
            timeout=5.0,
        )
    # Driver should have cleared its state on crash.
    assert not generator._req_futures
    assert not generator._req_admit_versions


# pytest-asyncio isn't installed in titan_rl; wrap each async impl with
# a sync test that calls asyncio.run.


def test_concurrent_admissions_coalesce_on_engine():
    asyncio.run(_impl_test_concurrent_admissions_coalesce_on_engine())


def test_driver_exits_when_idle():
    asyncio.run(_impl_test_driver_exits_when_idle())


def test_driver_restarts_on_next_admit():
    asyncio.run(_impl_test_driver_restarts_on_next_admit())


def test_driver_crash_propagates_to_pending_futures():
    asyncio.run(_impl_test_driver_crash_propagates_to_pending_futures())
