# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the generator's `EngineLoop`: its decision logic (`_decide_next_action`)
and the thread it runs on.

Built on a fake engine, so the loop's admit/load/shutdown branching is tested without
vLLM or a GPU.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import gc
import threading
import weakref
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

import torchtitan.rl.generator.engine_loop as engine_loop_module
import torchtitan.rl.generator.generator as generator_module
from torchtitan.rl.generator.dispatcher import (
    _FinishedRequests,
    _RejectedRequests,
    RequestDispatcher,
)
from torchtitan.rl.generator.engine_loop import (
    CloseMessage,
    EngineLoop,
    EngineLoopMessage,
    EngineLoopQueue,
    EngineLoopThread,
    GenerationMessage,
    LoopAction,
    LoopDecision,
    ModelStateDictLoadMessage,
    SessionReleaseMessage,
)
from torchtitan.rl.generator.generator import Generator
from torchtitan.rl.generator.types import (
    EngineRequest,
    FinishedRequest,
    RequestMetricsInputs,
    SamplingConfig,
)
from torchtitan.rl.routing.intra_generator import IntraGeneratorRouter
from torchtitan.rl.routing.strategies import (
    LeastLoadedRoutingStrategy,
    RoutingStrategy,
    StickySessionRoutingStrategy,
)

_TIMEOUT_S = 5


@pytest.fixture
def runner():
    """One event loop for the whole test, so the queue stays bound to it."""
    with asyncio.Runner() as runner:
        yield runner


def _dispatcher(
    *,
    rank: int = 0,
    tp_rank: int = 0,
    dp_size: int = 1,
    dp_routing_strategy: RoutingStrategy.Config | None = None,
    open_result_channel: Callable[[], tuple[Any, Any]] | None = None,
) -> RequestDispatcher:
    return RequestDispatcher(
        rank=rank,
        dp_rank=0,
        tp_rank=tp_rank,
        dp_degree=dp_size,
        broadcast_group=None,
        open_result_channel=open_result_channel,
        intra_generator_router=IntraGeneratorRouter.Config(
            strategy=dp_routing_strategy or LeastLoadedRoutingStrategy.Config()
        ),
    )


def _bare_loop(
    *,
    event_loop: asyncio.AbstractEventLoop,
    engine: Any = None,
    rank: int = 0,
    request_dispatcher: RequestDispatcher | None = None,
    inflight: bool = False,
    dp_size: int = 1,
    dp_routing_strategy: RoutingStrategy.Config | None = None,
    reset_kv_cache_on_model_state_dict_load: bool = False,
) -> EngineLoop:
    # _decide_next_action delegates the in-flight check and routing to the
    # dispatcher, so wire up a bare one (no engine / GPU needed here).
    engine_loop = EngineLoop(
        engine if engine is not None else _FakeEngine(),
        event_loop=event_loop,
        request_dispatcher=request_dispatcher
        or _dispatcher(dp_size=dp_size, dp_routing_strategy=dp_routing_strategy),
        rank=rank,
        broadcast_group=None,
        max_engine_steps_between_decisions=16,
        reset_kv_cache_on_model_state_dict_load=reset_kv_cache_on_model_state_dict_load,
    )
    # An outstanding generation models in-flight work (possibly in a peer DP rank).
    if inflight:
        engine_loop._request_dispatcher._rank0_outstanding_generations = {
            "inflight": object()
        }
    return engine_loop


def _request(
    request_id: str = "r0",
    *,
    group_id: int = 0,
    routing_session_id: str | None = None,
    prompt_token_ids: list[int] | None = None,
) -> EngineRequest:
    return EngineRequest(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids or [1, 2],
        sampling=SamplingConfig(stop_token_ids=[]),
        group_id=group_id,
        routing_session_id=routing_session_id or request_id,
        arrival_time=0.0,
    )


def _message(engine_request: EngineRequest) -> GenerationMessage:
    return GenerationMessage(
        engine_request=engine_request,
        metrics_prefix="generator",
        reply=concurrent.futures.Future(),
    )


def _load(version: int, *, drain: bool = False) -> ModelStateDictLoadMessage:
    return ModelStateDictLoadMessage(
        version=version, reply=concurrent.futures.Future(), drain=drain
    )


async def _put(
    engine_loop: EngineLoop, *messages: EngineLoopMessage | EngineRequest
) -> None:
    """Puts `messages` on the queue (a `CloseMessage` closes it), wrapping each bare request in a
    `GenerationMessage` as `generate` does."""
    for message in messages:
        if isinstance(message, CloseMessage):
            engine_loop._queue.close(message, reason="generator is closed")
            continue
        if isinstance(message, EngineRequest):
            message = _message(message)
        engine_loop._queue.put(message)
    await asyncio.sleep(0)  # `put` and `close` land on the next loop iteration


def _decide(
    runner: asyncio.Runner,
    engine_loop: EngineLoop,
    pending_engine_requests: list[EngineRequest],
    *messages: EngineLoopMessage | EngineRequest,
    pending_load_messages: list[ModelStateDictLoadMessage] | None = None,
) -> LoopDecision:
    """Puts `messages` on the queue, then runs one decision."""

    async def run() -> LoopDecision:
        await _put(engine_loop, *messages)
        return await asyncio.wait_for(
            engine_loop._decide_next_action(
                pending_engine_requests,
                [] if pending_load_messages is None else pending_load_messages,
            ),
            _TIMEOUT_S,
        )

    return runner.run(run())


def _admit(
    runner: asyncio.Runner, engine_loop: EngineLoop, request: EngineRequest
) -> int:
    """Puts `request` on the queue, runs one STEP decision, and returns its min policy version."""
    decision = _decide(runner, engine_loop, [], request)
    assert decision.action is LoopAction.STEP
    return request.min_policy_version


def test_queue_put_from_another_thread_wakes_an_idle_get() -> None:
    event_loop = asyncio.new_event_loop()
    # Debug mode makes a non-thread-safe call into the loop raise rather than risk a lost wakeup.
    event_loop.set_debug(True)
    thread = threading.Thread(target=event_loop.run_forever, daemon=True)
    thread.start()
    queue = EngineLoopQueue(event_loop)
    message = _message(_request())

    async def park_get() -> concurrent.futures.Future[EngineLoopMessage]:
        got: concurrent.futures.Future[EngineLoopMessage] = concurrent.futures.Future()
        getting = asyncio.create_task(queue.get())
        getting.add_done_callback(lambda task: got.set_result(task.result()))
        await asyncio.sleep(0)  # park `getting` in `get()`
        return got

    try:
        got = asyncio.run_coroutine_threadsafe(park_get(), event_loop).result(
            _TIMEOUT_S
        )
        queue.put(message)
        assert got.result(_TIMEOUT_S) is message
    finally:
        event_loop.call_soon_threadsafe(event_loop.stop)
        thread.join(_TIMEOUT_S)
        event_loop.close()


def test_queue_close_lands_behind_earlier_puts_and_rejects_later_ones(runner) -> None:
    queue = EngineLoopQueue(runner.get_loop())
    message, close_message = _message(_request()), CloseMessage()
    queue.put(message)
    assert not queue.closed
    queue.close(close_message, reason="closed for the test")
    assert queue.closed
    queue.close(CloseMessage(), reason="closed again")  # a no-op
    with pytest.raises(RuntimeError, match="closed for the test"):
        queue.put(_message(_request("r1")))

    async def take_two() -> list[EngineLoopMessage]:
        return [await queue.get(), await queue.get()]

    assert runner.run(take_two()) == [message, close_message]
    assert queue.empty()


def test_queue_close_lands_behind_a_put_from_another_thread_in_progress(runner) -> None:
    put_paused, resume_put = threading.Event(), threading.Event()

    def call_soon_threadsafe(*args):
        # Pause only the put's hand-off, after it has found the queue open.
        if not put_paused.is_set():
            put_paused.set()
            assert resume_put.wait(_TIMEOUT_S)
        return runner.get_loop().call_soon_threadsafe(*args)

    queue = EngineLoopQueue(SimpleNamespace(call_soon_threadsafe=call_soon_threadsafe))
    message, close_message = _message(_request()), CloseMessage()
    putting = threading.Thread(target=queue.put, args=(message,))
    putting.start()
    assert put_paused.wait(_TIMEOUT_S)
    closing = threading.Thread(
        target=queue.close, args=(close_message, "closed for the test")
    )
    closing.start()
    # Time for `close` to overtake the paused put, were it not ordered behind it.
    closing.join(0.1)
    resume_put.set()
    for thread in (putting, closing):
        thread.join(_TIMEOUT_S)
        assert not thread.is_alive()

    async def take_two() -> list[EngineLoopMessage]:
        return [await queue.get(), await queue.get()]

    assert runner.run(take_two()) == [message, close_message]


def test_close_takes_precedence_over_everything(runner) -> None:
    engine_loop = _bare_loop(event_loop=runner.get_loop())
    load = _load(5)
    pending_load_messages: list[ModelStateDictLoadMessage] = []
    decision = _decide(
        runner,
        engine_loop,
        [],
        _request(),
        load,
        CloseMessage(),
        pending_load_messages=pending_load_messages,
    )
    assert decision.action is LoopAction.CLOSE
    assert pending_load_messages == [load]  # for the engine loop to fail


@pytest.mark.parametrize("load", [False, True])
def test_session_releases_ride_the_load_or_step_decision(runner, load) -> None:
    engine_loop = _bare_loop(event_loop=runner.get_loop())
    messages = [
        SessionReleaseMessage(session_ids=["s0"]),
        _request(),
        SessionReleaseMessage(group_ids=[3]),
    ]
    decision = _decide(
        runner, engine_loop, [], *messages, *([_load(5)] if load else [])
    )
    assert decision.action is (
        LoopAction.LOAD_MODEL_STATE_DICT if load else LoopAction.STEP
    )
    assert (decision.released_session_ids, decision.released_group_ids) == (["s0"], [3])


def test_session_releases_are_queued_only_when_the_engine_holds_session_kv(
    runner,
) -> None:
    for holds_session_kv in (False, True):
        engine = _FakeEngine()
        engine.holds_session_kv = holds_session_kv
        engine_loop = _bare_loop(event_loop=runner.get_loop(), engine=engine)
        engine_loop.release_sessions(["s0"])
        engine_loop.release_groups([3])
        runner.run(asyncio.sleep(0))  # `put` lands on the next loop iteration
        assert engine_loop._queue.empty() is not holds_session_kv


def test_load_takes_precedence_over_queued_requests(runner) -> None:
    engine_loop = _bare_loop(event_loop=runner.get_loop())
    request = _request()
    pending_engine_requests: list[EngineRequest] = []
    load = _load(5)
    pending_load_messages: list[ModelStateDictLoadMessage] = []
    decision = _decide(
        runner,
        engine_loop,
        pending_engine_requests,
        request,
        load,
        pending_load_messages=pending_load_messages,
    )
    assert (
        decision.action is LoopAction.LOAD_MODEL_STATE_DICT
        and decision.load_version == 5
    )
    assert pending_load_messages == [load]
    assert pending_engine_requests == [request]  # NOT admitted -- load runs first

    # The carried-over request is admitted at the next decision without a new message.
    decision = _decide(runner, engine_loop, pending_engine_requests)
    assert decision.action is LoopAction.STEP
    assert decision.requests_per_dp_rank == [[request]]
    assert pending_engine_requests == []


def test_loads_taken_off_the_queue_together_coalesce_at_the_highest_version(
    runner,
) -> None:
    engine_loop = _bare_loop(event_loop=runner.get_loop())
    # Out of order, so the highest version wins rather than the last.
    queued = [_load(5), _load(4)]
    pending_load_messages: list[ModelStateDictLoadMessage] = []
    decision = _decide(
        runner, engine_loop, [], *queued, pending_load_messages=pending_load_messages
    )
    assert (
        decision.action is LoopAction.LOAD_MODEL_STATE_DICT
        and decision.load_version == 5
    )
    assert pending_load_messages == queued


def test_draining_load_waits_for_admitted_work_and_holds_new_requests(runner) -> None:
    # One request is admitted and still running when a draining load arrives with a new request.
    engine_loop = _bare_loop(event_loop=runner.get_loop(), inflight=True)
    request, load = _request("r1"), _load(5, drain=True)
    pending_engine_requests: list[EngineRequest] = []
    pending_load_messages: list[ModelStateDictLoadMessage] = []
    decision = _decide(
        runner,
        engine_loop,
        pending_engine_requests,
        load,
        request,
        pending_load_messages=pending_load_messages,
    )
    assert decision.action is LoopAction.STEP and decision.requests_per_dp_rank == [[]]
    assert pending_engine_requests == [request] and pending_load_messages == [load]

    # The admitted request finishes: the load lands with no new message.
    del engine_loop._request_dispatcher._rank0_outstanding_generations["inflight"]
    decision = _decide(
        runner,
        engine_loop,
        pending_engine_requests,
        pending_load_messages=pending_load_messages,
    )
    assert (
        decision.action is LoopAction.LOAD_MODEL_STATE_DICT
        and decision.load_version == 5
    )

    # As `_run` does after applying it; the held request is then admitted with no new message.
    pending_load_messages.clear()
    decision = _decide(runner, engine_loop, pending_engine_requests)
    assert decision.action is LoopAction.STEP
    assert decision.requests_per_dp_rank == [[request]]


def test_draining_load_at_an_idle_engine_applies_at_once(runner) -> None:
    engine_loop = _bare_loop(event_loop=runner.get_loop())
    load = _load(5, drain=True)
    decision = _decide(runner, engine_loop, [], load)
    assert (
        decision.action is LoopAction.LOAD_MODEL_STATE_DICT
        and decision.load_version == 5
    )


def test_session_releases_during_a_drain_ride_the_step_decision(runner) -> None:
    engine_loop = _bare_loop(event_loop=runner.get_loop(), inflight=True)
    decision = _decide(
        runner,
        engine_loop,
        [],
        SessionReleaseMessage(session_ids=["s0"], group_ids=[3]),
        pending_load_messages=[_load(5, drain=True)],
    )
    assert decision.action is LoopAction.STEP and decision.requests_per_dp_rank == [[]]
    assert (decision.released_session_ids, decision.released_group_ids) == (["s0"], [3])


def test_calls_cancelled_on_the_queue_are_skipped(runner) -> None:
    engine_loop = _bare_loop(event_loop=runner.get_loop())
    message, load = _message(_request()), _load(5)
    for call in (message, load):
        call.reply.cancel()
    pending_engine_requests: list[EngineRequest] = []
    pending_load_messages: list[ModelStateDictLoadMessage] = []
    decision = _decide(
        runner,
        engine_loop,
        pending_engine_requests,
        message,
        load,
        pending_load_messages=pending_load_messages,
    )
    assert decision.action is LoopAction.STEP and decision.requests_per_dp_rank == [[]]
    assert pending_engine_requests == [] and pending_load_messages == []
    assert engine_loop._request_dispatcher._rank0_outstanding_generations == {}


def test_duplicate_request_id_is_fatal(runner) -> None:
    engine_loop = _bare_loop(event_loop=runner.get_loop())
    with pytest.raises(AssertionError, match="already in flight"):
        _decide(runner, engine_loop, [], _request("r0"), _request("r0"))


def test_step_drains_the_queue(runner) -> None:
    engine_loop = _bare_loop(event_loop=runner.get_loop())
    request = _request()
    pending_engine_requests: list[EngineRequest] = []
    decision = _decide(runner, engine_loop, pending_engine_requests, request)
    # DP=1: a single DP rank holds the whole batch.
    assert decision.action is LoopAction.STEP and decision.requests_per_dp_rank == [
        [request]
    ]
    assert engine_loop._queue.empty() and pending_engine_requests == []
    assert engine_loop._request_dispatcher._rank0_dp_router is None


def test_step_with_empty_queue_when_only_in_flight_work_remains(runner) -> None:
    # No message, but an outstanding generation means a request is still in flight
    # (possibly in a peer DP rank), so rank 0 must keep issuing STEP.
    engine_loop = _bare_loop(event_loop=runner.get_loop(), inflight=True)
    decision = _decide(runner, engine_loop, [])
    assert decision.action is LoopAction.STEP and decision.requests_per_dp_rank == [[]]


def test_idle_decision_waits_for_the_next_message(runner) -> None:
    engine_loop = _bare_loop(event_loop=runner.get_loop())
    request = _request()

    async def run() -> LoopDecision:
        decision = asyncio.create_task(engine_loop._decide_next_action([], []))
        await asyncio.sleep(0.1)
        assert not decision.done()
        await _put(engine_loop, request)
        return await asyncio.wait_for(decision, _TIMEOUT_S)

    decision = runner.run(run())
    assert decision.action is LoopAction.STEP
    assert decision.requests_per_dp_rank == [[request]]


def test_step_routes_requests_across_dp_ranks(runner) -> None:
    # Least-loaded over 3 idle DP ranks: r0 -> rank 0, r1 -> rank 1 (rank 0 now loaded).
    engine_loop = _bare_loop(event_loop=runner.get_loop(), dp_size=3)
    requests = [_request("r0"), _request("r1")]
    decision = _decide(runner, engine_loop, [], *requests)
    assert decision.action is LoopAction.STEP
    assert decision.requests_per_dp_rank == [[requests[0]], [requests[1]], []]
    # Each request reserves one load unit on its chosen DP rank.
    dp_router = engine_loop._request_dispatcher._rank0_dp_router
    assert dp_router._reservations == {"r0": 0, "r1": 1}
    assert [h.reserved_load for h in dp_router._handles] == [1, 1, 0]


def test_step_sticky_session_reuses_dp_rank(runner) -> None:
    engine_loop = _bare_loop(
        event_loop=runner.get_loop(),
        dp_size=3,
        dp_routing_strategy=StickySessionRoutingStrategy.Config(repin_load_ratio=None),
    )
    pending_engine_requests: list[EngineRequest] = []
    first = _request("r0", routing_session_id="s0")

    first_decision = _decide(runner, engine_loop, pending_engine_requests, first)
    assert first_decision.action is LoopAction.STEP
    assert first_decision.requests_per_dp_rank == [[first], [], []]

    same_session = _request("r1", routing_session_id="s0")
    new_session = _request("r2", routing_session_id="s1")
    second_decision = _decide(
        runner, engine_loop, pending_engine_requests, same_session, new_session
    )
    assert second_decision.action is LoopAction.STEP
    assert second_decision.requests_per_dp_rank == [
        [same_session],
        [new_session],
        [],
    ]
    # r0 and r1 share session s0 -> same DP rank; r2's new session falls back.
    assert engine_loop._request_dispatcher._rank0_dp_router._reservations == {
        "r0": 0,
        "r1": 0,
        "r2": 1,
    }


def test_step_pins_min_policy_version_per_group(runner) -> None:
    engine_loop = _bare_loop(event_loop=runner.get_loop())
    engine_loop.policy_version = 3
    assert (
        _admit(runner, engine_loop, _request("t0", group_id=1, routing_session_id="s0"))
        == 3
    )

    engine_loop.policy_version = 4
    # Later turns and new rollouts of the group keep the salt its first admission
    # pinned, so they can reuse the group's KV; a new group pins the current version.
    assert (
        _admit(runner, engine_loop, _request("t1", group_id=1, routing_session_id="s0"))
        == 3
    )
    assert (
        _admit(runner, engine_loop, _request("t2", group_id=1, routing_session_id="s1"))
        == 3
    )
    assert (
        _admit(runner, engine_loop, _request("t3", group_id=2, routing_session_id="s2"))
        == 4
    )


def test_step_with_kv_reset_uses_current_version_without_pins(runner) -> None:
    engine_loop = _bare_loop(
        event_loop=runner.get_loop(), reset_kv_cache_on_model_state_dict_load=True
    )
    engine_loop.policy_version = 3
    assert (
        _admit(runner, engine_loop, _request("t0", group_id=1, routing_session_id="s0"))
        == 3
    )

    engine_loop.policy_version = 4
    assert (
        _admit(runner, engine_loop, _request("t1", group_id=1, routing_session_id="s0"))
        == 4
    )
    assert engine_loop._group_min_policy_versions == {}


def test_release_groups_drops_pins(runner) -> None:
    engine_loop = _bare_loop(event_loop=runner.get_loop())
    engine_loop.policy_version = 3
    _admit(runner, engine_loop, _request("t0", group_id=1, routing_session_id="s0"))
    _admit(runner, engine_loop, _request("t1", group_id=2, routing_session_id="s1"))

    # Releasing a group this generator never served is a no-op.
    engine_loop.release_groups([1, 9])

    assert engine_loop._group_min_policy_versions == {2: 3}
    engine_loop.policy_version = 4
    assert (
        _admit(runner, engine_loop, _request("t2", group_id=1, routing_session_id="s0"))
        == 4
    )


# --- engine loop thread ---


def _finished(request_id: str) -> FinishedRequest:
    """A request that finished with one token."""
    return FinishedRequest(
        request_id=request_id,
        token_ids=[7],
        token_logprobs=[-0.5],
        finish_reason="stop",
        metrics_inputs=RequestMetricsInputs(
            num_cached_tokens=0,
            has_stats=True,
            first_token_latency=0.01,
            queued_ts=1.0,
            scheduled_ts=1.0,
            first_token_ts=1.01,
            last_token_ts=1.02,
            num_generation_tokens=1,
            num_preemptions=0,
        ),
    )


class _FakeEngine:
    """Every step finishes all admitted requests. `step_hook` runs inside each step;
    `threads` records which threads admitted and stepped, and `shutdowns` which shut it down."""

    holds_session_kv = False

    def __init__(self, step_hook: Callable[[], None] = lambda: None):
        self.step_hook = step_hook
        self.running: list[str] = []
        self.threads: set[threading.Thread] = set()
        self.shutdowns: list[threading.Thread] = []
        self.released_sessions: list[tuple[list[str], list[int]]] = []

    def release_sessions(self, session_ids: list[str], group_ids: list[int]) -> None:
        self.released_sessions.append((session_ids, group_ids))

    def add_requests(
        self, requests: list[EngineRequest]
    ) -> list[tuple[str, Exception]]:
        rejected_requests = []
        for request in requests:
            try:
                self.admit(request)
            except ValueError as exc:
                rejected_requests.append((request.request_id, exc))
        return rejected_requests

    def admit(self, request: EngineRequest) -> None:
        self.threads.add(threading.current_thread())
        self.running.append(request.request_id)

    def has_unfinished_requests(self) -> bool:
        return bool(self.running)

    def step(self) -> list[FinishedRequest]:
        self.threads.add(threading.current_thread())
        self.step_hook()
        finished, self.running = self.running, []
        return [_finished(request_id) for request_id in finished]

    def load_prefetched_model_state_dict(self) -> None:
        pass

    def shutdown(self) -> None:
        self.shutdowns.append(threading.current_thread())


class _RejectingEngine(_FakeEngine):
    """Rejects prompts longer than `max_prompt_len` at admission, as vLLM does."""

    max_prompt_len = 2

    def admit(self, request: EngineRequest) -> None:
        if len(request.prompt_token_ids) > self.max_prompt_len:
            raise ValueError(f"prompt of {request.request_id} is too long")
        super().admit(request)


class _StuckEngine(_FakeEngine):
    """Never finishes a request."""

    def step(self) -> list[FinishedRequest]:
        self.step_hook()
        return []


class _HeldEngine(_FakeEngine):
    """Keeps stepping without finishing a request until `finish` is set."""

    def __init__(self, step_hook: Callable[[], None] = lambda: None):
        super().__init__(step_hook)
        self.finish = threading.Event()

    def step(self) -> list[FinishedRequest]:
        if self.finish.is_set():
            return super().step()
        self.step_hook()
        return []


class _StepGate:
    """A `step_hook` that holds every step until `release()`; `entered` is set once one starts."""

    def __init__(self):
        self.entered, self._released = threading.Event(), threading.Event()

    def __call__(self):
        self.entered.set()
        assert self._released.wait(timeout=_TIMEOUT_S)

    def release(self):
        self._released.set()


def _stop_engine_thread(engine_thread: EngineLoopThread) -> None:
    # Production never stops the engine loop thread's event loop; tests do, to join the thread.
    event_loop = engine_thread.event_loop
    event_loop.call_soon_threadsafe(event_loop.stop)
    engine_thread.thread.join(timeout=_TIMEOUT_S)
    assert not engine_thread.thread.is_alive()
    event_loop.close()


@pytest.fixture
def engine_thread(monkeypatch):
    """Returns a function that builds an engine loop around `engine`, with its engine loop thread
    running; stops those threads afterwards."""
    # Rank 0 is the only rank, so the decision broadcast is a no-op.
    monkeypatch.setattr(
        engine_loop_module.dist, "broadcast_object_list", lambda *a, **k: None
    )
    engine_threads: list[EngineLoopThread] = []

    def start(engine: _FakeEngine, **kwargs: Any) -> EngineLoop:
        thread = EngineLoopThread(name="engine-loop")
        engine_threads.append(thread)
        engine_loop = _bare_loop(event_loop=thread.event_loop, engine=engine, **kwargs)
        engine_loop.engine_thread = thread  # for the tests to check
        return engine_loop

    yield start
    for thread in engine_threads:
        _stop_engine_thread(thread)


def _loading_engine(monkeypatch, load_pre_hook=None) -> _FakeEngine:
    """A fake engine whose model state dict load first awaits `load_pre_hook`."""
    original_load = EngineLoop._load_model_state_dict

    async def new_load(self, version):
        if load_pre_hook is not None:
            await load_pre_hook()
        await original_load(self, version)

    monkeypatch.setattr(EngineLoop, "_load_model_state_dict", new_load)
    return _FakeEngine()


def _generate(engine_loop: EngineLoop, request_id: str, **kwargs: Any) -> asyncio.Task:
    return asyncio.create_task(
        engine_loop.generate(_request(request_id, **kwargs), metrics_prefix="generator")
    )


async def _on_engine_loop(engine_loop: EngineLoop, fn: Callable[[], Any]) -> Any:
    """Run `fn` on the engine loop thread's event loop and return its result."""

    async def call() -> Any:
        return fn()

    return await asyncio.wrap_future(
        asyncio.run_coroutine_threadsafe(call(), engine_loop._event_loop)
    )


def test_call_on_engine_thread_runs_on_it_in_the_callers_context() -> None:
    caller_context = contextvars.ContextVar("caller_context", default=None)
    caller_context.set("caller")
    engine_thread = EngineLoopThread(name="engine-loop")
    try:
        result = engine_thread.call(
            lambda: (threading.current_thread(), caller_context.get())
        )
        assert engine_thread.thread is not threading.current_thread()
        assert result == (engine_thread.thread, "caller")
    finally:
        _stop_engine_thread(engine_thread)


def test_call_on_engine_thread_raises_any_failure_to_the_caller() -> None:
    # Not an `Exception`: the caller must not wait forever on any failure, e.g. of the engine build.
    def fail():
        raise SystemExit("build failed")

    engine_thread = EngineLoopThread(name="engine-loop")
    try:
        with pytest.raises(SystemExit, match="build failed"):
            engine_thread.call(fail)
    finally:
        _stop_engine_thread(engine_thread)


def test_close_releases_the_engine() -> None:
    built: list[weakref.ref[_FakeEngine]] = []
    shutdowns: list[threading.Thread] = []

    def build_engine() -> _FakeEngine:
        engine = _FakeEngine()
        engine.shutdown = lambda: shutdowns.append(threading.current_thread())
        built.append(weakref.ref(engine))
        return engine

    engine_thread = EngineLoopThread(name="engine-loop")
    # Built as `Generator.__init__` builds it, so the engine loop thread must keep no reference to it.
    engine_loop = _bare_loop(
        event_loop=engine_thread.event_loop, engine=engine_thread.call(build_engine)
    )
    try:
        asyncio.run(asyncio.wait_for(engine_loop.close(), _TIMEOUT_S))
        gc.collect()
        assert engine_thread.thread.is_alive()
        assert built[0]() is None
        assert shutdowns == [engine_thread.thread]
    finally:
        _stop_engine_thread(engine_thread)


def test_actor_loop_takes_calls_while_the_engine_steps(engine_thread) -> None:
    gate = _StepGate()
    engine = _FakeEngine(gate)

    async def run() -> None:
        engine_loop = engine_thread(engine)
        engine_loop.start()
        first = _generate(engine_loop, "r0")
        # engine.step() blocks the engine loop thread, not this loop.
        assert await asyncio.to_thread(gate.entered.wait, _TIMEOUT_S)
        second = _generate(engine_loop, "r1")
        gate.release()

        completions = await asyncio.wait_for(asyncio.gather(first, second), _TIMEOUT_S)
        assert [c.request_id for c in completions] == ["r0", "r1"]
        assert engine.threads == {engine_loop.engine_thread.thread}
        assert engine_loop._request_dispatcher._rank0_outstanding_generations == {}

        await asyncio.wait_for(engine_loop.close(), _TIMEOUT_S)
        assert engine_loop._engine is None

    asyncio.run(run())


def test_load_runs_on_the_engine_thread(engine_thread, monkeypatch) -> None:
    # Monarch finds the calling actor through a ContextVar, so the engine loop's work must
    # see the context of the endpoint that started it.
    endpoint_context = contextvars.ContextVar("endpoint_context", default=None)
    reads: list[tuple] = []

    async def load_pre_hook():
        reads.append(
            (
                threading.current_thread(),
                asyncio.get_running_loop(),
                endpoint_context.get(),
            )
        )

    async def run() -> None:
        engine_loop = engine_thread(_loading_engine(monkeypatch, load_pre_hook))
        # Set after the engine loop thread started, so only `start` can carry it there.
        endpoint_context.set("endpoint")
        engine_loop.start()

        await asyncio.wait_for(engine_loop.load_model_state_dict(4), _TIMEOUT_S)
        assert engine_loop.policy_version == 4
        assert reads == [
            (engine_loop.engine_thread.thread, engine_loop._event_loop, "endpoint")
        ]

        completion = await asyncio.wait_for(_generate(engine_loop, "r0"), _TIMEOUT_S)
        assert completion.min_policy_version == completion.max_policy_version == 4

        await asyncio.wait_for(engine_loop.close(), _TIMEOUT_S)

    asyncio.run(run())


def test_generate_endpoint_queues_a_request_with_the_default_sampling(
    engine_thread,
) -> None:
    engine = _FakeEngine()
    admitted: list[EngineRequest] = []
    engine.admit = lambda request: (
        admitted.append(request),
        _FakeEngine.admit(engine, request),
    )
    sampling = SamplingConfig(temperature=0.0, stop_token_ids=[9])

    async def run() -> None:
        # The generator's endpoints around the engine loop, without building vLLM.
        generator = object.__new__(Generator)
        generator.config = SimpleNamespace(sampling=sampling)
        generator._engine_loop = engine_thread(engine)
        await generator.start_engine_loop()
        waiting = asyncio.create_task(generator.wait_engine_loop())
        with pytest.raises(AssertionError, match="stop_token_ids must be set"):
            await generator.generate(
                [1, 2],
                request_id="r0",
                group_id=4,
                routing_session_id="s0",
                sampling_config=SamplingConfig(),
            )

        completion = await asyncio.wait_for(
            generator.generate(
                [1, 2, 3], request_id="r1", group_id=4, routing_session_id="s1"
            ),
            _TIMEOUT_S,
        )
        assert completion.request_id == "r1"
        [request] = admitted
        assert request.prompt_token_ids == [1, 2, 3]
        assert request.sampling is sampling
        assert (request.group_id, request.routing_session_id) == (4, "s1")

        await generator.release_groups([4])
        assert generator._engine_loop._group_min_policy_versions == {}
        await asyncio.wait_for(generator.close(), _TIMEOUT_S)
        await asyncio.wait_for(waiting, _TIMEOUT_S)

    asyncio.run(run())


def test_prefetch_reads_torchstore_on_the_actor_loop_while_the_engine_steps(
    engine_thread, monkeypatch
) -> None:
    reads: list[tuple[threading.Thread, asyncio.AbstractEventLoop]] = []

    async def get_state_dict(*args, **kwargs):
        reads.append((threading.current_thread(), asyncio.get_running_loop()))

    monkeypatch.setattr(generator_module.ts, "get_state_dict", get_state_dict)
    gate = _StepGate()
    engine = _loading_engine(monkeypatch)
    engine.step_hook = gate

    async def run() -> None:
        # The generator's endpoints around the engine loop, without building vLLM.
        generator = object.__new__(Generator)
        generator._engine_loop = engine_thread(engine)
        generator._prefetched_model_state_dict = {}
        await generator.start_engine_loop()
        first = _generate(generator._engine_loop, "r0")
        assert await asyncio.to_thread(gate.entered.wait, _TIMEOUT_S)

        await asyncio.wait_for(generator.prefetch_model_state_dict(), _TIMEOUT_S)
        actor_loop = (threading.current_thread(), asyncio.get_running_loop())
        assert reads == [actor_loop]

        gate.release()
        await asyncio.wait_for(first, _TIMEOUT_S)
        # The load applies the prefetched weights without reading TorchStore again.
        await asyncio.wait_for(generator.load_model_state_dict(4), _TIMEOUT_S)
        assert reads == [actor_loop]
        assert generator.policy_version == 4

        await asyncio.wait_for(generator.close(), _TIMEOUT_S)

    asyncio.run(run())


def test_loads_queued_during_a_load_are_applied_together_after_it(
    engine_thread, monkeypatch
) -> None:
    reading = [threading.Event(), threading.Event()]
    released = [threading.Event(), threading.Event()]
    reads: list[int] = []

    async def load_pre_hook():
        read = len(reads)
        reads.append(read)
        reading[read].set()
        assert await asyncio.to_thread(released[read].wait, _TIMEOUT_S)

    async def run() -> None:
        engine_loop = engine_thread(_loading_engine(monkeypatch, load_pre_hook))
        engine_loop.start()
        first = asyncio.create_task(engine_loop.load_model_state_dict(3))
        assert await asyncio.to_thread(reading[0].wait, _TIMEOUT_S)

        later = [
            asyncio.create_task(engine_loop.load_model_state_dict(version))
            for version in (4, 5)
        ]
        await asyncio.sleep(0)  # both put their call on the queue
        # Queued on the engine loop behind both puts, so both loads are on the queue once it returns.
        await _on_engine_loop(engine_loop, lambda: None)
        released[0].set()
        await asyncio.wait_for(first, _TIMEOUT_S)
        assert engine_loop.policy_version == 3

        # The first load resolves only its own caller; the later two wait for one more read.
        assert await asyncio.to_thread(reading[1].wait, _TIMEOUT_S)
        assert not any(load.done() for load in later)
        released[1].set()
        await asyncio.wait_for(asyncio.gather(*later), _TIMEOUT_S)
        assert engine_loop.policy_version == 5
        assert len(reads) == 2

        await asyncio.wait_for(engine_loop.close(), _TIMEOUT_S)

    asyncio.run(run())


def test_draining_load_finishes_admitted_requests_then_admits_held_ones(
    engine_thread,
) -> None:
    stepping = threading.Event()
    engine = _HeldEngine(stepping.set)

    async def run() -> None:
        engine_loop = engine_thread(engine)
        # (request id, policy version at admission)
        admitted: list[tuple[str, int]] = []
        admit = engine.admit
        engine.admit = lambda request: (
            admitted.append((request.request_id, engine_loop.policy_version)),
            admit(request),
        )
        engine_loop.start()
        running = _generate(engine_loop, "r0")
        assert await asyncio.to_thread(stepping.wait, _TIMEOUT_S)  # r0 is admitted

        load = asyncio.create_task(engine_loop.load_model_state_dict(2, drain=True))
        # A fresh group, so no older salt pin keeps its min policy version below the new one.
        held = _generate(engine_loop, "r1", group_id=9)
        await asyncio.sleep(0)  # both put their call on the queue
        # Queued on the engine loop behind both puts, so the loop has taken them off once it returns.
        await _on_engine_loop(engine_loop, lambda: None)
        await _on_engine_loop(engine_loop, lambda: None)
        assert not load.done() and not held.done()
        assert admitted == [("r0", 0)]  # r1 waits for the load

        # r0 finishes, the load lands, and r1 is admitted on the new weights with no further message.
        engine.finish.set()
        await asyncio.wait_for(asyncio.gather(running, load), _TIMEOUT_S)
        completion = await asyncio.wait_for(held, _TIMEOUT_S)
        assert admitted == [("r0", 0), ("r1", 2)]
        assert (completion.min_policy_version, completion.max_policy_version) == (2, 2)

        await asyncio.wait_for(engine_loop.close(), _TIMEOUT_S)

    asyncio.run(run())


def test_close_during_a_drain_fails_the_load_and_every_request(engine_thread) -> None:
    stepping = threading.Event()
    engine = _HeldEngine(stepping.set)

    async def run() -> None:
        engine_loop = engine_thread(engine)
        engine_loop.start()
        running = _generate(engine_loop, "r0")
        assert await asyncio.to_thread(stepping.wait, _TIMEOUT_S)  # r0 is admitted
        load = asyncio.create_task(engine_loop.load_model_state_dict(2, drain=True))
        held = _generate(engine_loop, "r1")
        await asyncio.sleep(0)  # both put their call on the queue
        await _on_engine_loop(engine_loop, lambda: None)
        await _on_engine_loop(engine_loop, lambda: None)

        await asyncio.wait_for(engine_loop.close(), _TIMEOUT_S)
        # The admitted request, the parked one and the pending load all fail; none hangs.
        for call in (running, held, load):
            with pytest.raises(RuntimeError):
                await asyncio.wait_for(call, _TIMEOUT_S)

    asyncio.run(run())


def test_crash_fails_outstanding_and_queued_calls_and_later_calls(
    engine_thread, monkeypatch
) -> None:
    reading, released = threading.Event(), threading.Event()

    async def load_pre_hook():
        reading.set()
        assert await asyncio.to_thread(released.wait, _TIMEOUT_S)
        raise RuntimeError("TorchStore is down")

    async def run() -> None:
        engine_loop = engine_thread(_loading_engine(monkeypatch, load_pre_hook))
        engine_loop.start()
        failing_load = asyncio.create_task(engine_loop.load_model_state_dict(3))
        assert await asyncio.to_thread(reading.wait, _TIMEOUT_S)

        # A session release (no reply) queued ahead of the calls must not stop them from failing.
        engine_loop._queue.put(SessionReleaseMessage(session_ids=["s"]))
        queued = [
            _generate(engine_loop, "r0"),
            asyncio.create_task(engine_loop.load_model_state_dict(4)),
        ]
        await asyncio.sleep(0)  # both put their call on the queue
        # Queued on the engine loop behind both puts, so both are on the queue once it returns.
        await _on_engine_loop(engine_loop, lambda: None)
        released.set()

        for call in (failing_load, *queued):
            with pytest.raises(RuntimeError, match="TorchStore is down"):
                await asyncio.wait_for(call, _TIMEOUT_S)
        with pytest.raises(RuntimeError, match="generator is closed"):
            await asyncio.wait_for(_generate(engine_loop, "r1"), _TIMEOUT_S)
        # `close` only logs the loop's error, so check that failing the queue didn't replace it.
        exc = engine_loop._future.exception(timeout=_TIMEOUT_S)
        assert isinstance(exc, RuntimeError) and str(exc) == "TorchStore is down"
        assert engine_loop._engine is None  # released by the loop on its way out
        with pytest.raises(RuntimeError, match="TorchStore is down"):
            await asyncio.wait_for(engine_loop.wait(), _TIMEOUT_S)
        await asyncio.wait_for(engine_loop.close(), _TIMEOUT_S)

    asyncio.run(run())


def test_generate_cancelled_on_the_queue_never_reaches_the_engine(
    engine_thread,
) -> None:
    gate = _StepGate()
    engine = _FakeEngine()
    stepped: list[list[str]] = []

    def step_hook() -> None:
        stepped.append(list(engine.running))
        gate()

    engine.step_hook = step_hook

    async def run() -> None:
        engine_loop = engine_thread(engine)
        engine_loop.start()
        first = _generate(engine_loop, "r0")
        assert await asyncio.to_thread(gate.entered.wait, _TIMEOUT_S)
        cancelled = _generate(engine_loop, "r1")
        # r1 is put on the queue, which the held engine loop thread can't drain.
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        gate.release()

        await asyncio.wait_for(first, _TIMEOUT_S)
        await asyncio.wait_for(_generate(engine_loop, "r2"), _TIMEOUT_S)
        assert stepped == [["r0"], ["r2"]]
        assert engine_loop._request_dispatcher._rank0_outstanding_generations == {}

        await asyncio.wait_for(engine_loop.close(), _TIMEOUT_S)

    asyncio.run(run())


def test_generate_cancelled_after_admission_leaves_the_loop_running(
    engine_thread,
) -> None:
    stepping = threading.Event()
    engine = _HeldEngine(stepping.set)

    async def run() -> None:
        engine_loop = engine_thread(engine)
        engine_loop.start()
        cancelled = _generate(engine_loop, "r0")
        assert await asyncio.to_thread(stepping.wait, _TIMEOUT_S)  # r0 is admitted
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        engine.finish.set()

        # r0 still finishes, and its reply is resolved without crashing the loop.
        completion = await asyncio.wait_for(_generate(engine_loop, "r1"), _TIMEOUT_S)
        assert completion.request_id == "r1"
        assert engine_loop._request_dispatcher._rank0_outstanding_generations == {}

        await asyncio.wait_for(engine_loop.close(), _TIMEOUT_S)

    asyncio.run(run())


def test_request_rejected_at_admission_fails_alone(engine_thread) -> None:
    engine = _RejectingEngine()

    async def run() -> None:
        engine_loop = engine_thread(engine)
        engine_loop.start()
        rejected = _generate(engine_loop, "long", prompt_token_ids=[1, 2, 3])
        admitted = _generate(engine_loop, "r0")

        with pytest.raises(ValueError, match="prompt of long is too long"):
            await asyncio.wait_for(rejected, _TIMEOUT_S)
        assert (await asyncio.wait_for(admitted, _TIMEOUT_S)).request_id == "r0"
        # The loop keeps serving.
        assert (
            await asyncio.wait_for(_generate(engine_loop, "r1"), _TIMEOUT_S)
        ).request_id == "r1"
        assert engine_loop._request_dispatcher._rank0_outstanding_generations == {}

        await asyncio.wait_for(engine_loop.close(), _TIMEOUT_S)

    asyncio.run(run())


def test_close_fails_outstanding_requests_and_later_calls(engine_thread) -> None:
    gate = _StepGate()

    async def run() -> None:
        engine_loop = engine_thread(_StuckEngine(gate))
        engine_loop.start()
        engine_loop_future = engine_loop._future
        waiting = asyncio.create_task(engine_loop.wait())
        in_flight = _generate(engine_loop, "r0")
        assert await asyncio.to_thread(gate.entered.wait, _TIMEOUT_S)
        queued = _generate(engine_loop, "r1")
        queued_load = asyncio.create_task(engine_loop.load_model_state_dict(4))
        closing = asyncio.create_task(engine_loop.close())
        # Runs after `close` has closed the queue, while the loop still runs, so the guard rejects it
        # before it reaches the queue.
        during_close = _generate(engine_loop, "r2")
        # Run all four calls while `step` holds the engine loop thread, so the loop takes the load and the
        # `CloseMessage` off the queue together; alone, the load would run on `_StuckEngine`.
        await asyncio.sleep(0)
        # `close` and the rejected call ran on this loop, without waiting for the engine loop thread.
        assert engine_loop._queue.closed and during_close.done()
        gate.release()

        await asyncio.wait_for(closing, _TIMEOUT_S)
        assert engine_loop_future.done()
        assert engine_loop._engine is None
        await asyncio.wait_for(
            waiting, _TIMEOUT_S
        )  # a clean close ends the wait without error
        for request in (in_flight, queued):
            with pytest.raises(
                RuntimeError, match="closed before the request finished"
            ):
                await request
        with pytest.raises(RuntimeError, match="closed before the load was applied"):
            await asyncio.wait_for(queued_load, _TIMEOUT_S)
        with pytest.raises(
            RuntimeError, match="generator is closed; cannot call generate"
        ):
            await during_close

        # Later calls fail instead of hanging, and a second `close` still returns.
        with pytest.raises(RuntimeError, match="generator is closed"):
            await asyncio.wait_for(_generate(engine_loop, "r3"), _TIMEOUT_S)
        with pytest.raises(RuntimeError, match="generator is closed"):
            await asyncio.wait_for(engine_loop.load_model_state_dict(4), _TIMEOUT_S)
        await asyncio.wait_for(engine_loop.close(), _TIMEOUT_S)

    asyncio.run(run())


def test_follower_applies_broadcast_decisions_on_the_engine_thread(
    engine_thread, monkeypatch
) -> None:
    request = _request("r0")
    request.min_policy_version = 0  # rank 0 pins it on admission
    decisions = iter(
        [
            LoopDecision(action=LoopAction.STEP, requests_per_dp_rank=[[request]]),
            LoopDecision(
                action=LoopAction.LOAD_MODEL_STATE_DICT,
                requests_per_dp_rank=[],
                load_version=4,
                released_session_ids=["s0"],
                released_group_ids=[3],
            ),
            LoopDecision(action=LoopAction.CLOSE, requests_per_dp_rank=[]),
        ]
    )

    def broadcast_object_list(container, **kwargs):
        # A follower sends nothing and receives rank 0's decision.
        assert container == [None]
        container[0] = next(decisions)

    monkeypatch.setattr(
        engine_loop_module.dist, "broadcast_object_list", broadcast_object_list
    )
    reads: list[threading.Thread] = []

    async def load_pre_hook():
        reads.append(threading.current_thread())

    async def run() -> None:
        engine = _loading_engine(monkeypatch, load_pre_hook)
        stepped: list[list[str]] = []
        engine.step_hook = lambda: stepped.append(list(engine.running))
        # TP rank 1 of a DP=1, TP=2 generator.
        engine_loop = engine_thread(
            engine, rank=1, request_dispatcher=_dispatcher(rank=1, tp_rank=1)
        )
        engine_loop.start()

        # The broadcast CLOSE ends the loop; a follower's close only waits for that.
        await asyncio.wait_for(engine_loop.close(), _TIMEOUT_S)
        assert stepped == [["r0"]]
        assert engine.threads == {engine_loop.engine_thread.thread}
        assert reads == [engine_loop.engine_thread.thread]
        assert engine_loop.policy_version == 4
        # Every rank releases the sessions rank 0 decided, before acting on the decision.
        assert engine.released_sessions == [([], []), (["s0"], [3])]

    asyncio.run(run())


@pytest.mark.parametrize("peer_rejects", [False, True])
def test_drain_task_resolves_peer_results_on_the_engine_thread(
    engine_thread, monkeypatch, peer_rejects
) -> None:
    peer_results: asyncio.Queue = asyncio.Queue()
    recv_threads: list[threading.Thread] = []

    class ResultReceiver:
        async def recv(self):
            recv_threads.append(threading.current_thread())
            return await peer_results.get()

    async def run() -> None:
        engine = _FakeEngine()
        dispatcher = _dispatcher(
            dp_size=2, open_result_channel=lambda: ("port", ResultReceiver())
        )
        # Load DP rank 0, so the request goes to the peer, DP rank 1.
        dispatcher._rank0_dp_router.reserve("busy", routing_session_id="busy")
        engine_loop = engine_thread(engine, request_dispatcher=dispatcher)
        peer_requests: list[str] = []

        def broadcast_object_list(container, **kwargs):
            # Stand-in for DP rank 1, which meets rank 0 at every decision broadcast: it sends
            # back the finished (or rejected) requests it took at the previous one,
            # then takes its share of this one.
            if peer_requests:
                if peer_rejects:
                    results = _RejectedRequests(
                        requests=[
                            (request_id, ValueError(f"{request_id} rejected"))
                            for request_id in peer_requests
                        ]
                    )
                else:
                    results = _FinishedRequests(
                        requests=[
                            _finished(request_id) for request_id in peer_requests
                        ],
                        # The peer installed a newer version before they finished.
                        policy_version=5,
                    )
                peer_requests.clear()
                engine_loop._event_loop.call_soon_threadsafe(
                    peer_results.put_nowait, results
                )
            decision = container[0]
            if (
                isinstance(decision, LoopDecision)
                and decision.action is LoopAction.STEP
            ):
                peer_requests.extend(
                    r.request_id for r in decision.requests_per_dp_rank[1]
                )

        monkeypatch.setattr(
            engine_loop_module.dist, "broadcast_object_list", broadcast_object_list
        )
        engine_loop.policy_version = 3  # rank 0 stamps it at admission
        engine_loop.start()

        if peer_rejects:
            with pytest.raises(ValueError, match="r0 rejected"):
                await asyncio.wait_for(_generate(engine_loop, "r0"), _TIMEOUT_S)
        else:
            completion = await asyncio.wait_for(
                _generate(engine_loop, "r0"), _TIMEOUT_S
            )
            assert completion.request_id == "r0"
            assert (completion.min_policy_version, completion.max_policy_version) == (
                3,
                5,
            )
        # Either way the peer's result frees its DP rank: only the "busy" reservation is left.
        assert dispatcher._rank0_dp_router._reservations == {"busy": 0}
        assert engine.threads == set()  # rank 0's own DP replica served nothing
        assert set(recv_threads) == {engine_loop.engine_thread.thread}

        await asyncio.wait_for(engine_loop.close(), _TIMEOUT_S)

    asyncio.run(run())
