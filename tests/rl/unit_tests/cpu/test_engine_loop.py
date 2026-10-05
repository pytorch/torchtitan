# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the generator engine loop: its decision logic (`_decide_next_action`)
and the engine thread it runs on.

Built on a bare `VLLMGenerator` (no vLLM engine) + a fake engine, so the loop's
admit/pull/shutdown branching is tested without a GPU.
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

import torchtitan.rl.generator as generator_module
from torchtitan.rl.distributed.routing.intra_generator import IntraGeneratorRouter
from torchtitan.rl.distributed.routing.strategies import (
    LeastLoadedRoutingStrategy,
    RoutingStrategy,
    StickySessionRoutingStrategy,
)

from torchtitan.rl.generator import (
    CloseRequest,
    EngineLoopInbox,
    EngineLoopMessage,
    EngineRequest,
    GenerationRequest,
    LoopAction,
    LoopDecision,
    ModelStateDictPullRequest,
    RequestDispatcher,
    SamplingConfig,
    VLLMGenerator,
)
from vllm.logprobs import FlatLogprobs, Logprob

_TIMEOUT_S = 5


@pytest.fixture
def runner():
    """One event loop for the whole test, so the inbox stays bound to it."""
    with asyncio.Runner() as runner:
        yield runner


def _bare_generator(
    *,
    event_loop: asyncio.AbstractEventLoop | None = None,
    inflight: bool = False,
    dp_size: int = 1,
    dp_routing_strategy: RoutingStrategy.Config | None = None,
    reset_kv_cache_on_weight_sync: bool = False,
) -> VLLMGenerator:
    # Bypass __init__ (which builds the vLLM engine); set only the loop's state.
    # _decide_next_action delegates the in-flight check and routing to the
    # dispatcher, so wire up a bare one (no engine / GPU needed here).
    generator = object.__new__(VLLMGenerator)
    generator.config = SimpleNamespace(
        reset_kv_cache_on_weight_sync=reset_kv_cache_on_weight_sync,
        enable_cpu_weight_prefetch=False,
    )
    generator.policy_version = 0
    generator._group_min_policy_versions = {}
    # Engine-thread tests bind the inbox once the engine thread's event loop exists.
    if event_loop is not None:
        generator._inbox = EngineLoopInbox(event_loop)
    generator._request_dispatcher = RequestDispatcher(
        rank=0,
        dp_rank=0,
        tp_rank=0,
        dp_degree=dp_size,
        broadcast_group=None,
        open_result_channel=None,
        intra_generator_router=IntraGeneratorRouter.Config(
            strategy=dp_routing_strategy or LeastLoadedRoutingStrategy.Config()
        ),
    )
    # A registered-but-unresolved future models in-flight work (possibly in a peer DP rank).
    if inflight:
        generator._request_dispatcher._rank0_generation_futures = {"inflight": object()}
    return generator


def _request(
    request_id: str = "r0",
    *,
    group_id: int = 0,
    routing_session_id: str | None = None,
) -> EngineRequest:
    return EngineRequest(
        request_id=request_id,
        prompt_token_ids=[1, 2],
        sampling=SamplingConfig(),
        group_id=group_id,
        routing_session_id=routing_session_id or request_id,
    )


def _message(engine_request: EngineRequest) -> GenerationRequest:
    return GenerationRequest(
        engine_request=engine_request,
        metrics_prefix="generator",
        reply=concurrent.futures.Future(),
    )


def _pull(version: int) -> ModelStateDictPullRequest:
    return ModelStateDictPullRequest(version=version, reply=concurrent.futures.Future())


async def _put(
    generator: VLLMGenerator, *messages: EngineLoopMessage | EngineRequest
) -> None:
    """Puts `messages` on the inbox (a `CloseRequest` closes it), wrapping each bare request in a
    `GenerationRequest` as `generate` does."""
    for message in messages:
        if isinstance(message, CloseRequest):
            generator._inbox.close(message, reason="generator is closed")
            continue
        if isinstance(message, EngineRequest):
            message = _message(message)
        generator._inbox.put(message)
    await asyncio.sleep(0)  # `put` and `close` land on the next loop iteration


def _decide(
    runner: asyncio.Runner,
    generator: VLLMGenerator,
    pending: list[EngineRequest],
    *messages: EngineLoopMessage | EngineRequest,
    pulls: list[ModelStateDictPullRequest] | None = None,
) -> LoopDecision:
    """Puts `messages` on the inbox, then runs one decision."""

    async def run() -> LoopDecision:
        await _put(generator, *messages)
        return await asyncio.wait_for(
            generator._decide_next_action(pending, [] if pulls is None else pulls),
            _TIMEOUT_S,
        )

    return runner.run(run())


def _admit(
    runner: asyncio.Runner, generator: VLLMGenerator, request: GenerationRequest
) -> int:
    """Puts `request` on the inbox, runs one STEP decision, and returns its min policy version."""
    decision = _decide(runner, generator, [], request)
    assert decision.action is LoopAction.STEP
    return request.min_policy_version


def test_inbox_put_from_another_thread_wakes_an_idle_get() -> None:
    event_loop = asyncio.new_event_loop()
    # Debug mode makes a non-thread-safe call into the loop raise rather than risk a lost wakeup.
    event_loop.set_debug(True)
    thread = threading.Thread(target=event_loop.run_forever, daemon=True)
    thread.start()
    inbox = EngineLoopInbox(event_loop)
    message = _message(_request())

    async def park_get() -> concurrent.futures.Future[EngineLoopMessage]:
        got: concurrent.futures.Future[EngineLoopMessage] = concurrent.futures.Future()
        getting = asyncio.create_task(inbox.get())
        getting.add_done_callback(lambda task: got.set_result(task.result()))
        await asyncio.sleep(0)  # park `getting` in `get()`
        return got

    try:
        got = asyncio.run_coroutine_threadsafe(park_get(), event_loop).result(
            _TIMEOUT_S
        )
        inbox.put(message)
        assert got.result(_TIMEOUT_S) is message
    finally:
        event_loop.call_soon_threadsafe(event_loop.stop)
        thread.join(_TIMEOUT_S)
        event_loop.close()


def test_inbox_close_lands_behind_earlier_puts_and_rejects_later_ones(runner) -> None:
    inbox = EngineLoopInbox(runner.get_loop())
    message, close_request = _message(_request()), CloseRequest()
    inbox.put(message)
    assert not inbox.closed
    inbox.close(close_request, reason="closed for the test")
    assert inbox.closed
    inbox.close(CloseRequest(), reason="closed again")  # a no-op
    with pytest.raises(RuntimeError, match="closed for the test"):
        inbox.put(_message(_request("r1")))

    async def take_two() -> list[EngineLoopMessage]:
        return [await inbox.get(), await inbox.get()]

    assert runner.run(take_two()) == [message, close_request]
    assert inbox.empty()


def test_inbox_close_lands_behind_a_put_from_another_thread_in_progress(runner) -> None:
    put_paused, resume_put = threading.Event(), threading.Event()

    def call_soon_threadsafe(*args):
        # Pause only the put's hand-off, after it has found the inbox open.
        if not put_paused.is_set():
            put_paused.set()
            assert resume_put.wait(_TIMEOUT_S)
        return runner.get_loop().call_soon_threadsafe(*args)

    inbox = EngineLoopInbox(SimpleNamespace(call_soon_threadsafe=call_soon_threadsafe))
    message, close_request = _message(_request()), CloseRequest()
    putting = threading.Thread(target=inbox.put, args=(message,))
    putting.start()
    assert put_paused.wait(_TIMEOUT_S)
    closing = threading.Thread(
        target=inbox.close, args=(close_request, "closed for the test")
    )
    closing.start()
    # Time for `close` to overtake the paused put, were it not ordered behind it.
    closing.join(0.1)
    resume_put.set()
    for thread in (putting, closing):
        thread.join(_TIMEOUT_S)
        assert not thread.is_alive()

    async def take_two() -> list[EngineLoopMessage]:
        return [await inbox.get(), await inbox.get()]

    assert runner.run(take_two()) == [message, close_request]


def test_close_takes_precedence_over_everything(runner) -> None:
    generator = _bare_generator(event_loop=runner.get_loop())
    pull = _pull(5)
    pulls: list[ModelStateDictPullRequest] = []
    decision = _decide(
        runner, generator, [], _request(), pull, CloseRequest(), pulls=pulls
    )
    assert decision.action is LoopAction.CLOSE
    assert pulls == [pull]  # for the engine loop to fail


def test_pull_takes_precedence_over_queued_requests(runner) -> None:
    generator = _bare_generator(event_loop=runner.get_loop())
    request = _request()
    pending: list[EngineRequest] = []
    pull = _pull(5)
    pulls: list[ModelStateDictPullRequest] = []
    decision = _decide(runner, generator, pending, request, pull, pulls=pulls)
    assert (
        decision.action is LoopAction.PULL_MODEL_STATE_DICT
        and decision.pull_version == 5
    )
    assert pulls == [pull]
    assert pending == [request]  # NOT admitted -- pull runs first

    # The carried-over request is admitted at the next decision without a new message.
    decision = _decide(runner, generator, pending)
    assert decision.action is LoopAction.STEP
    assert decision.requests_per_dp_rank == [[request]]
    assert pending == []


def test_pulls_taken_off_the_inbox_together_coalesce_at_the_highest_version(
    runner,
) -> None:
    generator = _bare_generator(event_loop=runner.get_loop())
    # Out of order, so the highest version wins rather than the last.
    queued = [_pull(5), _pull(4)]
    pulls: list[ModelStateDictPullRequest] = []
    decision = _decide(runner, generator, [], *queued, pulls=pulls)
    assert (
        decision.action is LoopAction.PULL_MODEL_STATE_DICT
        and decision.pull_version == 5
    )
    assert pulls == queued


def test_calls_cancelled_on_the_inbox_are_skipped(runner) -> None:
    generator = _bare_generator(event_loop=runner.get_loop())
    message, pull = _message(_request()), _pull(5)
    for call in (message, pull):
        call.reply.cancel()
    pending: list[EngineRequest] = []
    pulls: list[ModelStateDictPullRequest] = []
    decision = _decide(runner, generator, pending, message, pull, pulls=pulls)
    assert decision.action is LoopAction.STEP and decision.requests_per_dp_rank == [[]]
    assert pending == [] and pulls == []
    assert generator._request_dispatcher._rank0_generation_futures == {}


def test_duplicate_request_id_is_fatal(runner) -> None:
    generator = _bare_generator(event_loop=runner.get_loop())
    with pytest.raises(AssertionError, match="already in flight"):
        _decide(runner, generator, [], _request("r0"), _request("r0"))


def test_step_drains_the_inbox(runner) -> None:
    generator = _bare_generator(event_loop=runner.get_loop())
    request = _request()
    pending: list[EngineRequest] = []
    decision = _decide(runner, generator, pending, request)
    # DP=1: a single DP rank holds the whole batch.
    assert decision.action is LoopAction.STEP and decision.requests_per_dp_rank == [
        [request]
    ]
    assert generator._inbox.empty() and pending == []
    assert generator._request_dispatcher._rank0_dp_router is None


def test_step_with_empty_inbox_when_only_in_flight_work_remains(runner) -> None:
    # No message, but a registered future means a request is still in flight
    # (possibly in a peer DP rank), so rank 0 must keep issuing STEP.
    generator = _bare_generator(event_loop=runner.get_loop(), inflight=True)
    decision = _decide(runner, generator, [])
    assert decision.action is LoopAction.STEP and decision.requests_per_dp_rank == [[]]


def test_idle_decision_waits_for_the_next_message(runner) -> None:
    generator = _bare_generator(event_loop=runner.get_loop())
    request = _request()

    async def run() -> LoopDecision:
        decision = asyncio.create_task(generator._decide_next_action([], []))
        await asyncio.sleep(0.1)
        assert not decision.done()
        await _put(generator, request)
        return await asyncio.wait_for(decision, _TIMEOUT_S)

    decision = runner.run(run())
    assert decision.action is LoopAction.STEP
    assert decision.requests_per_dp_rank == [[request]]


def test_step_routes_requests_across_dp_ranks(runner) -> None:
    # Least-loaded over 3 idle DP ranks: r0 -> rank 0, r1 -> rank 1 (rank 0 now loaded).
    generator = _bare_generator(event_loop=runner.get_loop(), dp_size=3)
    requests = [_request("r0"), _request("r1")]
    decision = _decide(runner, generator, [], *requests)
    assert decision.action is LoopAction.STEP
    assert decision.requests_per_dp_rank == [[requests[0]], [requests[1]], []]
    # Each request reserves one load unit on its chosen DP rank.
    dp_router = generator._request_dispatcher._rank0_dp_router
    assert dp_router._reservations == {"r0": 0, "r1": 1}
    assert [h.reserved_load for h in dp_router._handles] == [1, 1, 0]


def test_step_sticky_session_reuses_dp_rank(runner) -> None:
    generator = _bare_generator(
        event_loop=runner.get_loop(),
        dp_size=3,
        dp_routing_strategy=StickySessionRoutingStrategy.Config(),
    )
    pending: list[EngineRequest] = []
    first = _request("r0", routing_session_id="s0")

    first_decision = _decide(runner, generator, pending, first)
    assert first_decision.action is LoopAction.STEP
    assert first_decision.requests_per_dp_rank == [[first], [], []]

    same_session = _request("r1", routing_session_id="s0")
    new_session = _request("r2", routing_session_id="s1")
    second_decision = _decide(runner, generator, pending, same_session, new_session)
    assert second_decision.action is LoopAction.STEP
    assert second_decision.requests_per_dp_rank == [
        [same_session],
        [new_session],
        [],
    ]
    # r0 and r1 share session s0 -> same DP rank; r2's new session falls back.
    assert generator._request_dispatcher._rank0_dp_router._reservations == {
        "r0": 0,
        "r1": 0,
        "r2": 1,
    }


def test_step_pins_min_policy_version_per_group(runner) -> None:
    generator = _bare_generator(event_loop=runner.get_loop())
    generator.policy_version = 3
    assert (
        _admit(runner, generator, _request("t0", group_id=1, routing_session_id="s0"))
        == 3
    )

    generator.policy_version = 4
    # Later turns and new rollouts of the group keep the salt its first admission
    # pinned, so they can reuse the group's KV; a new group pins the current version.
    assert (
        _admit(runner, generator, _request("t1", group_id=1, routing_session_id="s0"))
        == 3
    )
    assert (
        _admit(runner, generator, _request("t2", group_id=1, routing_session_id="s1"))
        == 3
    )
    assert (
        _admit(runner, generator, _request("t3", group_id=2, routing_session_id="s2"))
        == 4
    )


def test_step_with_kv_reset_uses_current_version_without_pins(runner) -> None:
    generator = _bare_generator(
        event_loop=runner.get_loop(), reset_kv_cache_on_weight_sync=True
    )
    generator.policy_version = 3
    assert (
        _admit(runner, generator, _request("t0", group_id=1, routing_session_id="s0"))
        == 3
    )

    generator.policy_version = 4
    assert (
        _admit(runner, generator, _request("t1", group_id=1, routing_session_id="s0"))
        == 4
    )
    assert generator._group_min_policy_versions == {}


def test_release_groups_drops_pins(runner) -> None:
    generator = _bare_generator(event_loop=runner.get_loop())
    generator.policy_version = 3
    _admit(runner, generator, _request("t0", group_id=1, routing_session_id="s0"))
    _admit(runner, generator, _request("t1", group_id=2, routing_session_id="s1"))

    # Releasing a group this generator never served is a no-op.
    runner.run(generator.release_groups([1, 9]))

    assert generator._group_min_policy_versions == {2: 3}
    generator.policy_version = 4
    assert (
        _admit(runner, generator, _request("t2", group_id=1, routing_session_id="s0"))
        == 4
    )


# --- engine thread ---


def _finished_output(request_id: str) -> SimpleNamespace:
    """A vLLM `RequestOutput` for a request that finished with one token."""
    logprobs = FlatLogprobs()
    logprobs.append({7: Logprob(logprob=-0.5)})
    return SimpleNamespace(
        request_id=request_id,
        num_cached_tokens=0,
        metrics=SimpleNamespace(
            first_token_latency=0.01,
            queued_ts=1.0,
            scheduled_ts=1.0,
            first_token_ts=1.01,
            last_token_ts=1.02,
            num_generation_tokens=1,
        ),
        outputs=[
            SimpleNamespace(
                token_ids=[7],
                logprobs=logprobs,
                finish_reason="stop",
            )
        ],
    )


class _FakeEngine:
    """Every step finishes all admitted requests. `step_hook` runs inside each step;
    `threads` records which threads admitted and stepped."""

    def __init__(self, step_hook: Callable[[], None] = lambda: None):
        self.renderer = SimpleNamespace(
            render_cmpl=lambda prompts: prompts, shutdown=lambda: None
        )
        self.step_hook = step_hook
        self.running: list[str] = []
        self.threads: set[threading.Thread] = set()

    def add_request(self, *, request_id, prompt, params):
        self.threads.add(threading.current_thread())
        self.running.append(request_id)

    def has_unfinished_requests(self) -> bool:
        return bool(self.running)

    def step(self):
        self.threads.add(threading.current_thread())
        self.step_hook()
        finished, self.running = self.running, []
        return [_finished_output(request_id) for request_id in finished]


class _StuckEngine(_FakeEngine):
    """Never finishes a request."""

    def step(self):
        self.step_hook()
        return []


class _HeldEngine(_FakeEngine):
    """Keeps stepping without finishing a request until `finish` is set."""

    def __init__(self, step_hook: Callable[[], None] = lambda: None):
        super().__init__(step_hook)
        self.finish = threading.Event()

    def step(self):
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


def _start_engine_thread(generator: VLLMGenerator) -> None:
    """Start the engine thread and its event loop, as `VLLMGenerator.__init__` does."""
    generator._engine_event_loop = asyncio.new_event_loop()
    generator._engine_thread = threading.Thread(
        target=generator._engine_event_loop.run_forever,
        name="vllm-engine",
        daemon=True,
    )
    generator._engine_thread.start()


def _stop_engine_thread(generator: VLLMGenerator) -> None:
    # Production never stops the engine thread's event loop; tests do, to join the thread.
    event_loop = generator._engine_event_loop
    event_loop.call_soon_threadsafe(event_loop.stop)
    generator._engine_thread.join(timeout=_TIMEOUT_S)
    assert not generator._engine_thread.is_alive()
    event_loop.close()


@pytest.fixture
def engine_thread(monkeypatch):
    """Returns a function that builds a rank-0 generator around `engine`, with its
    engine thread running; stops those threads afterwards."""
    # Rank 0 is the only rank, so the decision broadcast is a no-op.
    monkeypatch.setattr(
        generator_module.dist, "broadcast_object_list", lambda *a, **k: None
    )
    generators: list[VLLMGenerator] = []

    def start(engine: _FakeEngine) -> VLLMGenerator:
        generator = _bare_generator()
        generator.config = SimpleNamespace(
            sampling=SamplingConfig(stop_token_ids=[]),
            max_engine_steps_between_decisions=16,
            reset_kv_cache_on_weight_sync=False,
            enable_cpu_weight_prefetch=False,
        )
        generator.policy_version = 0
        generator._rank = 0
        generator._broadcast_group = None
        generator._engine_loop_future = None
        _start_engine_thread(generator)
        generator._engine = engine
        generator._inbox = EngineLoopInbox(generator._engine_event_loop)
        generators.append(generator)
        return generator

    yield start
    for generator in generators:
        _stop_engine_thread(generator)


def _pulling_engine(monkeypatch, get_state_dict) -> _FakeEngine:
    """A fake engine whose weight pull reads TorchStore through `get_state_dict`."""
    monkeypatch.setattr(generator_module.ts, "get_state_dict", get_state_dict)
    monkeypatch.setattr(
        generator_module, "plain_tensor_to_dtensor_state_dict", lambda sd, **k: sd
    )
    monkeypatch.setattr(generator_module, "dtensor_to_plain_tensor_state_dict", dict)
    model = SimpleNamespace(
        model=SimpleNamespace(state_dict=dict, load_state_dict=lambda sd, strict: None),
        get_state_dict_layouts=dict,
        parallelism_context=None,
    )
    engine = _FakeEngine()
    engine.model_executor = SimpleNamespace(
        driver_worker=SimpleNamespace(get_model=lambda: model)
    )
    return engine


def _generate(generator: VLLMGenerator, request_id: str) -> asyncio.Task:
    return asyncio.create_task(
        generator.generate(
            [1, 2], request_id=request_id, group_id=0, routing_session_id=request_id
        )
    )


async def _on_engine_loop(generator: VLLMGenerator, fn: Callable[[], Any]) -> Any:
    """Run `fn` on the engine thread's event loop and return its result."""

    async def call() -> Any:
        return fn()

    return await asyncio.wrap_future(
        asyncio.run_coroutine_threadsafe(call(), generator._engine_event_loop)
    )


def test_call_on_engine_thread_runs_on_it_in_the_callers_context() -> None:
    caller_context = contextvars.ContextVar("caller_context", default=None)
    caller_context.set("caller")
    generator = _bare_generator()
    _start_engine_thread(generator)
    try:
        result = generator._call_on_engine_thread(
            lambda: (threading.current_thread(), caller_context.get())
        )
        assert generator._engine_thread is not threading.current_thread()
        assert result == (generator._engine_thread, "caller")
    finally:
        _stop_engine_thread(generator)


def test_call_on_engine_thread_raises_any_failure_to_the_caller() -> None:
    # Not an `Exception`: the caller must not wait forever on any failure, e.g. of the engine build.
    def fail():
        raise SystemExit("build failed")

    generator = _bare_generator()
    _start_engine_thread(generator)
    try:
        with pytest.raises(SystemExit, match="build failed"):
            generator._call_on_engine_thread(fail)
    finally:
        _stop_engine_thread(generator)


def test_close_releases_the_engine() -> None:
    built: list[weakref.ref[_FakeEngine]] = []
    shutdowns: list[threading.Thread] = []

    def build_engine() -> _FakeEngine:
        engine = _FakeEngine()
        engine.renderer.shutdown = lambda: shutdowns.append(threading.current_thread())
        built.append(weakref.ref(engine))
        return engine

    generator = _bare_generator()
    generator._rank = 0
    generator._engine_loop_future = None
    _start_engine_thread(generator)
    # Built as `__init__` builds it, so the engine thread must keep no reference to it.
    generator._engine = generator._call_on_engine_thread(build_engine)
    generator._inbox = EngineLoopInbox(generator._engine_event_loop)
    try:
        asyncio.run(asyncio.wait_for(generator.close(), _TIMEOUT_S))
        gc.collect()
        assert generator._engine_thread.is_alive()
        assert built[0]() is None
        assert shutdowns == [generator._engine_thread]
    finally:
        _stop_engine_thread(generator)


def test_actor_loop_takes_calls_while_the_engine_steps(engine_thread) -> None:
    gate = _StepGate()
    engine = _FakeEngine(gate)

    async def run() -> None:
        generator = engine_thread(engine)
        await generator.start_engine_loop()
        first = _generate(generator, "r0")
        # engine.step() blocks the engine thread, not this loop.
        assert await asyncio.to_thread(gate.entered.wait, _TIMEOUT_S)
        second = _generate(generator, "r1")
        gate.release()

        completions = await asyncio.wait_for(asyncio.gather(first, second), _TIMEOUT_S)
        assert [c.request_id for c in completions] == ["r0", "r1"]
        assert engine.threads == {generator._engine_thread}
        assert generator._request_dispatcher._rank0_generation_futures == {}

        await asyncio.wait_for(generator.close(), _TIMEOUT_S)
        assert generator._engine is None

    asyncio.run(run())


def test_pull_reads_torchstore_on_the_engine_thread(engine_thread, monkeypatch) -> None:
    # Monarch finds the calling actor through a ContextVar, so the engine loop's work must
    # see the context of the endpoint that started it.
    endpoint_context = contextvars.ContextVar("endpoint_context", default=None)
    reads: list[tuple] = []

    async def get_state_dict(*args, **kwargs):
        reads.append(
            (
                threading.current_thread(),
                asyncio.get_running_loop(),
                endpoint_context.get(),
            )
        )

    async def run() -> None:
        generator = engine_thread(_pulling_engine(monkeypatch, get_state_dict))
        # Set after the engine thread started, so only `start_engine_loop` can carry it there.
        endpoint_context.set("endpoint")
        await generator.start_engine_loop()

        await asyncio.wait_for(generator.pull_model_state_dict(4), _TIMEOUT_S)
        assert generator.policy_version == 4
        assert reads == [
            (generator._engine_thread, generator._engine_event_loop, "endpoint")
        ]

        completion = await asyncio.wait_for(_generate(generator, "r0"), _TIMEOUT_S)
        assert completion.min_policy_version == completion.max_policy_version == 4

        await asyncio.wait_for(generator.close(), _TIMEOUT_S)

    asyncio.run(run())


def test_prefetch_reads_torchstore_on_the_actor_loop_while_the_engine_steps(
    engine_thread, monkeypatch
) -> None:
    reads: list[tuple[threading.Thread, asyncio.AbstractEventLoop]] = []

    async def get_state_dict(*args, **kwargs):
        reads.append((threading.current_thread(), asyncio.get_running_loop()))

    gate = _StepGate()
    engine = _pulling_engine(monkeypatch, get_state_dict)
    engine.step_hook = gate

    async def run() -> None:
        generator = engine_thread(engine)
        generator.config.enable_cpu_weight_prefetch = True
        generator._prefetched_model_state_dict = {}
        await generator.start_engine_loop()
        first = _generate(generator, "r0")
        assert await asyncio.to_thread(gate.entered.wait, _TIMEOUT_S)

        await asyncio.wait_for(generator.prefetch_model_state_dict(), _TIMEOUT_S)
        actor_loop = (threading.current_thread(), asyncio.get_running_loop())
        assert reads == [actor_loop]

        gate.release()
        await asyncio.wait_for(first, _TIMEOUT_S)
        # The pull applies the prefetched weights without reading TorchStore again.
        await asyncio.wait_for(generator.pull_model_state_dict(4), _TIMEOUT_S)
        assert reads == [actor_loop]
        assert generator.policy_version == 4

        await asyncio.wait_for(generator.close(), _TIMEOUT_S)

    asyncio.run(run())


def test_pulls_queued_during_a_pull_are_applied_together_after_it(
    engine_thread, monkeypatch
) -> None:
    reading = [threading.Event(), threading.Event()]
    released = [threading.Event(), threading.Event()]
    reads: list[int] = []

    async def get_state_dict(*args, **kwargs):
        read = len(reads)
        reads.append(read)
        reading[read].set()
        assert await asyncio.to_thread(released[read].wait, _TIMEOUT_S)

    async def run() -> None:
        generator = engine_thread(_pulling_engine(monkeypatch, get_state_dict))
        await generator.start_engine_loop()
        first = asyncio.create_task(generator.pull_model_state_dict(3))
        assert await asyncio.to_thread(reading[0].wait, _TIMEOUT_S)

        later = [
            asyncio.create_task(generator.pull_model_state_dict(version))
            for version in (4, 5)
        ]
        await asyncio.sleep(0)  # both put their call on the inbox
        # Queued on the engine loop behind both puts, so both pulls are on the inbox once it returns.
        await _on_engine_loop(generator, lambda: None)
        released[0].set()
        await asyncio.wait_for(first, _TIMEOUT_S)
        assert generator.policy_version == 3

        # The first pull resolves only its own caller; the later two wait for one more read.
        assert await asyncio.to_thread(reading[1].wait, _TIMEOUT_S)
        assert not any(pull.done() for pull in later)
        released[1].set()
        await asyncio.wait_for(asyncio.gather(*later), _TIMEOUT_S)
        assert generator.policy_version == 5
        assert len(reads) == 2

        await asyncio.wait_for(generator.close(), _TIMEOUT_S)

    asyncio.run(run())


def test_crash_fails_outstanding_and_queued_calls_and_later_calls(
    engine_thread, monkeypatch
) -> None:
    reading, released = threading.Event(), threading.Event()

    async def get_state_dict(*args, **kwargs):
        reading.set()
        assert await asyncio.to_thread(released.wait, _TIMEOUT_S)
        raise RuntimeError("TorchStore is down")

    async def run() -> None:
        generator = engine_thread(_pulling_engine(monkeypatch, get_state_dict))
        await generator.start_engine_loop()
        failing_pull = asyncio.create_task(generator.pull_model_state_dict(3))
        assert await asyncio.to_thread(reading.wait, _TIMEOUT_S)

        queued = [
            _generate(generator, "r0"),
            asyncio.create_task(generator.pull_model_state_dict(4)),
        ]
        await asyncio.sleep(0)  # both put their call on the inbox
        # Queued on the engine loop behind both puts, so both are on the inbox once it returns.
        await _on_engine_loop(generator, lambda: None)
        released.set()

        for call in (failing_pull, *queued):
            with pytest.raises(RuntimeError, match="TorchStore is down"):
                await asyncio.wait_for(call, _TIMEOUT_S)
        with pytest.raises(RuntimeError, match="generator is closed"):
            await asyncio.wait_for(_generate(generator, "r1"), _TIMEOUT_S)
        # `close` only logs the loop's error, so check that failing the inbox didn't replace it.
        exc = generator._engine_loop_future.exception(timeout=_TIMEOUT_S)
        assert isinstance(exc, RuntimeError) and str(exc) == "TorchStore is down"
        assert generator._engine is None  # released by the loop on its way out
        await asyncio.wait_for(generator.close(), _TIMEOUT_S)

    asyncio.run(run())


def test_generate_cancelled_on_the_inbox_never_reaches_the_engine(
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
        generator = engine_thread(engine)
        await generator.start_engine_loop()
        first = _generate(generator, "r0")
        assert await asyncio.to_thread(gate.entered.wait, _TIMEOUT_S)
        cancelled = _generate(generator, "r1")
        # r1 is put on the inbox, which the held engine thread can't drain.
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        gate.release()

        await asyncio.wait_for(first, _TIMEOUT_S)
        await asyncio.wait_for(_generate(generator, "r2"), _TIMEOUT_S)
        assert stepped == [["r0"], ["r2"]]
        assert generator._request_dispatcher._rank0_generation_futures == {}

        await asyncio.wait_for(generator.close(), _TIMEOUT_S)

    asyncio.run(run())


def test_generate_cancelled_after_admission_leaves_the_loop_running(
    engine_thread,
) -> None:
    stepping = threading.Event()
    engine = _HeldEngine(stepping.set)

    async def run() -> None:
        generator = engine_thread(engine)
        await generator.start_engine_loop()
        cancelled = _generate(generator, "r0")
        assert await asyncio.to_thread(stepping.wait, _TIMEOUT_S)  # r0 is admitted
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        engine.finish.set()

        # r0 still finishes, and its reply is resolved without crashing the loop.
        completion = await asyncio.wait_for(_generate(generator, "r1"), _TIMEOUT_S)
        assert completion.request_id == "r1"
        assert generator._request_dispatcher._rank0_generation_futures == {}

        await asyncio.wait_for(generator.close(), _TIMEOUT_S)

    asyncio.run(run())


def test_close_fails_outstanding_requests_and_later_calls(engine_thread) -> None:
    gate = _StepGate()

    async def run() -> None:
        generator = engine_thread(_StuckEngine(gate))
        await generator.start_engine_loop()
        engine_loop_future = generator._engine_loop_future
        in_flight = _generate(generator, "r0")
        assert await asyncio.to_thread(gate.entered.wait, _TIMEOUT_S)
        queued = _generate(generator, "r1")
        queued_pull = asyncio.create_task(generator.pull_model_state_dict(4))
        closing = asyncio.create_task(generator.close())
        # Runs after `close` has closed the inbox, while the loop still runs, so the guard rejects it
        # before it reaches the inbox.
        during_close = _generate(generator, "r2")
        # Run all four calls while `step` holds the engine thread, so the loop takes the pull and the
        # `CloseRequest` off the inbox together; alone, the pull would run on `_StuckEngine`.
        await asyncio.sleep(0)
        # `close` and the rejected call ran on this loop, without waiting for the engine thread.
        assert generator._inbox.closed and during_close.done()
        gate.release()

        await asyncio.wait_for(closing, _TIMEOUT_S)
        assert engine_loop_future.done()
        assert generator._engine is None
        for request in (in_flight, queued):
            with pytest.raises(
                RuntimeError, match="closed before the request finished"
            ):
                await request
        with pytest.raises(RuntimeError, match="closed before the pull was applied"):
            await asyncio.wait_for(queued_pull, _TIMEOUT_S)
        with pytest.raises(
            RuntimeError, match="generator is closed; cannot call generate"
        ):
            await during_close

        # Later calls fail instead of hanging, and a second `close` still returns.
        with pytest.raises(RuntimeError, match="generator is closed"):
            await asyncio.wait_for(_generate(generator, "r3"), _TIMEOUT_S)
        with pytest.raises(RuntimeError, match="generator is closed"):
            await asyncio.wait_for(generator.pull_model_state_dict(4), _TIMEOUT_S)
        await asyncio.wait_for(generator.close(), _TIMEOUT_S)

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
                action=LoopAction.PULL_MODEL_STATE_DICT,
                requests_per_dp_rank=[],
                pull_version=4,
            ),
            LoopDecision(action=LoopAction.CLOSE, requests_per_dp_rank=[]),
        ]
    )

    def broadcast_object_list(container, **kwargs):
        # A follower sends nothing and receives rank 0's decision.
        assert container == [None]
        container[0] = next(decisions)

    monkeypatch.setattr(
        generator_module.dist, "broadcast_object_list", broadcast_object_list
    )
    reads: list[threading.Thread] = []

    async def get_state_dict(*args, **kwargs):
        reads.append(threading.current_thread())

    async def run() -> None:
        engine = _pulling_engine(monkeypatch, get_state_dict)
        stepped: list[list[str]] = []
        engine.step_hook = lambda: stepped.append(list(engine.running))
        generator = engine_thread(engine)
        # TP rank 1 of a DP=1, TP=2 generator.
        generator._rank = 1
        generator._request_dispatcher = RequestDispatcher(
            rank=1,
            dp_rank=0,
            tp_rank=1,
            dp_degree=1,
            broadcast_group=None,
            open_result_channel=None,
            intra_generator_router=IntraGeneratorRouter.Config(
                strategy=LeastLoadedRoutingStrategy.Config()
            ),
        )
        await generator.start_engine_loop()

        # The broadcast CLOSE ends the loop; a follower's close only waits for that.
        await asyncio.wait_for(generator.close(), _TIMEOUT_S)
        assert stepped == [["r0"]]
        assert engine.threads == {generator._engine_thread}
        assert reads == [generator._engine_thread]
        assert generator.policy_version == 4

    asyncio.run(run())


def test_drain_task_resolves_peer_completions_on_the_engine_thread(
    engine_thread, monkeypatch
) -> None:
    peer_results: asyncio.Queue = asyncio.Queue()
    recv_threads: list[threading.Thread] = []

    class ResultReceiver:
        async def recv(self):
            recv_threads.append(threading.current_thread())
            return await peer_results.get()

    async def run() -> None:
        engine = _FakeEngine()
        generator = engine_thread(engine)
        dispatcher = RequestDispatcher(
            rank=0,
            dp_rank=0,
            tp_rank=0,
            dp_degree=2,
            broadcast_group=None,
            open_result_channel=lambda: ("port", ResultReceiver()),
            intra_generator_router=IntraGeneratorRouter.Config(
                strategy=LeastLoadedRoutingStrategy.Config()
            ),
        )
        # Load DP rank 0, so the request goes to the peer, DP rank 1.
        dispatcher._rank0_dp_router.reserve("busy", routing_session_id="busy")
        generator._request_dispatcher = dispatcher
        peer_requests: list[str] = []

        def broadcast_object_list(container, **kwargs):
            # Stand-in for DP rank 1, which meets rank 0 at every decision broadcast: it sends
            # back the completions of the requests it admitted at the previous one, then admits
            # its share of this one.
            if peer_requests:
                completions = dispatcher._build_completions(
                    [_finished_output(request_id) for request_id in peer_requests], 0
                )
                peer_requests.clear()
                generator._engine_event_loop.call_soon_threadsafe(
                    peer_results.put_nowait, completions
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
            generator_module.dist, "broadcast_object_list", broadcast_object_list
        )
        await generator.start_engine_loop()

        completion = await asyncio.wait_for(_generate(generator, "r0"), _TIMEOUT_S)
        assert completion.request_id == "r0"
        assert engine.threads == set()  # rank 0's own DP replica served nothing
        assert set(recv_threads) == {generator._engine_thread}

        await asyncio.wait_for(generator.close(), _TIMEOUT_S)

    asyncio.run(run())
