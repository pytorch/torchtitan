# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the generator engine loop's decision logic (`_decide_next_action`).

Built on a bare `VLLMGenerator` (no vLLM engine) + a fake engine, so the loop's
admit/pull/shutdown branching is tested without a GPU.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from types import SimpleNamespace

import torch

import torchtitan.rl.generator as generator_module

from torchtitan.rl.distributed.routing.intra_generator import IntraGeneratorRouter
from torchtitan.rl.distributed.routing.strategies import (
    LeastLoadedRoutingStrategy,
    RoutingStrategy,
    StickySessionRoutingStrategy,
)

from torchtitan.rl.generator import (
    _initialize_engine_step_thread,
    CloseRequest,
    GenerationRequest,
    LoopAction,
    ModelStateDictPullRequest,
    RequestDispatcher,
    SamplingConfig,
    VLLMGenerator,
)


def _bare_generator(
    *,
    close_requested: bool = False,
    model_state_dict_pull_request: ModelStateDictPullRequest | None = None,
    pending: list[GenerationRequest] | None = None,
    inflight: bool = False,
    dp_size: int = 1,
    dp_routing_strategy: RoutingStrategy.Config | None = None,
) -> VLLMGenerator:
    # Bypass __init__ (which builds the vLLM engine); set only the loop's state.
    # _decide_next_action delegates the in-flight check and routing to the
    # dispatcher, so wire up a bare one (no engine / GPU needed here).
    generator = object.__new__(VLLMGenerator)
    generator._engine_loop_condition = asyncio.Condition()
    generator._close_request = CloseRequest() if close_requested else None
    generator._model_state_dict_pull_request = model_state_dict_pull_request
    generator._queued_generation_requests = pending or []
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
    routing_session_id: str | None = None,
) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        prompt_token_ids=[1, 2],
        sampling=SamplingConfig(),
        routing_session_id=routing_session_id or request_id,
    )


def test_closing_returns_close() -> None:
    decision = asyncio.run(_bare_generator(close_requested=True)._decide_next_action())
    assert decision.action is LoopAction.CLOSE


def test_pull_takes_precedence_over_queued_requests() -> None:
    request = _request()
    pull = ModelStateDictPullRequest(version=5)
    generator = _bare_generator(model_state_dict_pull_request=pull, pending=[request])
    decision = asyncio.run(generator._decide_next_action())
    assert (
        decision.action is LoopAction.PULL_MODEL_STATE_DICT
        and decision.pull_version == 5
    )
    # `_model_state_dict_pull_request` is NOT cleared at decide — the PULL_MODEL_STATE_DICT branch clears it after
    # applying; the single-threaded loop can't re-decide before then, so the predicate won't re-fire.
    assert generator._model_state_dict_pull_request is pull
    assert generator._queued_generation_requests == [
        request
    ]  # NOT consumed — pull runs first


def test_step_drains_the_queue() -> None:
    request = _request()
    generator = _bare_generator(pending=[request])
    decision = asyncio.run(generator._decide_next_action())
    # DP=1: a single DP rank holds the whole batch.
    assert decision.action is LoopAction.STEP and decision.requests_per_dp_rank == [
        [request]
    ]
    assert generator._queued_generation_requests == []  # drained into the decision
    assert generator._request_dispatcher._rank0_dp_router is None


def test_step_with_empty_queue_when_only_in_flight_work_remains() -> None:
    # No queue, no pull, but a registered future means a request is still in flight
    # (possibly in a peer DP rank), so rank 0 must keep issuing STEP.
    decision = asyncio.run(_bare_generator(inflight=True)._decide_next_action())
    assert decision.action is LoopAction.STEP and decision.requests_per_dp_rank == [[]]


def test_step_routes_requests_across_dp_ranks() -> None:
    # Least-loaded over 3 idle DP ranks: r0 -> rank 0, r1 -> rank 1 (rank 0 now loaded).
    requests = [_request("r0"), _request("r1")]
    generator = _bare_generator(pending=requests, dp_size=3)
    decision = asyncio.run(generator._decide_next_action())
    assert decision.action is LoopAction.STEP
    assert decision.requests_per_dp_rank == [[requests[0]], [requests[1]], []]
    # Each request reserves one load unit on its chosen DP rank.
    dp_router = generator._request_dispatcher._rank0_dp_router
    assert dp_router._reservations == {"r0": 0, "r1": 1}
    assert [h.reserved_load for h in dp_router._handles] == [1, 1, 0]


def test_step_sticky_session_reuses_dp_rank() -> None:
    first = _request("r0", routing_session_id="s0")
    generator = _bare_generator(
        pending=[first],
        dp_size=3,
        dp_routing_strategy=StickySessionRoutingStrategy.Config(),
    )

    first_decision = asyncio.run(generator._decide_next_action())
    assert first_decision.action is LoopAction.STEP
    assert first_decision.requests_per_dp_rank == [[first], [], []]

    same_session = _request("r1", routing_session_id="s0")
    new_session = _request("r2", routing_session_id="s1")
    generator._queued_generation_requests = [same_session, new_session]

    second_decision = asyncio.run(generator._decide_next_action())
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


def test_engine_step_does_not_block_generate_admission() -> None:
    """A long vLLM step must not block actor endpoint work."""

    async def run() -> None:
        entered_step = threading.Event()
        release_step = threading.Event()

        class BlockingEngine:
            def step(self):
                entered_step.set()
                release_step.wait()
                return []

        generator = _bare_generator()
        generator._rank = 0
        generator._engine = BlockingEngine()
        generator._engine_loop_task = object()
        generator.config = SimpleNamespace(sampling=SamplingConfig())
        generator._engine_step_executor = ThreadPoolExecutor(max_workers=1)

        # Release a regressed inline step so the test fails instead of hanging.
        watchdog = threading.Timer(5.0, release_step.set)
        watchdog.start()
        step_task = asyncio.create_task(generator._run_engine_step())
        generate_task = None
        try:
            await asyncio.wait_for(asyncio.to_thread(entered_step.wait), timeout=5.0)
            generate_task = asyncio.create_task(
                generator.generate(
                    [1, 2],
                    request_id="admitted-while-step-blocked",
                    routing_session_id="session",
                )
            )
            await asyncio.sleep(0)
            admitted_before_step_finished = [
                request.request_id for request in generator._queued_generation_requests
            ] == ["admitted-while-step-blocked"] and not release_step.is_set()
        finally:
            release_step.set()
            await step_task
            if generate_task is not None:
                generate_task.cancel()
                with suppress(asyncio.CancelledError):
                    await generate_task
            watchdog.cancel()
            generator._engine_step_executor.shutdown(wait=True)

        assert admitted_before_step_finished

    asyncio.run(run())


def test_engine_step_worker_initializes_thread_local_state(monkeypatch) -> None:
    state = threading.local()
    mesh = object()
    device_calls: list[tuple[int, int]] = []

    @contextmanager
    def mesh_context():
        state.device_mesh_stack = [mesh]
        try:
            yield
        finally:
            state.device_mesh_stack.pop()

    @contextmanager
    def spmd_context(*, parallel_dims):
        state.spmd_stack = [parallel_dims]
        try:
            yield
        finally:
            state.spmd_stack.pop()

    parallel_dims = SimpleNamespace(spmd_dense_mesh=mesh_context)

    monkeypatch.setattr(
        torch.accelerator,
        "set_device_index",
        lambda device_index: device_calls.append((device_index, threading.get_ident())),
    )
    monkeypatch.setattr(generator_module, "get_spmd_context", spmd_context)

    with ThreadPoolExecutor(
        max_workers=1,
        initializer=_initialize_engine_step_thread,
        initargs=(3, parallel_dims),
    ) as executor:
        device_mesh_stack, spmd_stack, worker_thread = executor.submit(
            lambda: (
                list(state.device_mesh_stack),
                list(state.spmd_stack),
                threading.get_ident(),
            )
        ).result()

    assert device_calls == [(3, worker_thread)]
    assert device_mesh_stack == []
    assert spmd_stack == []


def test_engine_step_worker_serializes_engine_access() -> None:
    async def run() -> None:
        first_step_entered = threading.Event()
        release_first_step = threading.Event()
        event_loop_grad_enabled = torch.is_grad_enabled()
        calls: list[int] = []
        grad_enabled: list[bool] = []

        class RecordingEngine:
            def step(self):
                calls.append(threading.get_ident())
                grad_enabled.append(torch.is_grad_enabled())
                if len(calls) == 1:
                    first_step_entered.set()
                    release_first_step.wait()
                return []

        generator = _bare_generator()
        generator._engine = RecordingEngine()
        generator._engine_step_executor = ThreadPoolExecutor(max_workers=1)

        first = asyncio.create_task(generator._run_engine_step())
        second = None
        watchdog = threading.Timer(5.0, release_first_step.set)
        watchdog.start()
        try:
            await asyncio.wait_for(
                asyncio.to_thread(first_step_entered.wait), timeout=5.0
            )
            second = asyncio.create_task(generator._run_engine_step())
            await asyncio.sleep(0)
            calls_before_release = len(calls)
        finally:
            release_first_step.set()
            tasks = [first] if second is None else [first, second]
            results = await asyncio.gather(*tasks)
            watchdog.cancel()
            generator._engine_step_executor.shutdown(wait=True)

        assert calls_before_release == 1
        assert results == [[], []]
        assert len(set(calls)) == 1
        assert grad_enabled == [False, False]
        assert torch.is_grad_enabled() == event_loop_grad_enabled

    asyncio.run(run())


def test_close_waits_for_engine_loop_before_releasing_engine() -> None:
    async def run() -> None:
        events: list[str] = []
        release_loop = asyncio.Event()

        async def finish_engine_loop() -> None:
            await release_loop.wait()
            events.append("engine-loop")

        class RecordingExecutor:
            def shutdown(self, *, wait, cancel_futures):
                assert wait and cancel_futures
                events.append("executor")

        class RecordingDispatcher:
            async def shutdown(self):
                events.append("dispatcher")

        class RecordingRenderer:
            def shutdown(self):
                events.append("renderer")

        generator = _bare_generator()
        generator._rank = 1
        generator._engine_loop_task = asyncio.create_task(finish_engine_loop())
        generator._engine_step_executor = RecordingExecutor()
        generator._request_dispatcher = RecordingDispatcher()
        generator._fail_outstanding_futures = lambda exc: events.append("futures")
        generator._engine = SimpleNamespace(renderer=RecordingRenderer())

        close_task = asyncio.create_task(generator.close())
        await asyncio.sleep(0)
        assert events == []
        release_loop.set()
        await close_task

        assert events == [
            "engine-loop",
            "executor",
            "dispatcher",
            "futures",
            "renderer",
        ]
        assert generator._engine is None

    asyncio.run(run())
