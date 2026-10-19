# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The generator's SPMD engine loop and the thread it runs on."""

from __future__ import annotations

import asyncio
import concurrent.futures
import enum
import gc
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

import torch
import torch.distributed as dist

from torchtitan.observability import structured_logger as sl
from torchtitan.rl.generator.dispatcher import RequestDispatcher
from torchtitan.rl.generator.engine import GenerationEngine
from torchtitan.rl.generator.types import EngineRequest
from torchtitan.rl.rollout.types import Completion

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class EngineLoopThread:
    """A daemon thread running its own event loop, where the engine and the engine loop live.

    `engine.step()` blocks, so the engine and everything the engine loop touches run here,
    keeping the caller's (e.g. an actor's) event loop free to take calls while the engine steps.

    Args:
        name: The thread's name.
    """

    def __init__(self, name: str):
        self.event_loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self.event_loop.run_forever, name=name, daemon=True
        )
        self.thread.start()

    def call(self, fn: Callable[[], _T]) -> _T:
        """Call `fn` on the engine loop thread, in the caller's contextvars, and block until it returns.
        For setup, before the engine loop starts."""
        result: concurrent.futures.Future[_T] = concurrent.futures.Future()

        def call() -> None:
            # A plain callback rather than a task, which would re-raise a `SystemExit` out of
            # `run_forever` before resolving its future, leaving the caller waiting forever.
            try:
                result.set_result(fn())
            except BaseException as exc:
                result.set_exception(exc)

        # `call_soon_threadsafe` runs `call` in a copy of the caller's contextvars.
        self.event_loop.call_soon_threadsafe(call)
        return result.result()


class EngineLoop:
    """Drives concurrent `generate` calls and model state dict loads through one SPMD engine loop.

    Rank 0 puts each call on a thread-safe queue and awaits its reply. One background loop per
    rank executes the `LoopDecision` rank 0 makes from the queue. Rank 0 resolves each reply when
    its request finishes.

    Notice that the engine step, which is a TP collective, and the request intake are decoupled,
    so a new request can join mid-flight, instead of waiting for the current batch to drain.

    One loop iteration, the `RequestDispatcher` dispatches requests to different ranks and
    collects results. Take DP=1, TP=2 for example, after the caller fired generate(prompt_0) and
    generate(prompt_1):

        # request intake in `generate` takes a prompt, puts in a queue, releases control back to the caller
        generate(prompt_0): enqueue prompt_0, await reply_0   ┐ rank 0 owns the queue + replies
        generate(prompt_1): enqueue prompt_1, await reply_1   ┘ (other ranks are no-op)

        # meanwhile, the engine loop, which is its own coroutine, is continuously running.
        rank 0   _decide_next_action -> LoopDecision(STEP, [prompt_0, prompt_1])─┐  broadcast_object_list (gloo)
        rank 1   (blocked inside the broadcast) ─────────────────────────────────┘  decision from rank0 is broadcast

        # The loop now executes the decision. In this example: STEP.
        ALL      engine.add_requests([prompt_0, prompt_1])
        ALL      engine.step() * max_engine_steps_between_decisions # N step burst before a new decision

        # resolve the reply, waking up `generate` so it returns the result to the caller.
        # Note that prompt_1 can be done before prompt_0. The result is per request, not per batch.
        rank 0   request_dispatcher.process_finished_requests -> prompt_1 done? reply_1.set_result(Completion)
        rank 1   request_dispatcher.process_finished_requests -> no-op (tp_rank != 0, holds no replies)

    For DP>1, the requests will be routed among DPs first. See RequestDispatcher's docstring for more details.

    Threading: the engine and everything the loop touches (queue, replies, dispatcher) live on
    the engine loop thread, whose event loop is `event_loop`. `generate` and `load_model_state_dict`
    run on the caller's event loop and reach the loop only through the thread-safe queue and the
    `concurrent.futures.Future`s it resolves. `release_groups` is an exception: it pops cache-salt
    pins directly; each pop, like the loop's `setdefault`, is a single dict operation, atomic
    under the GIL.

    A model state dict load rides the same loop: `load_model_state_dict` puts a
    `ModelStateDictLoadMessage` on the queue, which rank 0 turns into a
    `LoopDecision(LoopAction.LOAD_MODEL_STATE_DICT)` applied between step bursts. The engine does
    NOT drain in-flight requests first ("hotswap"). This behavior can be changed in the
    inter-generator router, by blocking new requests until the engine is drained. The network
    transfer into the engine's pinned CPU buffers happens before this loop action, which then
    performs the local CPU-to-GPU copy.

    Args:
        engine: The inference engine, built on the engine loop thread. The loop releases it on exit.
        event_loop: The engine loop thread's event loop.
        request_dispatcher: Dispatches requests across DP replicas and resolves their replies.
        rank: Global rank of this process.
        broadcast_group: All-ranks gloo group that `LoopDecision`s are broadcast over.
        max_engine_steps_between_decisions: ``Generator.Config``'s field.
        reset_kv_cache_on_model_state_dict_load: ``Generator.Config``'s field.
    """

    def __init__(
        self,
        engine: GenerationEngine,
        *,
        event_loop: asyncio.AbstractEventLoop,
        request_dispatcher: RequestDispatcher,
        rank: int,
        broadcast_group: dist.ProcessGroup,
        max_engine_steps_between_decisions: int,
        reset_kv_cache_on_model_state_dict_load: bool,
    ):
        self._engine: GenerationEngine | None = engine
        self._event_loop = event_loop
        self._request_dispatcher = request_dispatcher
        self._rank = rank
        self._broadcast_group = broadcast_group
        self._max_engine_steps_between_decisions = max_engine_steps_between_decisions
        self._reset_kv_cache_on_model_state_dict_load = (
            reset_kv_cache_on_model_state_dict_load
        )
        # Read once: the engine is released on exit, and only the engine loop thread uses it.
        self._holds_session_kv = engine.holds_session_kv

        self.policy_version = 0
        # RANK 0: group id -> min policy version the group is pinned to, set at the
        # group's first admission and used as its prefix cache salt. Unused with
        # reset_kv_cache_on_model_state_dict_load. All rollouts of a group share the pin, so a
        # rollout first admitted after a load still reuses its group's prompt KV, at the
        # cost of depending on the group's older version. Only the controller knows when
        # a group makes no more generation calls, so entries live until it calls
        # `release_groups`.
        self._group_min_policy_versions: dict[int, int] = {}

        # Rank 0: messages `generate` and `load_model_state_dict` put; the loop takes them off to decide.
        self._queue = EngineLoopQueue(event_loop)

        # `_run` running on the engine loop thread's event loop, as a future any thread can await;
        # None until `start` starts it.
        self._future: concurrent.futures.Future[None] | None = None

    def start(self) -> None:
        """Start the loop on the engine loop thread (one-time, idempotent)."""
        if self._future is None:
            self._future = asyncio.run_coroutine_threadsafe(
                self._run(), self._event_loop
            )

    async def wait(self) -> None:
        """Wait until the loop exits, raising whatever crashed it."""
        if self._future is None:
            raise RuntimeError(
                "engine loop not started; call start_engine_loop before wait_engine_loop"
            )
        await asyncio.wrap_future(self._future)

    async def generate(
        self, engine_request: EngineRequest, *, metrics_prefix: str
    ) -> Completion:
        """RANK 0: queue `engine_request` and wait for its `Completion`. If the engine rejects the
        request at admission, only this call raises the engine's error."""
        self._rank0_check_running("generate")
        # Put the call on the queue; the loop will admit + process it, then resolve `reply`.
        reply: concurrent.futures.Future[Completion] = concurrent.futures.Future()
        self._queue.put(
            GenerationMessage(
                engine_request=engine_request,
                metrics_prefix=metrics_prefix,
                reply=reply,
            )
        )
        return await asyncio.wrap_future(reply)

    async def load_model_state_dict(self, version: int) -> None:
        """RANK 0: queue a model state dict load for `version` and wait until the loop has applied
        it. Loads queued together are applied once, at the highest version."""
        self._rank0_check_running("load_model_state_dict")
        reply: concurrent.futures.Future[None] = concurrent.futures.Future()
        self._queue.put(ModelStateDictLoadMessage(version=version, reply=reply))
        await asyncio.wrap_future(reply)

    def release_groups(self, group_ids: list[int]) -> None:
        """RANK 0: drop the pinned cache salts of finished rollout groups."""
        # A pop can land partway through `_decide_next_action`'s stamping loop, so requests of a
        # released group in one batch can get different versions. If every batch must see one
        # consistent snapshot of the pins, send releases through the queue instead.
        for group_id in group_ids:
            self._group_min_policy_versions.pop(group_id, None)
        if self._holds_session_kv and self._rank == 0:
            self._queue.put(SessionReleaseMessage(group_ids=group_ids))

    def release_sessions(self, session_ids: list[str]) -> None:
        """RANK 0: release the held KV of sessions whose rollout ended (a no-op unless the
        engine holds session KV)."""
        if self._holds_session_kv and self._rank == 0:
            self._queue.put(SessionReleaseMessage(session_ids=session_ids))

    def log_engine_stats(self, step: int) -> None:
        """Log the engine's per-step stats, if it has any (rank 0)."""
        engine = self._engine
        if engine is not None and self._rank == 0:
            engine.log_stats(step)

    async def close(self) -> None:
        """Stop the loop, which releases the dispatcher and the engine on exit.

        Rank 0 closes the queue: calls already queued stay ahead of the `CloseMessage` that makes
        the loop quit, and later calls raise.
        """
        if self._rank == 0:
            self._queue.close(CloseMessage(), reason="generator is closed")

        if self._future is None:
            # The loop never started (or an earlier `close` already awaited it), so release here, on
            # the engine loop thread, which makes every engine call.
            await asyncio.wrap_future(
                asyncio.run_coroutine_threadsafe(
                    self._release_resources(), self._event_loop
                )
            )
            return
        try:
            await asyncio.wrap_future(self._future)
        except Exception:
            logger.exception("engine loop raised during shutdown")
        self._future = None

    def _rank0_check_running(self, endpoint_name: str) -> None:
        """Guard for the rank-0-only calls"""
        assert self._rank == 0, f"{endpoint_name} must be routed to rank 0 only"
        if self._queue.closed:
            raise RuntimeError(f"generator is closed; cannot call {endpoint_name}")
        if self._future is None:
            raise RuntimeError(
                "engine loop not started; call start_engine_loop on all ranks "
                f"before {endpoint_name}"
            )

    @sl.log_trace_span("engine_loop")
    async def _run(self) -> None:
        """Non-stop loop running on all ranks to produce new tokens.

        Rank 0 decides a `LoopDecision` and broadcasts it; ALL ranks apply it in
        lockstep until `LoopAction.CLOSE`. On crash, fail every outstanding reply so callers don't hang. On exit,
        release the dispatcher and the engine.

        `_decide_next_action` is consulted once every `max_engine_steps_between_decisions` steps (a burst),
        so new requests buffer and prefill together instead of on every step.

        Example:
            max_engine_steps_between_decisions = 16
            check `_decide_next_action` --> "STEP"         --> run engine.step 16 times
            check `_decide_next_action` --> "LOAD_MODEL_STATE_DICT" --> run `_load_model_state_dict`
            check `_decide_next_action` --> "STEP"         --> run engine.step 16 times
            check `_decide_next_action` --> "CLOSE"        --> stop
        """
        # Loop state (rank 0): engine requests taken off the queue but not yet put in a `LoopAction.STEP`.
        # We need to cache such requests in this field, so they can be carried across `_decide_next_action` calls.
        # Note these requests' parent `GenerationMessage`s have been registered as `OutstandingGeneration` in dispatcher.
        pending_engine_requests: list[EngineRequest] = []
        # Loop state (rank 0): model state dict loads taken off the queue but not yet applied.
        pending_load_messages: list[ModelStateDictLoadMessage] = []
        try:
            # One-time dispatcher setup before the loop starts.
            self._request_dispatcher.setup()
            while True:
                # Rank 0 decides next decision; followers pass None and learn from the broadcast.
                decision = (
                    await self._decide_next_action(
                        pending_engine_requests, pending_load_messages
                    )
                    if self._rank == 0
                    else None
                )

                # Barrier(gloo, CPU): Ship rank 0's decision (incl. prompts) to every TP rank via gloo/CPU, off the
                # NCCL stream. broadcast_object_list mutates a list in place; `to_thread` runs the
                # blocking call in a worker so the event loop can still serve generate/load/close.
                # TODO(perf): overlap this broadcast with the step burst (pipeline the next decision) so
                # gloo transfer hides behind GPU compute.
                # TODO(perf): swap broadcast_object_list (double serialization of pickle+broadcast) for a byte
                # broadcast, to cut pickle overhead. i.e. serialize the decision to bytes on rank 0, then broadcast.
                # TODO: Revisit when enabling DP>1
                decision_broadcast_container = [decision] if self._rank == 0 else [None]
                await asyncio.to_thread(
                    dist.broadcast_object_list,
                    decision_broadcast_container,
                    src=0,
                    group=self._broadcast_group,
                    device=torch.device("cpu"),
                )
                # get rank 0's broadcasted decision
                decision = decision_broadcast_container[0]  # [num_ranks]

                if decision.action is LoopAction.CLOSE:
                    if self._rank == 0:
                        _fail_loads(
                            pending_load_messages,
                            RuntimeError(
                                "generator closed before the load was applied"
                            ),
                        )
                    return

                self._engine.release_sessions(
                    decision.released_session_ids, decision.released_group_ids
                )

                if decision.action is LoopAction.LOAD_MODEL_STATE_DICT:
                    await self._load_model_state_dict(decision.load_version)
                    # One load serves every load this decision coalesced (only rank 0 holds any).
                    for load in pending_load_messages:
                        load.reply.set_result(None)
                    pending_load_messages.clear()
                    continue  # back to the start for the next decision

                if decision.action is LoopAction.STEP:
                    # Rank 0 holds every outstanding generation, so it stamps the admitted (min) version for the
                    # whole decision.
                    # TODO: move under the engine_step call (register at generation_start, not admission).
                    # The way to do it is probably to change to RequestOutputKind.CUMULATIVE and mark per token.
                    if self._rank == 0:
                        self._request_dispatcher.rank0_stamp_min_policy_version(
                            decision.requests_per_dp_rank
                        )
                    # Admit only this rank's DP replica slice. TP ranks in the same
                    # replica compute the same dp_rank, so they add the identical
                    # set in the same FCFS order.
                    rejected_requests = self._engine.add_requests(
                        decision.requests_per_dp_rank[self._request_dispatcher.dp_rank]
                    )
                    if rejected_requests:
                        self._request_dispatcher.process_rejected_requests(
                            rejected_requests
                        )

                # Barrier (NCCL): engine.step() runs SPMD in lockstep.
                # The step burst `max_engine_steps_between_decisions` gives the generator time to buffer
                # new requests and avoid a prefill in between every engine decode step, which is inefficient.
                with sl.log_trace_span("vllm_engine_step_burst"):
                    for _ in range(self._max_engine_steps_between_decisions):
                        if not self._engine.has_unfinished_requests():
                            break
                        finished_requests = self._engine.step()
                        self._request_dispatcher.process_finished_requests(
                            finished_requests, self.policy_version
                        )
                        await asyncio.sleep(0)  # let pending generate() calls enqueue

        except Exception as exc:
            logger.exception("engine loop crashed; failing all outstanding replies")
            if self._rank == 0:
                self._request_dispatcher.fail_outstanding_generations(exc)
                _fail_loads(pending_load_messages, exc)
                await self._rank0_close_and_fail_queue(exc)
            raise
        finally:
            await self._release_resources()

    async def _rank0_close_and_fail_queue(self, exc: Exception) -> None:
        """RANK 0: after the loop crashed, close the queue so later calls raise, and fail the
        replies of the calls still on it."""
        self._queue.close(CloseMessage(), reason="engine loop crashed")
        # Puts that beat the close have scheduled their enqueue ahead of this task's next step.
        await asyncio.sleep(0)
        while not self._queue.empty():
            message = self._queue.get_nowait()
            if isinstance(message, (CloseMessage, SessionReleaseMessage)):
                continue
            if message.reply.set_running_or_notify_cancel():
                message.reply.set_exception(exc)

    async def _decide_next_action(
        self,
        pending_engine_requests: list[EngineRequest],
        pending_load_messages: list[ModelStateDictLoadMessage],
    ) -> LoopDecision:
        """RANK 0: takes everything off the queue and picks the next action. Sleeps until there is
        something to do.
        """
        if pending_load_messages:
            raise AssertionError(
                "the engine loop applies or fails every load before deciding again"
            )

        # * If there are outstanding generations, we drain the queue and make the next decision right away,
        #   so outstanding generations can keep stepping.
        # * If there are none, there is nothing to step, so we wait on the queue for the next decision.
        messages: list[EngineLoopMessage]
        if self._request_dispatcher.rank0_has_outstanding_generations():
            messages = []
        else:
            if pending_engine_requests:
                raise AssertionError(
                    f"{len(pending_engine_requests)} pending engine requests but no outstanding "
                    "generation"
                )
            messages = [await self._queue.get()]

        # Take everything currently in the queue if there is any, so they all can be part of this decision.
        while not self._queue.empty():
            messages.append(self._queue.get_nowait())

        released_session_ids: list[str] = []
        released_group_ids: list[int] = []
        for message in messages:
            if isinstance(message, CloseMessage):
                # Drops nothing: the queue rejects puts once closed, so the `CloseMessage` is the last message.
                return LoopDecision(action=LoopAction.CLOSE, requests_per_dp_rank=[])
            if isinstance(message, SessionReleaseMessage):
                released_session_ids.extend(message.session_ids)
                released_group_ids.extend(message.group_ids)
                continue
            # Skip a call its caller already cancelled; once running, the reply ignores cancellation, so only
            # the loop resolves it.
            if not message.reply.set_running_or_notify_cancel():
                continue
            if isinstance(message, ModelStateDictLoadMessage):
                pending_load_messages.append(message)
            else:
                self._request_dispatcher.rank0_register_generation(
                    message.engine_request.request_id,
                    message.metrics_prefix,
                    message.reply,
                )
                pending_engine_requests.append(message.engine_request)

        # A model state dict load takes priority over admitting new requests. Loads taken off the queue
        # together are coalesced into one, at the highest version: every load reads the latest push from
        # one TorchStore key, so the weights read are at least as new as any version requested.
        if pending_load_messages:
            decision = LoopDecision(
                action=LoopAction.LOAD_MODEL_STATE_DICT,
                requests_per_dp_rank=[],
                load_version=max(load.version for load in pending_load_messages),
            )
        else:
            # `LoopAction.STEP`: admit whatever is pending (may be empty -> just keep stepping in-flight work).
            for request in pending_engine_requests:
                if self._reset_kv_cache_on_model_state_dict_load:
                    # Each load resets all KV, so requests need no pin or salt.
                    request.min_policy_version = self.policy_version
                else:
                    request.min_policy_version = (
                        self._group_min_policy_versions.setdefault(
                            request.group_id, self.policy_version
                        )
                    )
            requests_per_dp_rank = self._request_dispatcher.rank0_route(
                pending_engine_requests
            )
            pending_engine_requests.clear()
            decision = LoopDecision(
                action=LoopAction.STEP,
                requests_per_dp_rank=requests_per_dp_rank,
            )
        decision.released_session_ids = released_session_ids
        decision.released_group_ids = released_group_ids
        return decision

    @sl.log_trace_span("load_model_state_dict_copy")
    async def _load_model_state_dict(self, version: int) -> None:
        """ALL RANKS: collectively load the prefetched model state dict and bump the policy
        version.

        The model state dict was already fetched into the engine's pinned CPU buffers; this
        performs the local CPU-to-GPU copy.
        """
        self._engine.load_prefetched_model_state_dict()
        self.policy_version = version
        gc.collect()

    async def _release_resources(self) -> None:
        """Stop the dispatcher, fail the replies it left unresolved, and drop the engine. No-op
        once released."""
        # Stop the result-drain task on rank 0.
        await self._request_dispatcher.shutdown()

        # The loop has stopped; fail any replies it left unresolved so awaiting callers get an
        # exception instead of hanging.
        self._request_dispatcher.fail_outstanding_generations(
            RuntimeError("generator closed before the request finished")
        )

        if self._engine is not None:
            self._engine.shutdown()
            self._engine = None


# ---- Engine-loop queue: rank 0's queued calls; a LoopDecision broadcasts only their EngineRequests. ----


@dataclass(kw_only=True, slots=True)
class GenerationMessage:
    """A queued `generate` call: its `engine_request`, and what only rank 0 needs to answer the caller.
    Never broadcast; a `LoopDecision` carries just the `engine_request`."""

    engine_request: EngineRequest
    metrics_prefix: str
    reply: concurrent.futures.Future[Completion]


@dataclass(kw_only=True, slots=True)
class ModelStateDictLoadMessage:
    """A queued model state dict load: the policy `version` to load, and the `reply` the engine
    loop resolves once the load has been applied."""

    version: int
    reply: concurrent.futures.Future[None]


def _fail_loads(
    pending_load_messages: list[ModelStateDictLoadMessage], exc: BaseException
) -> None:
    """Fail the replies of `pending_load_messages`, which the engine loop took off the queue but never
    applied."""
    for load in pending_load_messages:
        if not load.reply.done():
            load.reply.set_exception(exc)


@dataclass(kw_only=True, slots=True)
class CloseMessage:
    """The shutdown signal (no payload) `EngineLoop.close` closes the queue with; the engine loop
    returns `LoopAction.CLOSE` when it sees one."""


@dataclass(kw_only=True, slots=True)
class SessionReleaseMessage:
    """Sessions (and every session of groups) whose held KV the engine loop releases; no reply."""

    session_ids: list[str] = field(default_factory=list)
    group_ids: list[int] = field(default_factory=list)


EngineLoopMessage = (
    GenerationMessage | ModelStateDictLoadMessage | CloseMessage | SessionReleaseMessage
)


class EngineLoopQueue:
    """Rank 0's queue of `EngineLoopMessage`s for the engine loop.

    `put`, `close` and `closed` may run on any thread. The other methods must run on `event_loop`,
    which owns the queue.

    Args:
        event_loop: The engine loop thread's event loop.
    """

    def __init__(self, event_loop: asyncio.AbstractEventLoop) -> None:
        self._event_loop = event_loop
        self._queue: asyncio.Queue[EngineLoopMessage] = asyncio.Queue()
        self._closed_reason: str | None = None
        # Orders each put against `close`, so that no message can land behind the last one.
        self._lock = threading.Lock()

    @property
    def closed(self) -> bool:
        """Whether `close` has been called."""
        return self._closed_reason is not None

    def put(self, message: EngineLoopMessage) -> None:
        """Enqueue `message`; it lands on the next iteration of `event_loop`. Raises once closed."""
        with self._lock:
            if self._closed_reason is not None:
                raise RuntimeError(self._closed_reason)
            # asyncio.Queue is not thread-safe, so hand the put to the loop that owns it.
            self._event_loop.call_soon_threadsafe(self._queue.put_nowait, message)

    def close(self, last_message: EngineLoopMessage, reason: str) -> None:
        """Queue `last_message` behind every message already put; later puts raise `RuntimeError(reason)`.
        No-op once closed."""
        with self._lock:
            if self._closed_reason is not None:
                return
            self._closed_reason = reason
            self._event_loop.call_soon_threadsafe(self._queue.put_nowait, last_message)

    async def get(self) -> EngineLoopMessage:
        """Wait for the next message."""
        return await self._queue.get()

    def get_nowait(self) -> EngineLoopMessage:
        """Take the next message; raises `asyncio.QueueEmpty` if there is none."""
        return self._queue.get_nowait()

    def empty(self) -> bool:
        """Whether no message is waiting."""
        return self._queue.empty()


class LoopAction(enum.Enum):
    """What the engine loop does each loop iteration (rank 0 decides; the choice is broadcast to all)."""

    STEP = "step"
    # run a burst of engine.step(); first admit any newly-queued requests.

    LOAD_MODEL_STATE_DICT = "load_model_state_dict"
    # load the latest model state dict between bursts

    CLOSE = "close"
    # stop the loop


@dataclass(kw_only=True, slots=True)
class LoopDecision:
    """The per-iteration decision rank 0 broadcasts so every rank acts identically.
    This is broadcasted (pickled) in the engine loop"""

    action: LoopAction

    requests_per_dp_rank: list[list[EngineRequest]] | None = None
    # Per-DP-rank requests to admit before a STEP burst; index == DP rank, fixed
    # length data_parallel_degree. Each rank admits only its own DP-rank slice.
    # (empty unless any queued)

    load_version: int | None = None
    # set iff action is LOAD_MODEL_STATE_DICT

    released_session_ids: list[str] = field(default_factory=list)
    released_group_ids: list[int] = field(default_factory=list)
    # sessions (and groups) whose held KV every rank releases before acting (session_kv_holder)
