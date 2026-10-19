# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The generator's request dispatch across data-parallel replicas and completion fan-in."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import cloudpickle
import torch
import torch.distributed as dist

from torchtitan.rl.generator.types import (
    EngineRequest,
    FinishedRequest,
    RequestMetricsInputs,
)
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.types import Completion
from torchtitan.rl.routing.intra_generator import IntraGeneratorRouter

logger = logging.getLogger(__name__)


class RequestDispatcher:
    """Handles the generator's DP/TP request dispatch.

    Every rank holds one dispatcher; methods act according to the rank's role:
    - Rank 0 is the coordinator (and DP0's tp_rank=0): it holds the outstanding
      generations, routes requests, opens the fan-in port, and resolves every
      completion -- its own replica's locally, peers' via the drain task. State
      and methods only ever used on rank 0 are prefixed ``rank0_``.
    - Other DP's tp_rank=0 send their finished requests to rank 0 over the port.
    - tp_rank!=0 hold no outputs; their dispatcher only carries the layout.

    Supported rank layout:

        global_rank = dp_rank * tp_degree + tp_rank

    EP reuses the same global-rank -> (dp, tp) mapping, so it needs no special
    handling here.

    Data flow:

    Take DP=2, TP=2 for example. Only rank 0 holds the replies and talks to the
    controller, so requests finished by any other DP replica must be sent
    back ("fanned in") to rank 0:

        controller --generate--> rank 0  (registers the generation)
                                   |
            rank0_route(): pick a DP rank for the queued requests
                                   |   (broadcast in the LoopDecision, elsewhere)
              +--------------------+--------------------+
              v                                         v
        DP0's tp_rank=0 (i.e. rank 0)               DP1's tp_rank=0 (i.e. rank 2)
          engine.step()                             engine.step()
          resolve own replies locally   <--port--   send finished requests to rank 0
                                                    (tp_rank!=0: hold no outputs)
        rank 0 background drain task: recv from port -> resolve those replies
    """

    def __init__(
        self,
        *,
        rank: int,
        dp_rank: int,
        tp_rank: int,
        dp_degree: int,
        broadcast_group: dist.ProcessGroup,
        intra_generator_router: IntraGeneratorRouter.Config,
        open_result_channel: Callable[[], tuple[Any, Any]] | None,
    ):
        self._rank = rank
        self._dp_degree = dp_degree
        self.dp_rank = dp_rank
        self._tp_rank = tp_rank
        # Reused for the one-time result-port broadcast (see ``setup``).
        self._broadcast_group = broadcast_group

        # RANK 0: generations taken off the queue and not yet answered, keyed by
        # request id. An entry is removed once its reply is resolved, with the
        # completion or an error. Only rank 0 ever populates this.
        self._rank0_outstanding_generations: dict[str, OutstandingGeneration] = {}

        # --- Result fan-in (only when DP>1) ---
        # rank 0 opens the channel, keeps the receiving end, and its drain task
        # resolves whatever peer tp_rank=0 send. ``_result_port`` is the sending
        # end, held only by peer tp_rank=0; it stays None on rank 0 (resolves
        # locally) and on tp_rank!=0 (no outputs). All None for DP=1 since there
        # is no peer DP to send from.
        self._open_result_channel = open_result_channel
        self._result_port: Any | None = None
        self._rank0_result_receiver: Any | None = None
        self._rank0_drain_task: asyncio.Task | None = None

        # --- DP routing ---
        # RANK-0 DP routing: pick a DP rank per request, reserving its load until
        # the completion resolves.
        self._rank0_dp_router: IntraGeneratorRouter | None = (
            intra_generator_router.build(dp_degree=self._dp_degree)
            if self._rank == 0 and self._dp_degree > 1
            else None
        )
        # `_rank0_dp_router.num_repins` at the last `rank0_repin_metrics`.
        self._num_repins_logged = 0

    def rank0_repin_metrics(self) -> list[m.Metric]:
        """RANK 0: sessions the DP router moved to another DP rank since the last call (each move
        recomputes the session's prompt there); [] without a DP router."""
        if self._rank0_dp_router is None:
            return []
        num_repins = self._rank0_dp_router.num_repins
        repinned = num_repins - self._num_repins_logged
        self._num_repins_logged = num_repins
        return [m.Metric("generator/session_kv/dp_repinned_sessions", m.Sum(repinned))]

    def rank0_register_generation(
        self,
        request_id: str,
        metrics_prefix: str,
        reply: concurrent.futures.Future[Completion],
    ) -> None:
        """RANK 0: register a generation taken off the queue, whose ``reply`` is resolved with
        ``request_id``'s completion."""
        if request_id in self._rank0_outstanding_generations:
            raise AssertionError(f"request_id {request_id!r} is already in flight")
        self._rank0_outstanding_generations[request_id] = OutstandingGeneration(
            reply=reply, metrics_prefix=metrics_prefix
        )

    def rank0_num_outstanding_generations(self) -> int:
        """RANK 0: how many generations are outstanding, including ones taken off the queue but not
        admitted yet."""
        return len(self._rank0_outstanding_generations)

    def rank0_has_outstanding_generations(self) -> bool:
        """RANK 0: whether any generation is outstanding (reply unresolved).

        A generation stays registered until its completion comes back, so this stays
        True while any peer DP rank is still running it.
        """
        return bool(self._rank0_outstanding_generations)

    def rank0_route(self, requests: list[EngineRequest]) -> list[list[EngineRequest]]:
        """RANK 0: pick which DP rank serves each queued request.

        Returns a fixed-length (``dp_degree``) list; index == DP rank. Each rank
        later admits only its own slice. With a single DP rank, everything goes
        to DP rank 0; otherwise ``IntraGeneratorRouter`` reserves a DP rank per
        request and that reservation is released when the request's completion
        resolves.
        """
        requests_per_dp_rank: list[list[EngineRequest]] = [
            [] for _ in range(self._dp_degree)
        ]
        for request in requests:
            if self._rank0_dp_router is None:
                dp_rank = 0
            else:
                # Pick a DP rank for this request, and increment this DP rank's
                # load by 1 (i.e. measured by request count).
                dp_rank = self._rank0_dp_router.reserve(
                    request.request_id,
                    routing_session_id=request.routing_session_id,
                )
            requests_per_dp_rank[dp_rank].append(request)

        return requests_per_dp_rank

    def rank0_stamp_min_policy_version(
        self,
        requests_per_dp_rank: list[list[EngineRequest]],
    ) -> None:
        """RANK 0: stamp each request's min policy version on its outstanding generation, for every
        request in this `LoopAction.STEP` decision, across all DP ranks. Without a KV reset the request
        may reuse KV cached under that version, so it is the oldest policy the completion can depend
        on. Rank 0 holds the outstanding generations regardless of which DP rank serves the request,
        so it stamps them all here."""
        for dp_requests in requests_per_dp_rank:
            for request in dp_requests:
                self._rank0_outstanding_generations[
                    request.request_id
                ].min_policy_version = request.min_policy_version

    def setup(self) -> None:
        """One-time setup before the engine loop starts (DP>1): distribute rank 0's
        result-fan-in port and start rank 0's drain task.

        All ranks call this so the broadcast over the all-ranks ``_broadcast_group``
        completes; only TP rank 0 keep the port.
        """
        if self._dp_degree == 1:
            return

        # rank 0 opens the channel and keeps the receiving end; it broadcasts only
        # the sending port to the peers.
        if self._rank == 0:
            if self._open_result_channel is None:
                raise RuntimeError(
                    "data-parallel generation requires a result-channel factory"
                )
            port, self._rank0_result_receiver = self._open_result_channel()
            # Monarch Port objects need cloudpickle, so we cloudpickle it into bytes
            # first. Otherwise, broadcast_object_list will attempt to pickle the
            # port object with stdlib pickle and result in error.
            container = [cloudpickle.dumps(port)]
        else:
            container = [None]
        dist.broadcast_object_list(
            container, src=0, group=self._broadcast_group, device=torch.device("cpu")
        )
        assert container[0] is not None
        # Only peer TP rank 0s send finished requests to global rank 0, so only
        # they keep the port. Global rank 0 resolves locally and tp_rank!=0
        # produce no outputs.
        if self._rank != 0 and self._tp_rank == 0:
            self._result_port = cloudpickle.loads(container[0])
        if self._rank == 0:
            self._rank0_drain_task = asyncio.create_task(self._rank0_drain_results())

    def process_finished_requests(
        self, finished_requests: list[FinishedRequest], policy_version: int
    ) -> None:
        """TP rank 0s send finished requests to global rank 0 after each engine step:
          - Global Rank 0 resolves its own DP replica's requests locally;
          - every other TP rank 0 sends them over the port to global rank 0's drain
            task.
          - Other ranks hold no finished outputs and do nothing.

        ``policy_version`` is the version installed when they finished.
        """
        if self._tp_rank != 0:
            return

        if self._rank == 0:
            self._rank0_resolve_generations(finished_requests, policy_version)
        elif finished_requests:
            self._result_port.send(
                _FinishedRequests(
                    requests=finished_requests, policy_version=policy_version
                )
            )

    def process_rejected_requests(
        self, rejected_requests: list[tuple[str, Exception]]
    ) -> None:
        """TP rank 0s report the requests their DP replica's engine rejected at admission:
        - global rank 0 fails its own replica's replies locally;
        - every other TP rank 0 sends them over the port to global rank 0's drain task;
        - other ranks rejected the same requests and do nothing.
        """
        if self._tp_rank != 0:
            return

        for request_id, exc in rejected_requests:
            logger.warning("Engine rejected request %s: %s", request_id, exc)
        if self._rank == 0:
            self._rank0_fail_generations(rejected_requests)
        else:
            self._result_port.send(_RejectedRequests(requests=rejected_requests))

    def process_aborted_requests(self, request_ids: list[str]) -> None:
        """Like `process_rejected_requests`, for the requests this replica's engine aborted
        because their caller left, without a warning per request (aborts come in bursts, e.g. a
        group timing out)."""
        if self._tp_rank != 0 or not request_ids:
            return

        aborted_requests: list[tuple[str, Exception]] = [
            (request_id, RuntimeError(f"{request_id} aborted: its caller left"))
            for request_id in request_ids
        ]
        if self._rank == 0:
            self._rank0_fail_generations(aborted_requests)
        else:
            self._result_port.send(_RejectedRequests(requests=aborted_requests))

    def rank0_fail_pending_generations(self, request_ids: list[str]) -> None:
        """RANK 0: fail generations taken off the queue but not admitted yet, whose callers left.
        They hold no DP reservation: `rank0_route` reserves only at admission."""
        for request_id in request_ids:
            self._rank0_outstanding_generations.pop(request_id).reply.set_exception(
                RuntimeError(f"{request_id} aborted: its caller left")
            )

    def _rank0_resolve_generations(
        self, finished_requests: list[FinishedRequest], policy_version: int
    ) -> None:
        """RANK 0: build each finished request's ``Completion``, including its metrics (rank 0
        is the only place that knows the request's ``metrics_prefix`` and admitted version), then
        resolve its reply."""
        for finished in finished_requests:
            # in flight when this one finished (includes itself; counted before the pop)
            inflight_requests_at_completion = float(
                len(self._rank0_outstanding_generations)
            )
            generation = self._rank0_outstanding_generations.pop(finished.request_id)
            metrics_prefix = generation.metrics_prefix

            metrics = _prepare_generation_request_metrics(
                finished.metrics_inputs, prefix=metrics_prefix
            )
            for metric_type in [m.Max, m.Mean]:
                metrics.append(
                    m.Metric(
                        f"{metrics_prefix}/inflight_requests_at_completion",
                        metric_type(inflight_requests_at_completion),
                    )
                )

            generation.reply.set_result(
                Completion(
                    # The version stamped at admission, and the version installed at finish.
                    min_policy_version=generation.min_policy_version,
                    max_policy_version=policy_version,
                    request_id=finished.request_id,
                    token_ids=finished.token_ids,
                    token_logprobs=finished.token_logprobs,
                    topk_token_ids=finished.topk_token_ids,
                    topk_logprobs=finished.topk_logprobs,
                    finish_reason=finished.finish_reason,
                    routed_expert_ids=finished.routed_expert_ids,
                    metrics=metrics,
                )
            )
            # Free the request's reserved load on its DP rank so load-aware
            # routing sees the accurate loads on DPs.
            if self._rank0_dp_router is not None:
                self._rank0_dp_router.release(finished.request_id)

    def _rank0_fail_generations(
        self, rejected_requests: list[tuple[str, Exception]]
    ) -> None:
        """RANK 0: fail the reply of each rejected request with its error."""
        for request_id, exc in rejected_requests:
            self._rank0_outstanding_generations.pop(request_id).reply.set_exception(exc)
            if self._rank0_dp_router is not None:
                self._rank0_dp_router.release(request_id)

    async def _rank0_drain_results(self) -> None:
        """RANK 0 background task which receives and resolves finished (or rejected, or aborted)
        requests pushed by peer TP rank 0s.
        """
        # TODO: wait_engine_loop does not watch this task. If it raises (e.g. an unknown request
        # id), later peer replies are never resolved and their generate() calls hang; make its
        # failure fail the engine loop.
        while True:
            results = await self._rank0_result_receiver.recv()
            if isinstance(results, _RejectedRequests):
                self._rank0_fail_generations(results.requests)
            else:
                self._rank0_resolve_generations(
                    results.requests, results.policy_version
                )

    def fail_outstanding_generations(self, exc: BaseException) -> None:
        """RANK 0: fail the reply of every outstanding generation after an exception or
        teardown (no-op elsewhere, where the map is empty)."""
        for generation in self._rank0_outstanding_generations.values():
            if not generation.reply.done():
                generation.reply.set_exception(exc)
        self._rank0_outstanding_generations.clear()

    async def shutdown(self) -> None:
        """Stop rank 0's drain task, if any (no-op elsewhere)."""
        if self._rank0_drain_task is not None:
            self._rank0_drain_task.cancel()
            try:
                await self._rank0_drain_task
            except (asyncio.CancelledError, Exception):
                pass
            self._rank0_drain_task = None


@dataclass(kw_only=True, slots=True)
class OutstandingGeneration:
    """A generation rank 0 took off the queue and has not answered yet: the `reply` it resolves with
    the `Completion`, and what it needs to build that `Completion`."""

    reply: concurrent.futures.Future[Completion]
    metrics_prefix: str
    """Namespaces this generation's metrics (e.g. `generator` vs `validation_generator`)."""
    min_policy_version: int = field(init=False)
    """Policy version the request was admitted (sampled) under; the max is read at finish (see `Completion`)."""


@dataclass(frozen=True, slots=True)
class _FinishedRequests:
    """Requests a peer DP replica's engine finished, sent to rank 0 with the policy version
    installed when they finished."""

    requests: list[FinishedRequest]
    policy_version: int


@dataclass(frozen=True, slots=True)
class _RejectedRequests:
    """Requests a peer DP replica's engine rejected at admission or aborted, sent to rank 0 with
    each error."""

    requests: list[tuple[str, Exception]]


def _prepare_generation_request_metrics(
    inputs: RequestMetricsInputs, *, prefix: str
) -> list[m.Metric]:
    """Prepare per-request generation metrics from the raw inputs.

    Caveat for vLLM under `SamplingParams.n > 1`: vLLM stores one `RequestStateStats`
    per child request; the parent output exposes the **last-finishing**
    child's timeline. `arrival_time` is shared across siblings, but
    [`queued_ts`, `scheduled_ts`, `first_token_ts`, `last_token_ts`,
    `num_generation_tokens`, `num_preemptions`] describe one specific child -
    not an aggregate, not the first sibling's. The other `n-1` siblings' stats
    are dropped by vLLM at ``output_processor._finish_request``.
    """

    # TODO: Per-request fields here come from vLLM's RequestOutput.metrics
    # (RequestStateStats). Engine-aggregate stats, such as KV-cache usage,
    # prefix-cache hit rate, exact preemption counts, and batch occupancy, live
    # in SchedulerStats / IterationStats and require registering a
    # vllm.v1.metrics.loggers.StatLoggerBase via
    # LLMEngine.from_engine_args(..., stat_loggers=[...]); VllmOtelStatLogger
    # does this when VLLMEngine.Config.stat_logger is set.

    metric_values: dict[str, float] = {}
    if inputs.num_cached_tokens is not None:
        metric_values[f"{prefix}/num_cached_tokens"] = inputs.num_cached_tokens

    if inputs.has_stats:
        metric_values[f"{prefix}/queue_time_ms"] = (
            inputs.scheduled_ts - inputs.queued_ts
        ) * 1000
        metric_values[f"{prefix}/num_preemptions"] = inputs.num_preemptions

        if inputs.num_generation_tokens > 0:
            metric_values[f"{prefix}/time_to_first_token_ms"] = (
                inputs.first_token_latency * 1000
            )
            metric_values[f"{prefix}/prefill_time_ms"] = (
                inputs.first_token_ts - inputs.scheduled_ts
            ) * 1000

        if inputs.num_generation_tokens > 1:
            first_to_last_token_ms = (
                inputs.last_token_ts - inputs.first_token_ts
            ) * 1000
            metric_values[f"{prefix}/decode_time_ms"] = first_to_last_token_ms
            metric_values[
                f"{prefix}/inter_token_latency_ms"
            ] = first_to_last_token_ms / (inputs.num_generation_tokens - 1)

    # Emit each value with both Mean and Max aggregators.
    return [
        metric
        for key, value in metric_values.items()
        for metric in (m.Metric(key, m.Mean(value)), m.Metric(key, m.Max(value)))
    ]
