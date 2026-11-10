# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Coordinate external-launcher schedulers while KV transfers run asynchronously.

Each TP group shares request ordering and transfer completions. DP replicas handle
independent requests in their own TP groups. Shared Titan request IDs match reports
across ranks; each scheduler applies results using its own vLLM request IDs.

TODO(acisseJZhong): upstream shared request IDs and TP KV completion aggregation
before Scheduler.update_from_output so external-launcher engines can use the
native scheduler. Until then this replaces the private `waiting`/`skipped_waiting`
queues and overrides `_select_waiting_queue_for_scheduling`, `_free_blocks` and
`_update_from_kv_xfer_finished`.
"""

import heapq
from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace
from typing import Any

import torch.distributed as dist

from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus


class _SharedIDRequestQueue(RequestQueue):
    """Priority heap that breaks ties by shared Titan request ID, not local ID.

    Local IDs have per-rank random suffixes, so ranks could pop different requests:
        Titan ID    Rank 0 local ID    Rank 1 local ID
        r           r-bbbbbbbb         r-88888888
        r-a         r-a-aaaaaaaa       r-a-99999999
    With equal priority and arrival_time, local IDs pick r-a on rank 0 but r on
    rank 1; shared IDs pick r on both.

    Entries are (priority, arrival_time, shared ID, request); shared IDs are
    unique, so comparisons never reach the request.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[int, float, str, Request]] = []

    def _entry(self, request: Request) -> tuple[int, float, str, Request]:
        return (
            request.priority,
            request.arrival_time,
            request.kv_transfer_params["shared_request_id"],
            request,
        )

    def add_request(self, request: Request) -> None:
        heapq.heappush(self._heap, self._entry(request))

    def pop_request(self) -> Request:
        return heapq.heappop(self._heap)[-1]

    def peek_request(self) -> Request:
        return self._heap[0][-1]

    def prepend_request(self, request: Request) -> None:
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        self._heap.remove(self._entry(request))
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        ids = {request.request_id for request in requests}
        self._heap = [entry for entry in self._heap if entry[-1].request_id not in ids]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __len__(self) -> int:
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        return (entry[-1] for entry in sorted(self._heap))


class KVOffloadScheduler(Scheduler):
    """Keep a TP group's schedulers in lockstep while KV transfers run.

    Used when KV offload is configured and TP * DP > 1 (AsyncKVOffloadScheduler
    under async scheduling); single-rank generators keep vLLM's scheduler.

    With external_launcher each TP rank runs its own scheduler, and every TP
    forward needs them to schedule the same tokens, so their caches must stay
    identical. Free order matters: freed blocks keep their prefix and are evicted
    oldest first, so freeing blocks 7, 9 on rank 0 but 9, 7 on rank 1 evicts
    different prefixes and later gives the ranks different cache hits.

    Decisions that could differ per rank therefore use the shared Titan request
    ID instead of local IDs (per-rank random suffixes) or set iteration order
    (per-process hashing): queue tie-breaks, TP-aggregated KV completions, and
    the order requests free blocks. Each DP replica coordinates its own TP group.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._kv_output_aggregator = KVOutputAggregator.from_connector(
            self.connector, get_tp_group().world_size
        )
        self.shared_to_local_request_ids: dict[str, str] = {}
        if self.policy == SchedulingPolicy.PRIORITY:
            # Requests waiting to be scheduled.
            self.waiting = _SharedIDRequestQueue()
            # Requests deferred by async dependencies (e.g. KV loads) or constraints.
            self.skipped_waiting = _SharedIDRequestQueue()

    def add_request(self, request: Request) -> None:
        """Override vLLM admission to track shared IDs before calling the base method.

        The base scheduler still chooses waiting or skipped_waiting.
        """
        shared_request_id = request.kv_transfer_params["shared_request_id"]
        assert shared_request_id not in self.shared_to_local_request_ids
        self.shared_to_local_request_ids[shared_request_id] = request.request_id
        super().add_request(request)

    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        """Override vLLM priority queue selection with a shared-ID tie-break."""
        if (
            self.policy == SchedulingPolicy.PRIORITY
            and self.waiting
            and self.skipped_waiting
        ):

            def key(queue: RequestQueue) -> tuple[int, float, str]:
                request = queue.peek_request()
                return (
                    request.priority,
                    request.arrival_time,
                    request.kv_transfer_params["shared_request_id"],
                )

            return min((self.waiting, self.skipped_waiting), key=key)
        return super()._select_waiting_queue_for_scheduling()

    def _free_blocks(self, request: Request) -> None:
        """Override vLLM block cleanup to also remove the shared-to-local ID mapping."""
        super()._free_blocks(request)
        # Keep IDs while a cancelled receive still owns the request. Offload saves
        # own separate integer job IDs and block references, not request IDs.
        del self.shared_to_local_request_ids[
            request.kv_transfer_params["shared_request_id"]
        ]

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        """Override vLLM output handling to apply KV completions TP-wide.

        Each worker reports only its own transfers, possibly on different steps.
        All-gather the reports (by shared ID) over the TP group's CPU group and
        accumulate them, so a transfer completes on every rank on the same step,
        once all TP workers have reported it.
        """
        group = get_tp_group()
        if group.world_size > 1:
            outputs: list[KVConnectorOutput | None] = [None] * group.world_size
            dist.all_gather_object(
                outputs,
                _remap_request_ids(
                    model_runner_output.kv_connector_output,
                    lambda local_id: self.requests[local_id].kv_transfer_params[
                        "shared_request_id"
                    ],
                ),
                group=group.cpu_group,
            )
            # The upstream aggregator takes ModelRunnerOutputs. Each shallow
            # copy keeps this rank's model output and carries one worker's KV
            # report; only those reports are aggregated, across ranks and steps.
            aggregated_output = self._kv_output_aggregator.aggregate(
                [replace(model_runner_output, kv_connector_output=o) for o in outputs]
            )
            assert aggregated_output is not None
            aggregated_output.kv_connector_output = _remap_request_ids(
                aggregated_output.kv_connector_output,
                lambda shared_request_id: self.shared_to_local_request_ids[
                    shared_request_id
                ],
            )
            model_runner_output = aggregated_output
        return super().update_from_output(scheduler_output, model_runner_output)

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[Request]:
        """Override vLLM request termination to free blocks in shared-ID order.

        Upstream iterates a set (per-process order); pass one ID per call, since
        it turns iterables back into a set.
        """
        if isinstance(request_ids, str):
            return super().finish_requests(request_ids, finished_status)
        ids = self.requests if request_ids is None else request_ids
        finished = []
        for request_id in sorted(
            set(ids) & self.requests.keys(),
            key=lambda local_id: self.requests[local_id].kv_transfer_params[
                "shared_request_id"
            ],
        ):
            finished.extend(super().finish_requests(request_id, finished_status))
        return finished

    def _update_from_kv_xfer_finished(self, output: KVConnectorOutput) -> None:
        """Override vLLM KV completion handling to free blocks in shared-ID order.

        Upstream iterates the finished_recving/finished_sending sets (per-process
        order) and may free GPU blocks or CPU cache references per request. Apply
        the non-request fields once, then pass each request to upstream
        separately, in shared-ID order. IDs here are already local (see
        update_from_output).
        """

        def shared_order(request_ids: set[str] | None) -> list[str]:
            return sorted(
                request_ids or (),
                key=lambda local_id: self.requests[local_id].kv_transfer_params[
                    "shared_request_id"
                ],
            )

        # Save-job completions and other non-request fields are applied once.
        super()._update_from_kv_xfer_finished(
            replace(output, finished_recving=None, finished_sending=None)
        )
        for request_id in shared_order(output.finished_recving):
            super()._update_from_kv_xfer_finished(
                KVConnectorOutput(finished_recving={request_id})
            )
        for request_id in shared_order(output.finished_sending):
            super()._update_from_kv_xfer_finished(
                KVConnectorOutput(finished_sending={request_id})
            )


def _remap_request_ids(
    output: KVConnectorOutput | None, get_request_id: Callable[[str], str]
) -> KVConnectorOutput | None:
    """Return a KV report with request IDs converted by get_request_id.

    Before TP aggregation: local vLLM IDs -> shared request IDs.
    After TP aggregation: shared request IDs -> this rank's local vLLM IDs.
    Remaps finished_recving, finished_sending, and failed_recving in a copy;
    save-job IDs and other fields are unchanged.
    """
    if output is None:
        return None
    return replace(
        output,
        finished_recving=(
            {get_request_id(req_id) for req_id in output.finished_recving}
            if output.finished_recving is not None
            else None
        ),
        finished_sending=(
            {get_request_id(req_id) for req_id in output.finished_sending}
            if output.finished_sending is not None
            else None
        ),
        failed_recving={get_request_id(req_id) for req_id in output.failed_recving},
    )


class AsyncKVOffloadScheduler(AsyncScheduler, KVOffloadScheduler):
    """Preserve vLLM's async scheduling while coordinating KV completions."""


def create_kv_offload_scheduler(
    vllm_config: VllmConfig,
    *args: Any,
    **kwargs: Any,
) -> Scheduler:
    # vLLM resolves automatic async scheduling while building VllmConfig, after
    # Titan creates EngineArgs. Select the matching implementation here.
    scheduler_cls = (
        AsyncKVOffloadScheduler
        if vllm_config.scheduler_config.async_scheduling
        else KVOffloadScheduler
    )
    return scheduler_cls(vllm_config, *args, **kwargs)
