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
native scheduler. This currently depends on vLLM's private scheduler methods.
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
    """A priority heap with a shared request-ID tie-break instead of local IDs.

    Example with equal priority and arrival_time:
        Titan ID    Rank 0 local ID    Rank 1 local ID
        r           r-bbbbbbbb         r-88888888
        r-a         r-a-aaaaaaaa       r-a-99999999

    Local-ID sorting picks r-a on rank 0 ("a" < "b") but r on rank 1
    ("8" < "a"). Sorting by Titan ID makes both ranks pick r, then r-a.
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
    """Coordinate CPU KV offload across Titan's external-launcher schedulers.

    Titan enables this adapter when KV offload is configured and TP * DP > 1,
    using AsyncKVOffloadScheduler for async scheduling. Single-rank generators
    keep vLLM's native scheduler.

    Each rank runs its own scheduler. Within each TP group, this adapter uses
    shared request IDs for consistent ordering, aggregates KV completions before
    updating request state, and releases GPU blocks in the same order. Each DP
    replica coordinates its own TP group. Native batching and token-budget
    scheduling are reused.
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
        """Override vLLM output handling to aggregate TP KV reports first.

        Then call the base implementation, which dispatches completion handling
        to this class's _update_from_kv_xfer_finished().
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
        """Override vLLM request termination to release blocks in shared-ID order."""
        # Free in the same order on every TP rank so future physical block IDs
        # agree. Upstream converts iterable IDs back to a set; pass strings.
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
        """Override vLLM KV completion handling with shared-ID ordering.

        update_from_output() has already aggregated TP reports and restored local
        request IDs. Completed loads let waiting requests resume on a later step;
        requests that ended during loading can release their GPU blocks. The
        connector handles save-job completion and its held block references.

        Unlike upstream's unordered traversal, process receive callbacks and
        request cleanup in shared-ID order so CPU cache LRU updates and GPU block
        reuse stay consistent across TP ranks.
        """
        # The connector tracks save jobs by job ID and releases their pinned blocks once
        # all required workers finish, independently of request cleanup below.
        if self.connector is not None:
            self.connector.update_connector_output(
                replace(output, finished_recving=None)
            )
        for request_id in sorted(
            output.finished_recving or (),
            key=lambda local_id: self.requests[local_id].kv_transfer_params[
                "shared_request_id"
            ],
        ):
            # CPU offload also frees cache references in this callback. Apply
            # receives in shared request-ID order so every rank keeps the same LRU order.
            if self.connector is not None:
                self.connector.update_connector_output(
                    KVConnectorOutput(finished_recving={request_id})
                )
            request = self.requests[request_id]
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                # schedule() checks this flag before resuming the request.
                self.finished_recving_kv_req_ids.add(request_id)
            else:
                # The request ended during the load; its blocks can now be freed.
                assert RequestStatus.is_finished(request.status)
                self._free_blocks(request)

        # Offload saves use job metadata above; retain generic send-completion handling.
        for request_id in sorted(
            output.finished_sending or (),
            key=lambda local_id: self.requests[local_id].kv_transfer_params[
                "shared_request_id"
            ],
        ):
            self._free_blocks(self.requests[request_id])


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
