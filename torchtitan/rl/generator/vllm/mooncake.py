# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Coordinate Mooncake cache decisions between external-launcher TP schedulers.

Schedulers in a TP group agree on request order, cache hits, transfer completion,
and when to release GPU blocks. Each DP replica has its own requests and TP
group. Each rank runs a scheduler and worker in the same process. Upstream
Mooncake workers transfer each rank's KV data.

Each request has one shared Titan ID, but vLLM may assign a different internal
ID on each rank. Map completion reports to Titan IDs before aggregation, then
map results back to local vLLM IDs before updating each scheduler.

Cache-hit flow within one TP group (TP=2; ranks below are TP-local)::

    1. LOOKUP: share cache metadata
        Scheduler 0 <-> Worker 0's lookup service <-> Mooncake Store
                       (checks all required TP shards)
        Scheduler 0 -- broadcast result + LoadSpec --> Scheduler 1
        Both schedulers use the same hit/miss/pending result.

    2. LOAD: reserve local GPU blocks, then load each worker's KV shard
        Rank 0 / GPU 0                      Rank 1 / GPU 1
        Scheduler 0                         Scheduler 1
            | reserve GPU blocks                | reserve GPU blocks
            v                                   v
        Build load metadata                 Build load metadata
        (token ranges + local block IDs)    (token ranges + local block IDs)
            |                                   |
            v                                   v
        Worker 0: Store -> GPU 0             Worker 1: Store -> GPU 1
                  KV shard 0                          KV shard 1

    3. COMPLETE: agree on transfer completion before scheduler updates
        Worker 0's report                   Worker 1's report
            | map to Titan request IDs          | map to Titan request IDs
            +-----------------+-----------------+
                              v
        all_gather: BOTH schedulers receive BOTH workers' reports
                              |
            +-----------------+-----------------+
            v                                   v
        Scheduler 0 aggregates              Scheduler 1 aggregates
        reports across steps                reports across steps
            | map to local vLLM IDs             | map to local vLLM IDs
            v                                   v
        Update Scheduler 0                  Update Scheduler 1
                  (same logical completion/error state)

LoadSpec describes cached token ranges and tail boundaries. Each scheduler adds
its local destination block IDs to load metadata after allocation. Only hits
that need loading enter phase 2; pending lookups retry on later engine steps.

Each all_gather waits for every TP rank to report; the report can be empty while
transfers are in flight. Receive completions accumulate across steps until all
required workers finish. The Store scheduler separately counts completed_saves
per save job before releasing its pinned blocks. KV tensors travel between the
Store and each worker; collectives exchange metadata only.

TODO(acisseJZhong): upstream external-launcher KV coordination so these adapters
can be removed: https://github.com/vllm-project/vllm/issues/11400
"""

from typing import Any

import torch.distributed as dist
from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import (
    MooncakeStoreConnector as VllmMooncakeStoreConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
    LookupKeyServer,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request


class MooncakeStoreConnector(VllmMooncakeStoreConnector):
    """Adapt distributed Store lookup and save completion to Titan's TP groups.

    Share the leader's lookup result and scope save completion to this TP group.
    KVOffloadScheduler handles
    request ordering and completion aggregation for both CPU offload connectors.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        group = get_tp_group()
        if self.connector_scheduler is not None:
            # external_launcher world_size includes DP, whose requests and save
            # jobs are independent. A save waits only for this TP group's shards.
            self.connector_scheduler._num_workers = group.world_size
        worker = self.connector_worker
        if (
            worker is not None
            and group.rank_in_group == 0
            and worker.lookup_server is None
        ):
            # Upstream starts this only on global rank 0; each DP replica needs
            # its own server at the connector's DP-local IPC address.
            worker.lookup_server = LookupKeyServer(worker, vllm_config)

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        assert self.connector_scheduler is not None
        group = get_tp_group()
        if group.world_size == 1:
            return super().get_num_new_matched_tokens(request, num_computed_tokens)

        # The lookup service checks every required TP shard. Publish its whole
        # result, including pending status and partial-hit load metadata, once.
        payload: list[Any] = [None]
        if group.rank_in_group == 0:
            try:
                result = super().get_num_new_matched_tokens(
                    request, num_computed_tokens
                )
                payload[0] = (
                    result,
                    self.connector_scheduler.load_specs.get(request.request_id),
                    None,
                )
            except Exception as exc:
                payload[0] = (None, None, str(exc))
        dist.broadcast_object_list(payload, src=group.ranks[0], group=group.cpu_group)
        result, load_spec, error = payload[0]
        if error is not None:
            raise RuntimeError(f"Mooncake lookup failed on TP rank 0: {error}")
        if load_spec is not None:
            self.connector_scheduler.load_specs[request.request_id] = load_spec
        return result
