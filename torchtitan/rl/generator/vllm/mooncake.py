# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Adapt vLLM's MooncakeStoreConnector to external-launcher TP groups.

Each rank runs its own scheduler; KVOffloadScheduler (``cpu_kv_offload.py``)
keeps their request order and transfer completions in sync. This adapter adds
the Mooncake-specific pieces:

- Lookup: only the TP leader queries the Store (its lookup checks every TP
  shard) and broadcasts the result and LoadSpec, so all ranks allocate and load
  the same blocks.
- Save completion waits for this TP group's workers, not every external-launcher
  rank (TP * DP).
- Each DP replica's TP leader starts the lookup server at its DP-local address.

TODO(acisseJZhong): upstream to vLLM: with external_launcher, start a lookup
server per DP replica, scope the save quorum to the TP group, and share the TP
leader's lookup result, so this adapter can be removed.
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
    """MooncakeStoreConnector with TP-leader lookups and TP-group save completion."""

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

        # Publish the leader's whole result, including pending status and the
        # partial-hit LoadSpec. Upstream reports lookup failures as a miss.
        payload: list[Any] = [None]
        if group.rank_in_group == 0:
            result = super().get_num_new_matched_tokens(request, num_computed_tokens)
            payload[0] = (
                result,
                self.connector_scheduler.load_specs.get(request.request_id),
            )
        dist.broadcast_object_list(payload, src=group.ranks[0], group=group.cpu_group)
        result, load_spec = payload[0]
        # Mirror the leader exactly, including a spec it dropped after an earlier hit.
        if load_spec is None:
            self.connector_scheduler.load_specs.pop(request.request_id, None)
        else:
            self.connector_scheduler.load_specs[request.request_id] = load_spec
        return result
