# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Adapt local CPU offload to Titan's TP groups and session release hints."""

from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector as VllmSimpleCPUOffloadConnector,
)
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    get_block_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_cache_interface import KVCacheConfig


class SimpleCPUOffloadConnector(VllmSimpleCPUOffloadConnector):
    """Local CPU caches with a save-completion quorum scoped to one TP group.

    Every scheduler queries its own CPU cache. Coordinated store completions and
    block release keep those caches consistent; no leader lookup is needed.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        if self.scheduler_manager is not None:
            # external_launcher world_size includes independent DP replicas.
            # TODO(acisseJZhong): upstream a TP-scoped save quorum instead of
            # setting the private _expected_worker_count.
            self.scheduler_manager._expected_worker_count = get_tp_group().world_size

    def get_pending_save_ids(self) -> set[int]:
        """Snapshot queued/in-flight saves, including the next finish-time batch."""
        manager = self.scheduler_manager
        assert manager is not None
        # TODO(acisseJZhong): reads the manager's private save state. Upstream
        # session eviction hints that apply after pending saves would replace it:
        # https://github.com/vllm-project/vllm/issues/51428
        pending = set(manager._store_event_to_blocks) | set(
            manager._abandoned_store_event_to_blocks
        )
        if manager._pending_finished_stores:
            # build_connector_meta folds these tails into the next store event.
            pending.add(manager._store_event_counter)
        return pending

    def release_session_cache(
        self, block_hashes: set[BlockHash], active_hashes: set[BlockHash]
    ) -> None:
        """Make unused CPU blocks the next evicted, preserving their cached data."""
        manager = self.scheduler_manager
        assert manager is not None
        pool = manager.cpu_block_pool
        blocks = {}
        for block_hash in block_hashes:
            for group_id in range(len(manager.group_block_sizes)):
                cached = pool.cached_block_hash_to_block._cache.get(
                    make_block_hash_with_group_id(block_hash, group_id)
                )
                if cached is None:
                    continue
                # Simultaneous saves may cache several physical copies of a hash.
                for block in cached.values() if isinstance(cached, dict) else [cached]:
                    aliases = [
                        block.block_hash,
                        *pool.cached_block_hashes_by_block.get(block.block_id, ()),
                    ]
                    if block.ref_cnt == 0 and not any(
                        get_block_hash(alias) in active_hashes for alias in aliases
                    ):
                        blocks[block.block_id] = block
        # TODO(acisseJZhong): replace direct cache-index and free-queue access
        # with upstream session eviction hints that protect active aliases:
        # https://github.com/vllm-project/vllm/issues/51428
        # Block IDs agree across TP ranks; hashes may use per-process seeds.
        ordered_blocks = [blocks[block_id] for block_id in sorted(blocks)]
        for block in ordered_blocks:
            pool.free_block_queue.remove(block)
        pool.free_block_queue.prepend_n(ordered_blocks)
