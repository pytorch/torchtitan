# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Session release must preserve shared prefixes and pending transfer sources."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from torchtitan.rl.generator.vllm.simple_cpu_offload import SimpleCPUOffloadConnector
from vllm.config import CacheConfig, DeviceConfig, KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id


@pytest.mark.parametrize("finished_hashes", [(b"a", b"z"), (b"z", b"a")])
def test_cpu_eviction_preserves_active_hashes_and_rank_independent_order(
    finished_hashes,
):
    pool = BlockPool(7, True, 16)
    (
        active_primary,
        active_secondary,
        finished,
        duplicate,
        other,
        pinned,
    ) = pool.get_new_blocks(6)
    first_hash, second_hash = finished_hashes
    for block, hashes in (
        (active_primary, [b"active-primary", b"released-alias"]),
        (active_secondary, [b"released-primary", b"active-alias"]),
        (finished, [first_hash]),
        (duplicate, [first_hash]),
        (other, [second_hash]),
        (pinned, [b"pinned"]),
    ):
        for block_hash in hashes:
            pool._insert_block_hash(
                make_block_hash_with_group_id(block_hash, 0), block, num_tokens=16
            )
    pool.free_blocks([active_primary, active_secondary, finished, duplicate, other])
    connector = SimpleCPUOffloadConnector.__new__(SimpleCPUOffloadConnector)
    connector.scheduler_manager = SimpleNamespace(
        cpu_block_pool=pool, group_block_sizes=[16]
    )
    connector.release_session_cache(
        {b"released-alias", b"released-primary", first_hash, second_hash, b"pinned"},
        {b"active-primary", b"active-alias"},
    )
    # Hash order can differ across TP ranks. Physical eviction order must not.
    # Demotion preserves cached data; allocation actually reclaims it later.
    assert pool.get_cached_block(first_hash, [0]) == [finished]
    assert pool.get_new_blocks(5) == [
        finished,
        duplicate,
        other,
        active_primary,
        active_secondary,
    ]
    assert pinned.ref_cnt == 1


def test_cpu_offload_rejects_disabled_prefix_cache():
    config = VllmConfig(
        device_config=DeviceConfig("cpu"),
        cache_config=CacheConfig(enable_prefix_caching=False),
        kv_transfer_config=KVTransferConfig(
            kv_connector="SimpleCPUOffloadConnector", kv_role="kv_both"
        ),
    )
    with pytest.raises(ValueError, match="requires prefix caching"):
        SimpleCPUOffloadConnector(config, KVConnectorRole.SCHEDULER, Mock())


def test_cpu_save_snapshot_includes_unsubmitted_tail():
    connector = SimpleCPUOffloadConnector.__new__(SimpleCPUOffloadConnector)
    manager = SimpleNamespace(
        _store_event_to_blocks={3: object()},
        _abandoned_store_event_to_blocks={2: object()},
        _pending_finished_stores=[object()],
        _store_event_counter=4,
    )
    connector.scheduler_manager = manager
    pending = connector.get_pending_save_ids()
    assert pending == {2, 3, 4}
    manager._pending_finished_stores.clear()
    manager._store_event_to_blocks[4] = object()
    manager._store_event_counter = 5
    assert connector.get_pending_save_ids() == pending
    manager._abandoned_store_event_to_blocks.clear()
    manager._store_event_to_blocks = {5: object()}
    assert not pending & connector.get_pending_save_ids()
