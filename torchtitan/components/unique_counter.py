from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import timedelta
from typing import Any

from torch import distributed as dist

from torchtitan.tools.logging import logger

# TCPStore has a 8 MiB transport limit, we chunk the payload
# This is almost never hit, unless the list of string is very high
_STORE_CHUNK_SIZE = 4 * 1024 * 1024


class StringUniqueCounter:
    """Counts unique strings while syncing only newly seen local strings."""

    def __init__(self, store_prefix: str = "unique_ids", sync_timeout_seconds: int = 30) -> None:
        self._store_prefix = store_prefix
        self._sync_timeout_seconds = sync_timeout_seconds
        self._sync_idx = 0
        self._base_count = 0
        self._last_global_count = 0
        self._ids: set[str] = set()
        self._pending_ids: set[str] = set()
        self._global_ids: set[str] = set()

    def update(self, element_names: Iterable[str]) -> None:
        for name in element_names:
            element_name = str(name)
            if element_name not in self._ids:
                self._pending_ids.add(element_name)
                self._ids.add(element_name)

    def reset(self) -> None:
        self._ids.clear()
        self._pending_ids.clear()
        self._global_ids.clear()
        self._base_count = 0
        self._last_global_count = 0

    @property
    def last_global_count(self) -> int:
        return self._last_global_count

    def local_count(self) -> int:
        self._last_global_count = self._base_count + len(self._ids)
        return self._last_global_count

    @staticmethod
    def _store_chunks(store: dist.Store, key_prefix: str, payload: bytes) -> int:
        chunks = [payload[i : i + _STORE_CHUNK_SIZE] for i in range(0, len(payload), _STORE_CHUNK_SIZE)]
        for chunk_idx, chunk in enumerate(chunks):
            store.set(f"{key_prefix}:chunk:{chunk_idx}", chunk)
        return len(chunks)

    def _get_chunks(self, store: dist.Store, key_prefix: str, num_chunks: int) -> bytes:
        keys = [f"{key_prefix}:chunk:{i}" for i in range(num_chunks)]
        store.wait(keys, timedelta(seconds=self._sync_timeout_seconds))
        payload = b"".join(store.get(key) for key in keys)
        for key in keys:
            store.delete_key(key)
        return payload

    def global_count(self, group: dist.ProcessGroup | None = None) -> int:
        if not dist.is_available() or not dist.is_initialized():
            self._global_ids.update(self._pending_ids)
            self._pending_ids.clear()
            self._last_global_count = self._base_count + len(self._global_ids)
            return self._last_global_count

        group_rank = dist.get_rank(group=group)
        global_rank = dist.get_rank()
        world_size = dist.get_world_size(group=group)
        if world_size == 1:
            self._global_ids.update(self._pending_ids)
            self._pending_ids.clear()
            self._last_global_count = self._base_count + len(self._global_ids)
            return self._last_global_count

        group_ranks = dist.get_process_group_ranks(group) if group is not None else list(range(world_size))
        store = dist.PrefixStore(self._store_prefix, dist.distributed_c10d._get_default_store())
        store_key_prefix = f"sync:{self._sync_idx}:rank:"
        self._sync_idx += 1
        num_chunks = self._store_chunks(
            store,
            f"{store_key_prefix}{global_rank}",
            json.dumps(list(self._pending_ids)).encode(),
        )
        chunk_counts = [0] * world_size
        # just use dist to make rank 0 aware of num_chunks
        dist.all_gather_object(chunk_counts, num_chunks, group=group)

        global_count = None
        if group_rank == 0:
            for rank, chunk_count in zip(group_ranks, chunk_counts, strict=True):
                payload = self._get_chunks(store, f"{store_key_prefix}{rank}", chunk_count)
                self._global_ids.update(json.loads(payload.decode()))
            global_count = self._base_count + len(self._global_ids)

        count_holder = [global_count]
        src_rank = group_ranks[0]
        dist.broadcast_object_list(count_holder, src=src_rank, group=group)
        self._last_global_count = int(count_holder[0])
        self._pending_ids.clear()
        return self._last_global_count

    def state_dict(self) -> dict[str, Any]:
        return {"count": self._last_global_count}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.reset()
        if "count" not in state_dict:
            return
        self._base_count = int(state_dict["count"])
        self._last_global_count = self._base_count
        logger.warning(
            "StringUniqueCounter restored cardinality only; the underlying set was "
            "not recovered, so subsequent counts are approximate and assume disjoint sets"
        )
