# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import timedelta
from typing import Any

from torch import distributed as dist


class StringUniqueCounter:
    """Counts unique strings while syncing only newly seen local strings."""

    def __init__(
        self, store_prefix: str = "unique_ids", sync_timeout_seconds: int = 30
    ) -> None:
        self._store_prefix = store_prefix
        self._sync_timeout_seconds = sync_timeout_seconds
        self._sync_idx = 0
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

    def local_count(self) -> int:
        return len(self._ids)

    def global_count(self, group: dist.ProcessGroup | None = None) -> int:
        if not dist.is_available() or not dist.is_initialized():
            self._global_ids.update(self._pending_ids)
            self._pending_ids.clear()
            return len(self._global_ids)

        group_rank = dist.get_rank(group=group)
        global_rank = dist.get_rank()
        world_size = dist.get_world_size(group=group)
        if world_size == 1:
            self._global_ids.update(self._pending_ids)
            self._pending_ids.clear()
            return len(self._global_ids)

        group_ranks = (
            dist.get_process_group_ranks(group)
            if group is not None
            else list(range(world_size))
        )
        store = dist.PrefixStore(
            self._store_prefix, dist.distributed_c10d._get_default_store()
        )
        store_key_prefix = f"sync:{self._sync_idx}:rank:"
        self._sync_idx += 1
        store.set(
            f"{store_key_prefix}{global_rank}", json.dumps(list(self._pending_ids))
        )
        store_keys = [f"{store_key_prefix}{rank}" for rank in group_ranks]

        global_count = None
        if group_rank == 0:
            store.wait(store_keys, timedelta(seconds=self._sync_timeout_seconds))
            for store_key in store_keys:
                self._global_ids.update(json.loads(store.get(store_key).decode()))
                store.delete_key(store_key)
            global_count = len(self._global_ids)

        count_holder = [global_count]
        src_rank = group_ranks[0]
        dist.broadcast_object_list(count_holder, src=src_rank, group=group)
        self._pending_ids.clear()
        assert count_holder[0] is not None
        return int(count_holder[0])

    def state_dict(self) -> dict[str, Any]:
        return {"ids": sorted(self._ids)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return
        if "ids" not in state_dict:
            self.reset()
            return
        self._ids = {str(name) for name in state_dict["ids"]}
        self._pending_ids = set(self._ids)
        self._global_ids.clear()
