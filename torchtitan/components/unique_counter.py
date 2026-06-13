from __future__ import annotations

import hashlib
from collections.abc import Iterable
from operator import length_hint
from typing import Any

import numpy as np
import torch
from torch import distributed as dist


_HASH_DTYPE = np.dtype([("hi", np.uint64), ("lo", np.uint64)])
_POPCOUNT_TABLE = torch.tensor([i.bit_count() for i in range(256)], dtype=torch.int64)
_cpu_group: dist.ProcessGroup | None = None


def _hash_name(name: str) -> tuple[int, int]:
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=16).digest()
    return int.from_bytes(digest[:8], "big"), int.from_bytes(digest[8:], "big")


def _default_cpu_group() -> dist.ProcessGroup | None:
    global _cpu_group
    if not dist.is_available() or not dist.is_initialized():
        return None
    if dist.get_world_size() == 1:
        return None
    if _cpu_group is None:
        _cpu_group = dist.new_group(backend="gloo")
    return _cpu_group


class BitsetUniqueCounter:
    def __init__(self, universe_names: Iterable[str]):
        count = length_hint(universe_names, -1)
        self._hashes = np.fromiter(
            (_hash_name(str(name)) for name in universe_names),
            dtype=_HASH_DTYPE,
            count=count,
        )
        self._hashes.sort(order=("hi", "lo"))
        if len(self._hashes) > 1:
            duplicate_hashes = self._hashes[1:] == self._hashes[:-1]
            if np.any(duplicate_hashes):
                raise ValueError("BitsetUniqueCounter universe contains duplicate 128-bit segment hashes")
        self._bitmap = torch.zeros((len(self._hashes) + 7) // 8, dtype=torch.uint8)

    @property
    def num_names(self) -> int:
        return len(self._hashes)

    @property
    def bitmap(self) -> torch.Tensor:
        return self._bitmap

    def _index_for_name(self, name: str) -> int:
        key = np.array(_hash_name(name), dtype=_HASH_DTYPE)
        idx = int(np.searchsorted(self._hashes, key))
        if idx >= len(self._hashes) or self._hashes[idx] != key:
            raise KeyError(f"unknown unique-counter name: {name}")
        return idx

    def update(self, names: Iterable[str]) -> None:
        for name in names:
            idx = self._index_for_name(name)
            byte_idx, bit_idx = divmod(idx, 8)
            self._bitmap[byte_idx] = self._bitmap[byte_idx] | (1 << bit_idx)

    def local_count(self) -> int:
        return self.count_bitmap(self._bitmap)

    def global_bitmap(self, group: dist.ProcessGroup | None = None) -> torch.Tensor:
        bitmap = self._bitmap.clone()
        if not dist.is_available() or not dist.is_initialized():
            return bitmap

        reduce_group = group if group is not None else _default_cpu_group()
        if reduce_group is None and dist.get_world_size() == 1:
            return bitmap
        dist.all_reduce(bitmap, op=dist.ReduceOp.BOR, group=reduce_group)
        return bitmap

    def global_count(self, group: dist.ProcessGroup | None = None) -> int:
        return self.count_bitmap(self.global_bitmap(group=group))

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_names": self.num_names,
            "bitmap": self._bitmap.clone(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return
        num_names = int(state_dict["num_names"])
        if num_names != self.num_names:
            raise ValueError(f"unique-counter universe size changed: {num_names} != {self.num_names}")
        bitmap = torch.as_tensor(state_dict["bitmap"], dtype=torch.uint8, device="cpu")
        if bitmap.shape != self._bitmap.shape:
            raise ValueError(f"unique-counter bitmap shape changed: {tuple(bitmap.shape)} != {tuple(self._bitmap.shape)}")
        self._bitmap.copy_(bitmap)

    @staticmethod
    def count_bitmap(bitmap: torch.Tensor) -> int:
        table = _POPCOUNT_TABLE.to(bitmap.device)
        return int(table[bitmap.to(torch.long)].sum().item())
