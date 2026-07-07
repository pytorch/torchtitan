# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU unit tests for the custom_comm all_reduce selection logic.

These exercise `_select_algo` and its eligibility gating without a GPU or a real
process group by stubbing the group-size / multicast / local-world helpers.
"""

import torch

from torchtitan.distributed import comms
from torchtitan.distributed.comms import _Algo


def _make(numel: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.empty(numel, dtype=dtype)


def _setup(monkeypatch, *, world_size=8, local_world_size=8, multicast=True):
    monkeypatch.setattr(comms, "_group_world_size", lambda group_name: world_size)
    monkeypatch.setattr(comms, "_local_world_size", lambda: local_world_size)
    monkeypatch.setattr(comms, "_has_multicast", lambda device_index: multicast)


def _bytes_to_numel(nbytes: int, dtype=torch.bfloat16) -> int:
    return nbytes // dtype.itemsize


def test_tiny_with_multicast_is_multimem_one_shot(monkeypatch):
    _setup(monkeypatch, multicast=True)
    x = _make(_bytes_to_numel(32 * 1024))  # <= 64 KiB, multicast -> mm one-shot
    assert comms._select_algo(x, "sum", "g") is _Algo.MULTIMEM_ONE_SHOT


def test_multimem_one_shot_upper_boundary(monkeypatch):
    _setup(monkeypatch, multicast=True)
    x = _make(_bytes_to_numel(64 * 1024))  # exactly the crossover, still one-shot
    assert comms._select_algo(x, "sum", "g") is _Algo.MULTIMEM_ONE_SHOT


def test_small_with_multicast_is_multimem_two_shot(monkeypatch):
    _setup(monkeypatch, multicast=True)
    x = _make(_bytes_to_numel(128 * 1024))  # > 64 KiB, multicast -> mm two-shot
    assert comms._select_algo(x, "sum", "g") is _Algo.MULTIMEM_TWO_SHOT


def test_tiny_without_multicast_is_one_shot(monkeypatch):
    _setup(monkeypatch, multicast=False)
    x = _make(_bytes_to_numel(64 * 1024))  # <= 128 KiB, no multicast -> one-shot
    assert comms._select_algo(x, "sum", "g") is _Algo.ONE_SHOT


def test_one_shot_upper_boundary(monkeypatch):
    _setup(monkeypatch, multicast=False)
    x = _make(_bytes_to_numel(128 * 1024))  # exactly the crossover, still one-shot
    assert comms._select_algo(x, "sum", "g") is _Algo.ONE_SHOT


def test_small_without_multicast_is_two_shot(monkeypatch):
    _setup(monkeypatch, multicast=False)
    x = _make(_bytes_to_numel(256 * 1024))  # > 128 KiB, no multicast -> two-shot
    assert comms._select_algo(x, "sum", "g") is _Algo.TWO_SHOT


def test_multimem_medium_with_multicast(monkeypatch):
    _setup(monkeypatch, multicast=True)
    x = _make(_bytes_to_numel(8 * 1024 * 1024))  # 8 MiB, multicast -> mm two-shot
    assert comms._select_algo(x, "sum", "g") is _Algo.MULTIMEM_TWO_SHOT


def test_two_shot_medium_without_multicast(monkeypatch):
    _setup(monkeypatch, multicast=False)
    x = _make(_bytes_to_numel(8 * 1024 * 1024))  # 8 MiB, no multicast -> two-shot
    assert comms._select_algo(x, "sum", "g") is _Algo.TWO_SHOT


def test_large_falls_back_to_nccl(monkeypatch):
    _setup(monkeypatch, multicast=True)
    x = _make(_bytes_to_numel(64 * 1024 * 1024))  # 64 MiB > multimem max
    assert comms._select_algo(x, "sum", "g") is _Algo.NCCL


def test_two_shot_large_falls_back(monkeypatch):
    _setup(monkeypatch, multicast=False)
    x = _make(_bytes_to_numel(32 * 1024 * 1024))  # 32 MiB > two-shot max
    assert comms._select_algo(x, "sum", "g") is _Algo.NCCL


def test_non_sum_falls_back(monkeypatch):
    _setup(monkeypatch)
    x = _make(_bytes_to_numel(8 * 1024 * 1024))
    assert comms._select_algo(x, "avg", "g") is _Algo.NCCL


def test_unsupported_dtype_falls_back(monkeypatch):
    _setup(monkeypatch)
    x = _make(_bytes_to_numel(8 * 1024 * 1024) // 4, dtype=torch.int64)
    assert comms._select_algo(x, "sum", "g") is _Algo.NCCL


def test_unsupported_world_size_falls_back(monkeypatch):
    _setup(monkeypatch, world_size=3)  # not in (2, 4, 8)
    x = _make(_bytes_to_numel(8 * 1024 * 1024))
    assert comms._select_algo(x, "sum", "g") is _Algo.NCCL


def test_inter_node_falls_back(monkeypatch):
    # group larger than the local node -> not intra-node.
    _setup(monkeypatch, world_size=8, local_world_size=4)
    x = _make(_bytes_to_numel(8 * 1024 * 1024))
    assert comms._select_algo(x, "sum", "g") is _Algo.NCCL


def test_misaligned_falls_back(monkeypatch):
    _setup(monkeypatch, world_size=8)
    # 8 MiB + 2 elements -> not a multiple of world_size * (16 / itemsize).
    x = _make(_bytes_to_numel(8 * 1024 * 1024) + 2)
    assert comms._select_algo(x, "sum", "g") is _Algo.NCCL


def test_non_contiguous_falls_back(monkeypatch):
    _setup(monkeypatch)
    x = _make(_bytes_to_numel(16 * 1024 * 1024))[::2]  # strided view
    assert comms._select_algo(x, "sum", "g") is _Algo.NCCL


def test_deterministic_mode_falls_back(monkeypatch):
    _setup(monkeypatch)
    x = _make(_bytes_to_numel(8 * 1024 * 1024))
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    assert comms._select_algo(x, "sum", "g") is _Algo.NCCL
