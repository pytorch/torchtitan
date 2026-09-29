# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn
import torch_remat as remat
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.distributed.activation_storage import (
    ActivationStorage,
    cpu_offload_all,
    HostBackend,
)


class _CopyBackend:
    def put(self, tensor: torch.Tensor, stream) -> torch.Tensor:
        return tensor.clone()

    def get(self, payload: torch.Tensor, out: torch.Tensor, stream) -> None:
        out.copy_(payload)

    def free(self, payload: torch.Tensor) -> None:
        pass


class _Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.w = nn.Linear(dim, dim, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.w(x)) + x


class _Stage(nn.Module):
    def __init__(self, dim: int, n: int, wrap: str) -> None:
        super().__init__()
        self.layers = nn.ModuleDict({str(i): _Block(dim) for i in range(n)})
        for name, block in list(self.layers.items()):
            if wrap == "torch":
                self.layers[name] = checkpoint_wrapper(
                    block, checkpoint_impl=CheckpointImpl.NO_REENTRANT
                )
            else:
                block.forward = remat.checkpoint(region_name=f"layers.{name}")(
                    block.forward
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.layers.values():
            x = block(x)
        return x


def _offload(tensor: torch.Tensor, chunk) -> CheckpointPolicy:
    return CheckpointPolicy.MUST_CPU_OFFLOAD


def _grads(
    stage: nn.Module,
    x: torch.Tensor,
    storage: ActivationStorage | None,
    prefetch: bool,
):
    stage.zero_grad(set_to_none=True)
    x = x.clone().requires_grad_(True)
    if storage is None:
        y = stage(x)
    else:
        with storage.forward(0, 0, keep=(x,)):
            y = stage(x)
        if prefetch:
            storage.prefetch_first(0, 0)
    y.square().sum().backward()
    if storage is not None:
        storage.finish(0, 0)
    return [p.grad.clone() for p in stage.parameters()] + [x.grad.clone()]


class TestActivationStorage(unittest.TestCase):
    def _storage(self, stage: _Stage, policy, backend=None) -> ActivationStorage:
        storage = ActivationStorage(
            torch.device("cpu"),
            policy,
            {"host": backend or _CopyBackend()},
            min_tensor_bytes=0,
        )
        storage.register_stage(0, stage.layers)
        return storage

    def _assert_bitwise(self, ref, got) -> None:
        for a, b in zip(ref, got, strict=True):
            self.assertTrue(torch.equal(a, b))

    def _check(self, wrap: str) -> None:
        torch.manual_seed(0)
        stage = _Stage(64, 3, wrap)
        x = torch.randn(32, 64, dtype=torch.float64)
        ref = _grads(stage, x, None, False)
        for prefetch in (False, True):
            storage = self._storage(stage, _offload)
            self._assert_bitwise(ref, _grads(stage, x, storage, prefetch))
            self.assertGreater(storage.stats["host_bytes"], 0)
            # the first layer's input is the stage input the caller keeps; later layers' inputs move
            self.assertEqual(storage.stats["late_fetches"], 0 if prefetch else 1)
            self.assertFalse(storage._chunks)

    def test_torch_checkpoint_offloads_round_trip_bitwise(self) -> None:
        self._check("torch")

    def test_remat_checkpoint_offloads_round_trip_bitwise(self) -> None:
        self._check("remat")

    def test_a_storage_pinned_during_the_forward_is_never_moved(self) -> None:
        torch.manual_seed(0)
        stage = _Stage(64, 3, "torch")
        x = torch.randn(32, 64, dtype=torch.float64)
        ref = _grads(stage, x, None, False)
        asked: list[int] = []

        def policy(tensor: torch.Tensor, chunk) -> CheckpointPolicy:
            asked.append(tensor.untyped_storage().data_ptr())
            return CheckpointPolicy.MUST_CPU_OFFLOAD

        storage = self._storage(stage, policy)
        pinned: list[int] = []

        def pin_output(module, args, output):
            storage.pin(output)
            pinned.append(output.untyped_storage().data_ptr())

        handle = stage.layers["0"].register_forward_hook(pin_output)
        got = _grads(stage, x, storage, True)
        handle.remove()
        self._assert_bitwise(ref, got)
        self.assertTrue(pinned and asked)
        self.assertFalse(set(pinned) & set(asked))
        self.assertGreater(storage.stats["pinned_peak_bytes"], 0)

    def test_saves_the_policy_keeps_are_left_to_autograd(self) -> None:
        torch.manual_seed(0)
        stage = _Stage(64, 3, "remat")
        x = torch.randn(32, 64, dtype=torch.float64)
        ref = _grads(stage, x, None, False)
        storage = self._storage(stage, lambda t, c: CheckpointPolicy.PREFER_SAVE)
        self._assert_bitwise(ref, _grads(stage, x, storage, False))
        self.assertEqual(storage.stats["host_bytes"], 0)
        x = x.clone().requires_grad_(True)
        with storage.forward(0, 0, keep=(x,)):
            y = stage(x)
        self.assertFalse(storage._chunks)
        y.square().sum().backward()
        storage.finish(0, 0)
        self.assertFalse(storage._started)

    def test_cpu_offload_all_keeps_the_skipped_layer(self) -> None:
        torch.manual_seed(0)
        stage = _Stage(64, 3, "torch")
        x = torch.randn(32, 64, dtype=torch.float64)
        ref = _grads(stage, x, None, False)
        moved: list[int] = []
        storage = self._storage(stage, cpu_offload_all(skip_layers={2}))
        original = storage._route_pending

        def record() -> None:
            original()
            moved.extend(sorted({chunk[2] for chunk in storage._chunks}))

        storage._route_pending = record
        self._assert_bitwise(ref, _grads(stage, x, storage, True))
        self.assertEqual(moved, [1])

    def test_a_full_host_backend_keeps_the_save_on_the_device(self) -> None:
        torch.manual_seed(0)
        stage = _Stage(64, 3, "torch")
        x = torch.randn(32, 64, dtype=torch.float64)
        ref = _grads(stage, x, None, False)
        one_save = 32 * 64 * torch.float64.itemsize
        storage = self._storage(stage, _offload, HostBackend(capacity_bytes=one_save))
        self._assert_bitwise(ref, _grads(stage, x, storage, True))
        self.assertEqual(storage.stats["host_bytes"], one_save)
        self.assertEqual(storage.stats["host_full"], 1)

    def test_a_recompute_policy_is_refused(self) -> None:
        stage = _Stage(64, 2, "torch")
        storage = self._storage(stage, lambda t, c: CheckpointPolicy.PREFER_RECOMPUTE)
        x = torch.randn(32, 64, dtype=torch.float64, requires_grad=True)
        with self.assertRaises(ValueError):
            with storage.forward(0, 0, keep=(x,)):
                stage(x)


if __name__ == "__main__":
    unittest.main()
