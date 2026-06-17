# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import unittest

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate, Shard
from torch.distributed.tensor.parallel import loss_parallel
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.loss import cross_entropy_loss, IGNORE_INDEX


def _reference_ce(pred, labels):
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(),
        labels.flatten(0, 1),
        reduction="sum",
        ignore_index=IGNORE_INDEX,
    )


class TestCrossEntropyDTensor(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    def _run_one(self, *, full_dtensor: bool, tp_shard_v: bool):
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.manual_seed(0)
        if full_dtensor:
            mesh = init_device_mesh(
                self.device_type, (2, 2, 2), mesh_dim_names=("dp_shard", "cp", "tp")
            )
            pred_pl = (Shard(0), Shard(1), Shard(2) if tp_shard_v else Replicate())
            labels_pl = (Shard(0), Shard(1), Replicate())
        else:
            mesh = init_device_mesh(self.device_type, (8,), mesh_dim_names=("tp",))
            pred_pl = (Shard(2) if tp_shard_v else Replicate(),)
            labels_pl = (Replicate(),)

        B, S, V = 4, 16, 64
        gen = torch.Generator(device=self.device_type).manual_seed(42)
        global_pred = torch.randn(B, S, V, device=self.device_type, generator=gen)
        global_labels = torch.randint(
            0, V, (B, S), device=self.device_type, dtype=torch.long, generator=gen
        )

        ref_loss = _reference_ce(global_pred, global_labels)

        pred_dt = distribute_tensor(global_pred.clone(), mesh, pred_pl)
        labels_dt = distribute_tensor(global_labels.clone(), mesh, labels_pl)
        pred_dt = pred_dt.detach().requires_grad_(True)

        # V-sharded pred requires the caller to provide loss_parallel; mirrors
        # how the trainer's train_context wraps the forward step.
        def _ctx():
            return loss_parallel() if tp_shard_v else contextlib.nullcontext()

        with _ctx():
            new_loss_dt = cross_entropy_loss(pred_dt, labels_dt)
            new_loss = new_loss_dt.full_tensor()

        # tp_shard_v=True routes through loss_parallel's Python decomposition
        # (differs from the C++ kernel at ULP level); tp_shard_v=False routes
        # through aten.nll_loss_forward on both paths.
        rtol, atol = (1e-6, 1e-6) if tp_shard_v else (0, 0)
        torch.testing.assert_close(new_loss, ref_loss, rtol=rtol, atol=atol)

        ref_pred = global_pred.clone().detach().requires_grad_(True)
        _reference_ce(ref_pred, global_labels).backward()
        with _ctx():
            new_loss_dt.backward()
        torch.testing.assert_close(
            pred_dt.grad.full_tensor(), ref_pred.grad, rtol=rtol, atol=atol
        )

    @with_comms
    def test_legacy_tp_disable_loss_parallel(self):
        self._run_one(full_dtensor=False, tp_shard_v=False)

    @with_comms
    def test_legacy_tp_loss_parallel(self):
        self._run_one(full_dtensor=False, tp_shard_v=True)

    @with_comms
    def test_full_dtensor_disable_loss_parallel(self):
        self._run_one(full_dtensor=True, tp_shard_v=False)


class TestCrossEntropyLossParallelGuard(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @with_comms
    def test_vocab_sharded_disable_loss_parallel_raises(self):
        # Vocab-sharded logits (Shard(2)) require the loss-parallel path; passing
        # loss_parallel=False is a contradictory config and must be rejected
        # rather than silently producing wrong numerics. world_size=2 suffices:
        # the guard fires inside local_map before any collective runs.
        mesh = init_device_mesh(self.device_type, (2,), mesh_dim_names=("tp",))
        B, S, V = 2, 8, 16
        gen = torch.Generator(device=self.device_type).manual_seed(0)
        pred = torch.randn(B, S, V, device=self.device_type, generator=gen)
        labels = torch.randint(
            0, V, (B, S), device=self.device_type, dtype=torch.long, generator=gen
        )
        pred_dt = distribute_tensor(pred, mesh, (Shard(2),))
        labels_dt = distribute_tensor(labels, mesh, (Replicate(),))
        with self.assertRaisesRegex(ValueError, "loss_parallel=False"):
            cross_entropy_loss(pred_dt, labels_dt, loss_parallel=False)


if __name__ == "__main__":
    unittest.main()
