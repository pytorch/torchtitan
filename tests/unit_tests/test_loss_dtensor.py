# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.loss import cross_entropy_loss, IGNORE_INDEX
from torchtitan.distributed.utils import get_spmd_backend, set_spmd_backend


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

    @with_comms
    def test_vocab_parallel_logits(self):
        """``cross_entropy_loss`` with a vocab-sharded DTensor ``pred``.

        This is the partial_dtensor path: only ``pred`` is a DTensor, labels
        come out of the dataloader as plain replicated tensors.
        """
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.manual_seed(0)
        mesh = init_device_mesh(self.device_type, (8,), mesh_dim_names=("tp",))

        B, S, V = 4, 16, 64
        gen = torch.Generator(device=self.device_type).manual_seed(42)
        global_pred = torch.randn(B, S, V, device=self.device_type, generator=gen)
        global_labels = torch.randint(
            0, V, (B, S), device=self.device_type, dtype=torch.long, generator=gen
        )

        ref_loss = _reference_ce(global_pred, global_labels)

        pred_dt = distribute_tensor(global_pred.clone(), mesh, (Shard(2),))
        pred_dt = pred_dt.detach().requires_grad_(True)
        labels = global_labels.clone()

        previous_backend = get_spmd_backend()
        set_spmd_backend("partial_dtensor")
        try:
            loss = cross_entropy_loss(pred_dt, labels)
            loss.backward()
        finally:
            set_spmd_backend(previous_backend)

        # The vocab-parallel decomposition differs from the fused C++ kernel
        # at the ULP level, so compare with a tolerance.
        rtol, atol = 1e-6, 1e-6
        torch.testing.assert_close(loss, ref_loss, rtol=rtol, atol=atol)

        ref_pred = global_pred.clone().detach().requires_grad_(True)
        _reference_ce(ref_pred, global_labels).backward()
        torch.testing.assert_close(
            pred_dt.grad.full_tensor(), ref_pred.grad, rtol=rtol, atol=atol
        )


if __name__ == "__main__":
    unittest.main()
