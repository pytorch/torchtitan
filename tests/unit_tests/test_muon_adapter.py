# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from spmd_types._test_utils import FakeProcessGroupTestCase
from torch.distributed.tensor import DTensor, Shard
from torchtitan.components.muon_adapter import MuonAdapter


def _has_batched_muon() -> bool:
    try:
        torch.optim.Muon([torch.nn.Parameter(torch.randn(1, 2, 3))])
    except ValueError:
        return False
    return True


class TestMuonAdapter(FakeProcessGroupTestCase):
    WORLD_SIZE = 2

    def _dtensor_parameter(self, local, grad, global_shape):
        stride = torch.empty(global_shape).stride()
        param = DTensor.from_local(
            local.clone(),
            self.mesh,
            [Shard(0)],
            run_check=False,
            shape=global_shape,
            stride=stride,
        )
        param.requires_grad_()
        param.grad = DTensor.from_local(
            grad,
            self.mesh,
            [Shard(0)],
            run_check=False,
            shape=global_shape,
            stride=stride,
        )
        return param

    @unittest.skipUnless(_has_batched_muon(), "requires PyTorch PR #190597")
    def test_matrix_shape_matches_batched_muon(self):
        head_dim, model_dim = 3, 4
        local_param = torch.randn(head_dim, model_dim)
        local_grad = torch.randn_like(local_param)
        param = self._dtensor_parameter(
            local_param, local_grad, (2 * head_dim, model_dim)
        )
        optimizer = MuonAdapter(
            [{"params": [param], "matrix_shape": (head_dim, model_dim)}],
            lr=0.02,
            ns_steps=1,
        )

        reference = torch.nn.Parameter(torch.stack([local_param, local_param]))
        reference.grad = torch.stack([local_grad, local_grad])
        reference_optimizer = torch.optim.Muon([reference], lr=0.02, ns_steps=1)

        optimizer.step()
        reference_optimizer.step()

        torch.testing.assert_close(param.to_local(), reference[0])
        momentum = optimizer.state[param]["momentum_buffer"]
        self.assertEqual(momentum.placements, param.placements)
        torch.testing.assert_close(
            momentum.to_local(),
            reference_optimizer.state[reference]["momentum_buffer"][0],
        )
