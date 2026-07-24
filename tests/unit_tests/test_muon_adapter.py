# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.distributed.tensor.placement_types as placement_types
from spmd_types._test_utils import FakeProcessGroupTestCase
from torch.distributed.tensor import DTensor, Replicate, Shard
from torchtitan.components.muon_adapter import MuonAdapter


_StridedShard = getattr(placement_types, "_StridedShard", None)


def _has_batched_muon() -> bool:
    try:
        torch.optim.Muon([torch.nn.Parameter(torch.randn(1, 2, 3))])
    except ValueError:
        return False
    return True


class TestMuonAdapter(FakeProcessGroupTestCase):
    WORLD_SIZE = 2

    def test_rejects_unsupported_implementation(self):
        param = torch.nn.Parameter(torch.randn(3, 4))
        with self.assertRaisesRegex(NotImplementedError, "fused or foreach"):
            MuonAdapter([{"params": [param], "fused": True}])

    def _sharded_dtensor(self, local, global_shape):
        stride = torch.empty(global_shape).stride()
        return DTensor.from_local(
            local,
            self.mesh,
            [Shard(0)],
            run_check=False,
            shape=global_shape,
            stride=stride,
        )

    def _dtensor_parameter(self, local, grad, global_shape):
        param = self._sharded_dtensor(local.clone(), global_shape)
        param.requires_grad_()
        param.grad = self._sharded_dtensor(grad, global_shape)
        return param

    @unittest.skipUnless(_has_batched_muon(), "requires PyTorch PR #190597")
    def test_ordinary_matrix_matches_muon_for_two_steps(self):
        initial = torch.arange(1, 13, dtype=torch.bfloat16).reshape(3, 4) / 13
        grads = (
            torch.arange(1, 13, dtype=torch.bfloat16).reshape(3, 4) / 17,
            torch.arange(12, 0, -1, dtype=torch.bfloat16).reshape(3, 4) / 19,
        )
        param = torch.nn.Parameter(initial.clone())
        reference = torch.nn.Parameter(initial.clone())
        kwargs = {
            "lr": 0.03,
            "weight_decay": 0.2,
            "momentum": 0.8,
            "nesterov": False,
            "ns_steps": 2,
        }
        optimizer = MuonAdapter([param], **kwargs)
        reference_optimizer = torch.optim.Muon([reference], **kwargs)
        expected_momentum = torch.zeros_like(param)

        for grad in grads:
            param.grad = grad.clone()
            reference.grad = grad.clone()
            expected_momentum.lerp_(grad, 1 - kwargs["momentum"])

            optimizer.step()
            reference_optimizer.step()

            torch.testing.assert_close(param, reference)
            torch.testing.assert_close(
                optimizer.state[param]["momentum_buffer"], expected_momentum
            )

    @unittest.skipUnless(_has_batched_muon(), "requires PyTorch PR #190597")
    def test_leading_expert_shard_matches_independent_muon_for_two_steps(self):
        local_experts, matrix_rows, matrix_cols = 2, 3, 4
        local_shape = (local_experts, matrix_rows, matrix_cols)
        global_shape = (self.WORLD_SIZE * local_experts, matrix_rows, matrix_cols)
        local_param = (
            torch.arange(1, 25, dtype=torch.bfloat16).reshape(local_shape) / 29
        )
        local_grads = (
            torch.arange(1, 25, dtype=torch.bfloat16).reshape(local_shape) / 31,
            torch.arange(24, 0, -1, dtype=torch.bfloat16).reshape(local_shape) / 37,
        )
        param = self._dtensor_parameter(
            local_param, local_grads[0].clone(), global_shape
        )
        reference_params = [
            torch.nn.Parameter(matrix.clone()) for matrix in local_param
        ]
        kwargs = {
            "lr": 0.03,
            "weight_decay": 0.2,
            "momentum": 0.8,
            "nesterov": False,
            "ns_steps": 2,
        }
        optimizer = MuonAdapter([param], **kwargs)
        reference_optimizer = torch.optim.Muon(reference_params, **kwargs)
        expected_momentum = torch.zeros_like(local_param)
        storage_placements = param.placements
        local_storage_ptr = param.to_local().data_ptr()
        persistent_momentum = None
        momentum_storage_ptr = None

        self.assertEqual(
            MuonAdapter._compute_placements(param, matrix_shape=None),
            storage_placements,
        )

        for local_grad in local_grads:
            param.grad = self._sharded_dtensor(local_grad.clone(), global_shape)
            for reference_param, reference_grad in zip(
                reference_params, local_grad, strict=True
            ):
                reference_param.grad = reference_grad.clone()
            expected_momentum.lerp_(local_grad, 1 - kwargs["momentum"])

            optimizer.step()
            reference_optimizer.step()

            self.assertIs(optimizer.param_groups[0]["params"][0], param)
            self.assertEqual(param.placements, storage_placements)
            self.assertEqual(param.to_local().data_ptr(), local_storage_ptr)
            torch.testing.assert_close(param.to_local(), torch.stack(reference_params))

            momentum = optimizer.state[param]["momentum_buffer"]
            reference_momentum = torch.stack(
                [
                    reference_optimizer.state[reference_param]["momentum_buffer"]
                    for reference_param in reference_params
                ]
            )
            self.assertIsInstance(momentum, DTensor)
            self.assertEqual(momentum.shape, param.shape)
            self.assertEqual(momentum.placements, storage_placements)
            if persistent_momentum is None:
                persistent_momentum = momentum
                momentum_storage_ptr = momentum.to_local().data_ptr()
            else:
                self.assertIs(momentum, persistent_momentum)
                self.assertEqual(momentum.to_local().data_ptr(), momentum_storage_ptr)
            torch.testing.assert_close(momentum.to_local(), expected_momentum)
            torch.testing.assert_close(momentum.to_local(), reference_momentum)

    @unittest.skipUnless(_has_batched_muon(), "requires PyTorch PR #190597")
    def test_matrix_shape_matches_batched_muon_for_two_steps(self):
        head_dim, model_dim = 3, 4
        global_shape = (2 * head_dim, model_dim)
        local_param = (
            torch.arange(1, 13, dtype=torch.bfloat16).reshape(head_dim, model_dim) / 13
        )
        local_grads = (
            torch.arange(1, 13, dtype=torch.bfloat16).reshape(head_dim, model_dim) / 17,
            torch.arange(12, 0, -1, dtype=torch.bfloat16).reshape(head_dim, model_dim)
            / 19,
        )
        param = self._dtensor_parameter(
            local_param, local_grads[0].clone(), global_shape
        )
        kwargs = {
            "lr": 0.03,
            "weight_decay": 0.2,
            "momentum": 0.8,
            "nesterov": False,
            "ns_steps": 2,
        }
        optimizer = MuonAdapter(
            [{"params": [param], "matrix_shape": (head_dim, model_dim)}],
            **kwargs,
        )

        reference = torch.nn.Parameter(torch.stack([local_param, local_param]))
        reference_optimizer = torch.optim.Muon([reference], **kwargs)
        expected_momentum = torch.zeros_like(local_param)
        storage_placements = param.placements
        persistent_momentum = None

        for local_grad in local_grads:
            param.grad = self._sharded_dtensor(local_grad.clone(), global_shape)
            reference.grad = torch.stack([local_grad, local_grad])
            expected_momentum.lerp_(local_grad, 1 - kwargs["momentum"])

            optimizer.step()
            reference_optimizer.step()

            self.assertIs(optimizer.param_groups[0]["params"][0], param)
            self.assertEqual(param.placements, storage_placements)
            torch.testing.assert_close(param.to_local(), reference[0])

            momentum = optimizer.state[param]["momentum_buffer"]
            self.assertIsInstance(momentum, DTensor)
            self.assertEqual(momentum.shape, param.shape)
            self.assertEqual(momentum.placements, storage_placements)
            if persistent_momentum is None:
                persistent_momentum = momentum
            else:
                self.assertIs(momentum, persistent_momentum)
            torch.testing.assert_close(momentum.to_local(), expected_momentum)
            torch.testing.assert_close(
                momentum.to_local(),
                reference_optimizer.state[reference]["momentum_buffer"][0],
            )


class TestMuonAdapterStridedPolicy(FakeProcessGroupTestCase):
    MESH_SHAPE = (2, 2)
    MESH_DIM_NAMES = ("dp", "tp")

    @unittest.skipIf(_StridedShard is None, "PyTorch has no private strided shard")
    def test_composed_strided_storage_uses_conservative_compute_layout(self):
        assert _StridedShard is not None
        storage_placements = (
            _StridedShard(0, split_factor=self.mesh["tp"].size()),
            Shard(0),
        )
        layout = DTensor.from_local(
            torch.zeros(2, 3, 4),
            self.mesh,
            storage_placements,
            run_check=False,
            shape=(8, 3, 4),
            stride=(12, 4, 1),
        )

        self.assertIs(type(storage_placements[0]), _StridedShard)
        self.assertEqual(
            MuonAdapter._compute_placements(layout, matrix_shape=None),
            (Replicate(), Shard(0)),
        )
        self.assertEqual(
            MuonAdapter._compute_placements(layout, matrix_shape=(3, 4)),
            (Replicate(), Replicate()),
        )
