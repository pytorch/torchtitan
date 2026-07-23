# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from spmd_types._test_utils import FakeProcessGroupTestCase
from spmd_types.checker import typecheck
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.debug import CommDebugMode
from torchtitan.components.muon_adapter import MuonAdapter
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig


def _has_batched_muon() -> bool:
    try:
        torch.optim.Muon([torch.nn.Parameter(torch.randn(1, 2, 3))])
    except ValueError:
        return False
    return True


class TestMuonAdapter(FakeProcessGroupTestCase):
    WORLD_SIZE = 2

    def _dtensor_parameter(self, local, placement, global_shape):
        stride = torch.empty(global_shape).stride()
        param = DTensor.from_local(
            local.clone(),
            self.mesh,
            [placement],
            run_check=False,
            shape=global_shape,
            stride=stride,
        )
        param.requires_grad_()
        return param

    def test_resolver_is_explicit(self):
        self.assertIs(OptimizersContainer._resolve_optimizer_cls("Muon"), MuonAdapter)

    def test_param_group_carries_explicit_matrix_shape(self):
        model = torch.nn.Module()
        model.head_weight = torch.nn.Parameter(torch.randn(6, 4))
        model.bias = torch.nn.Parameter(torch.randn(4))
        config = OptimizersContainer.Config(
            param_groups=[
                ParamGroupConfig(
                    pattern=r"^head_weight$",
                    optimizer_name="Muon",
                    optimizer_kwargs={
                        "lr": 0.02,
                        "matrix_shape": (3, 4),
                    },
                ),
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 1e-3},
                ),
            ]
        )

        container = config.build(model_parts=[model])
        optimizer = next(
            item for item in container.optimizers if isinstance(item, MuonAdapter)
        )
        self.assertEqual(optimizer.param_groups[0]["matrix_shape"], (3, 4))
        self.assertNotIn("fused", optimizer.param_groups[0])
        self.assertNotIn("foreach", optimizer.param_groups[0])

    def test_plain_untyped_muon_uses_adapter(self):
        model = torch.nn.Linear(4, 6, bias=False)
        config = OptimizersContainer.Config(
            param_groups=[
                ParamGroupConfig(
                    pattern=r"^weight$",
                    optimizer_name="Muon",
                    optimizer_kwargs={"lr": 0.02, "ns_steps": 1},
                )
            ]
        )

        container = config.build(model_parts=[model])
        optimizer = next(iter(container.optimizers))
        self.assertIsInstance(optimizer, MuonAdapter)
        model.weight.grad = torch.randn_like(model.weight)
        optimizer.step()

    def test_replicated_dtensor_step_keeps_persistent_state(self):
        param = self._dtensor_parameter(
            torch.randn(3, 4), Replicate(), global_shape=(3, 4)
        )
        param.grad = DTensor.from_local(
            torch.randn(3, 4),
            self.mesh,
            [Replicate()],
            run_check=False,
            shape=(3, 4),
            stride=(4, 1),
        )
        before = param.to_local().clone()
        optimizer = MuonAdapter([param], lr=0.02, ns_steps=1)

        comm_mode = CommDebugMode()
        with comm_mode, typecheck():
            optimizer.step()

        self.assertFalse(torch.equal(param.to_local(), before))
        self.assertEqual(comm_mode.get_total_counts(), 0)
        self.assertIn(param, optimizer.state)
        momentum = optimizer.state[param]["momentum_buffer"]
        self.assertIsInstance(momentum, DTensor)
        self.assertEqual(momentum.placements, param.placements)

    def test_matrix_dimension_shard_redistributes_for_compute(self):
        local_param = torch.randn(3, 4)
        local_grad = torch.randn(3, 4)
        param = self._dtensor_parameter(local_param, Shard(0), global_shape=(6, 4))
        param.grad = DTensor.from_local(
            local_grad,
            self.mesh,
            [Shard(0)],
            run_check=False,
            shape=(6, 4),
            stride=(4, 1),
        )
        optimizer = MuonAdapter([param], lr=0.02, ns_steps=1)
        reference = torch.nn.Parameter(torch.cat([local_param, local_param], dim=0))
        reference.grad = torch.cat([local_grad, local_grad], dim=0)
        reference_optimizer = torch.optim.Muon([reference], lr=0.02, ns_steps=1)

        comm_mode = CommDebugMode()
        with comm_mode:
            optimizer.step()
        reference_optimizer.step()

        self.assertGreater(comm_mode.get_total_counts(), 0)
        torch.testing.assert_close(param.to_local(), reference[:3])
        momentum = optimizer.state[param]["momentum_buffer"]
        reference_momentum = reference_optimizer.state[reference]["momentum_buffer"]
        self.assertIsInstance(momentum, DTensor)
        self.assertEqual(momentum.placements, param.placements)
        torch.testing.assert_close(momentum.to_local(), reference_momentum[:3])

    def test_validation_precedes_compute_views(self):
        param = self._dtensor_parameter(
            torch.randn(3, 4), Shard(0), global_shape=(6, 4)
        )
        param.grad = DTensor.from_local(
            torch.randn(3, 4),
            self.mesh,
            [Shard(0)],
            run_check=False,
            shape=(6, 4),
            stride=(4, 1),
        )
        complex_param = torch.nn.Parameter(torch.randn(3, 4, dtype=torch.complex64))
        complex_param.grad = torch.randn_like(complex_param)
        optimizer = MuonAdapter([param, complex_param], lr=0.02, ns_steps=1)

        comm_mode = CommDebugMode()
        with (
            comm_mode,
            self.assertRaisesRegex(
                RuntimeError, "Muon does not support complex parameters"
            ),
        ):
            optimizer.step()

        self.assertEqual(comm_mode.get_total_counts(), 0)
        self.assertEqual(len(optimizer.state), 0)
        self.assertFalse(hasattr(optimizer, "_compute_view_stack"))

    def test_dtensor_leading_shard_matrix_batch(self):
        # The same layout covers physical per-head [H, Dh, D] and grouped
        # per-expert [E, M, N] weights: only the leading owner axis is sharded.
        param = self._dtensor_parameter(
            torch.randn(2, 3, 4), Shard(0), global_shape=(4, 3, 4)
        )
        param.grad = DTensor.from_local(
            torch.randn(2, 3, 4),
            self.mesh,
            [Shard(0)],
            run_check=False,
            shape=(4, 3, 4),
            stride=(12, 4, 1),
        )
        before = param.to_local().clone()

        try:
            optimizer = MuonAdapter([param], lr=0.02, ns_steps=1)
            comm_mode = CommDebugMode()
            with comm_mode, typecheck():
                optimizer.step()
        except ValueError as error:
            if "2D" in str(error):
                self.skipTest("requires batched Muon from PyTorch PR #190597")
            raise

        self.assertFalse(torch.equal(param.to_local(), before))
        self.assertEqual(comm_mode.get_total_counts(), 0)
        momentum = optimizer.state[param]["momentum_buffer"]
        self.assertIsInstance(momentum, DTensor)
        self.assertEqual(momentum.shape, param.shape)
        self.assertEqual(momentum.placements, param.placements)

    def test_state_dict_keeps_dtensor_momentum(self):
        param = self._dtensor_parameter(
            torch.randn(3, 4), Replicate(), global_shape=(3, 4)
        )
        param.grad = DTensor.from_local(
            torch.randn(3, 4),
            self.mesh,
            [Replicate()],
            run_check=False,
            shape=(3, 4),
            stride=(4, 1),
        )
        optimizer = MuonAdapter([param], lr=0.02, ns_steps=1)
        optimizer.step()
        state_dict = optimizer.state_dict()

        restored_param = self._dtensor_parameter(
            torch.randn(3, 4), Replicate(), global_shape=(3, 4)
        )
        restored_optimizer = MuonAdapter([restored_param], lr=0.02, ns_steps=1)
        restored_optimizer.load_state_dict(state_dict)

        momentum = restored_optimizer.state[restored_param]["momentum_buffer"]
        self.assertIsInstance(momentum, DTensor)
        self.assertEqual(momentum.placements, restored_param.placements)

    def test_flattened_per_head_matrix_shape(self):
        heads, head_dim, model_dim = 2, 3, 4
        initial = torch.randn(heads, head_dim, model_dim)
        grad = torch.randn_like(initial)
        flat_param = self._dtensor_parameter(
            initial.reshape(-1, model_dim),
            Replicate(),
            global_shape=(heads * head_dim, model_dim),
        )
        flat_param.grad = DTensor.from_local(
            grad.reshape(-1, model_dim),
            self.mesh,
            [Replicate()],
            run_check=False,
            shape=(heads * head_dim, model_dim),
            stride=(model_dim, 1),
        )
        reference = torch.nn.Parameter(initial.clone())
        reference.grad = grad.clone()

        try:
            optimizer = MuonAdapter(
                [
                    {
                        "params": [flat_param],
                        "matrix_shape": (head_dim, model_dim),
                    }
                ],
                lr=0.02,
                ns_steps=1,
            )
            reference_optimizer = torch.optim.Muon(
                [reference], lr=0.02, ns_steps=1
            )
            optimizer.step()
            reference_optimizer.step()
        except ValueError as error:
            if "2D" in str(error):
                self.skipTest("requires batched Muon from PyTorch PR #190597")
            raise

        torch.testing.assert_close(
            flat_param.to_local().view_as(reference), reference
        )
        flat_momentum = optimizer.state[flat_param]["momentum_buffer"]
        torch.testing.assert_close(
            flat_momentum.to_local().view_as(reference),
            reference_optimizer.state[reference]["momentum_buffer"],
        )

    @unittest.skipUnless(_has_batched_muon(), "requires PyTorch PR #190597")
    def test_flattened_shard_redistributes_before_head_view(self):
        head_dim, model_dim = 3, 4
        local_param = torch.randn(head_dim, model_dim)
        local_grad = torch.randn_like(local_param)
        param = self._dtensor_parameter(
            local_param,
            Shard(0),
            global_shape=(2 * head_dim, model_dim),
        )
        param.grad = DTensor.from_local(
            local_grad,
            self.mesh,
            [Shard(0)],
            run_check=False,
            shape=(2 * head_dim, model_dim),
            stride=(model_dim, 1),
        )
        optimizer = MuonAdapter(
            [{"params": [param], "matrix_shape": (head_dim, model_dim)}],
            lr=0.02,
            ns_steps=1,
        )
        reference = torch.nn.Parameter(
            torch.stack([local_param, local_param], dim=0)
        )
        reference.grad = torch.stack([local_grad, local_grad], dim=0)
        reference_optimizer = torch.optim.Muon([reference], lr=0.02, ns_steps=1)

        optimizer.step()
        reference_optimizer.step()

        torch.testing.assert_close(param.to_local(), reference[0])
        momentum = optimizer.state[param]["momentum_buffer"]
        reference_momentum = reference_optimizer.state[reference]["momentum_buffer"]
        torch.testing.assert_close(momentum.to_local(), reference_momentum[0])


if __name__ == "__main__":
    unittest.main()
