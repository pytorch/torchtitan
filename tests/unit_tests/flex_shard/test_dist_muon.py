# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# @lint-ignore-every CITRINE

import unittest
from unittest import mock

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.components.optimizer.utils import (
    get_flat_optim_state_dict,
    init_optim_state,
    load_flat_optim_state_dict,
)
from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    build_dist_muon,
    ComputeLayout,
    Owned,
)
from torchtitan.distributed.flex_shard.dist_muon import (
    _adjust_muon_learning_rate,
    _LocalMatrixBatch,
    DistMuon,
)


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestDistMuon(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @property
    def device_type(self):
        return "cuda"

    @with_comms
    def test_matches_plain_muon_across_flat_checkpoint(self):
        lr = 0.03
        weight_decay = 0.2
        mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("dp_shard",),
        )
        device = torch.device(self.device_type, self.rank)
        stack_shapes = {
            # Two storage shards each own six rows, so their boundary splits
            # the middle matrix and exercises overshard redistribution.
            "layers.0.attention.oversharded": (3, 4, 3),
            # These aligned siblings remain local and share one batched NS
            # call because their local matrices are compatible.
            "layers.0.attention.wq": (4, 5, 3),
            "layers.0.attention.wkv": (4, 5, 3),
        }

        def make_parameter(value: torch.Tensor) -> torch.nn.Parameter:
            return torch.nn.Parameter(
                distribute_tensor(value.clone(), mesh, (Shard(0),))
            )

        def make_optimizer(
            redistributed: torch.nn.Parameter,
            stacks: dict[str, torch.nn.Parameter],
            ns_steps: int = 2,
        ):
            redistributed_fqn = "layers.0.redistributed"
            oversharded_fqn = "layers.0.attention.oversharded"
            aligned_fqns = ("layers.0.attention.wq", "layers.0.attention.wkv")
            aligned_compute_sharding = ComputeLayout(
                shardings_by_mesh_axis={
                    "dp_shard": BlockShard(dim=0, block_size=5),
                }
            )
            return build_dist_muon(
                [
                    {
                        "params": [
                            redistributed,
                            stacks[oversharded_fqn],
                            *(stacks[fqn] for fqn in aligned_fqns),
                        ],
                        "param_names": [
                            redistributed_fqn,
                            oversharded_fqn,
                            *aligned_fqns,
                        ],
                    }
                ],
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.8,
                nesterov=True,
                ns_steps=ns_steps,
                compute_sharding_by_fqn={
                    redistributed_fqn: ComputeLayout(
                        shardings_by_mesh_axis={
                            "dp_shard": Owned(),
                        },
                    ),
                    oversharded_fqn: ComputeLayout(
                        shardings_by_mesh_axis={
                            "dp_shard": BlockShard(dim=0, block_size=4),
                        },
                    ),
                    **{fqn: aligned_compute_sharding for fqn in aligned_fqns},
                },
                bucket_configs=[
                    BucketConfig(
                        patterns=("layers.0.*",),
                        name="layers.0",
                    )
                ],
            )

        def set_grads(
            redistributed: torch.nn.Parameter,
            stacks: dict[str, torch.nn.Parameter],
            redistributed_grad: torch.Tensor,
            stack_grads: dict[str, torch.Tensor],
        ) -> None:
            redistributed.grad = distribute_tensor(
                redistributed_grad.clone(), mesh, (Shard(0),)
            )
            for name, parameter in stacks.items():
                grad = stack_grads[name]
                parameter.grad = distribute_tensor(grad.clone(), mesh, (Shard(0),))

        def assert_matches_reference(
            redistributed: torch.nn.Parameter,
            stacks: dict[str, torch.nn.Parameter],
            reference_redistributed: torch.nn.Parameter,
            reference_stacks: dict[str, tuple[torch.nn.Parameter, ...]],
            stacks_before: dict[str, torch.Tensor],
            reference_stacks_before: dict[str, tuple[torch.Tensor, ...]],
        ) -> None:
            rank = mesh.get_local_rank()
            expected_redistributed = reference_redistributed.detach().chunk(
                self.world_size, dim=0
            )[rank]
            torch.testing.assert_close(
                redistributed.to_local(),
                expected_redistributed,
                rtol=0,
                atol=0,
            )

            for name, parameter in stacks.items():
                reference_blocks = reference_stacks[name]
                expected = torch.cat(
                    [reference.detach() for reference in reference_blocks], dim=0
                )
                expected_before = torch.cat(reference_stacks_before[name], dim=0)
                local_rows, row_offset = Shard.local_shard_size_and_offset(
                    expected.shape[0], self.world_size, rank
                )
                expected = expected.narrow(0, row_offset, local_rows)
                expected_before = expected_before.narrow(0, row_offset, local_rows)
                decay = 1 - lr * weight_decay
                adjusted_lr = _adjust_muon_learning_rate(
                    lr, None, reference_blocks[0].shape
                )
                actual_update = (
                    stacks_before[name] * decay - parameter.to_local()
                ) / adjusted_lr
                expected_update = (expected_before * decay - expected) / adjusted_lr
                # Batched BF16 Newton-Schulz can differ slightly across GEMM schedules.
                torch.testing.assert_close(
                    actual_update,
                    expected_update,
                    rtol=0,
                    atol=2e-2,
                )

        values = {}
        start = 12
        for name, (num_matrices, rows, columns) in stack_shapes.items():
            numel = num_matrices * rows * columns
            values[name] = (
                torch.arange(start, start + numel, device=device)
                .reshape(num_matrices * rows, columns)
                .float()
                .div_(10)
            )
            start += numel

        redistributed_value = (
            torch.arange(12, device=device).reshape(4, 3).float().div_(10).add_(1)
        )
        redistributed = make_parameter(redistributed_value)
        stacks = {name: make_parameter(value) for name, value in values.items()}
        optimizer = make_optimizer(redistributed, stacks)
        self.assertIs(type(optimizer), DistMuon)
        with self.assertRaisesRegex(RuntimeError, "parameter groups are frozen"):
            optimizer.add_param_group({"params": []})

        reference_redistributed = torch.nn.Parameter(redistributed_value.clone())
        reference_stacks = {
            name: tuple(
                torch.nn.Parameter(matrix.clone())
                for matrix in value.view(stack_shapes[name])
            )
            for name, value in values.items()
        }
        reference_optimizer = torch.optim.Muon(
            [
                reference_redistributed,
                *(
                    parameter
                    for stack in reference_stacks.values()
                    for parameter in stack
                ),
            ],
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
        )

        def step_and_assert(
            current_optimizer,
            current_redistributed: torch.nn.Parameter,
            current_stacks: dict[str, torch.nn.Parameter],
            redistributed_grad: torch.Tensor,
            stack_grads: dict[str, torch.Tensor],
        ) -> None:
            stacks_before = {
                name: parameter.to_local().clone()
                for name, parameter in current_stacks.items()
            }
            reference_stacks_before = {
                name: tuple(parameter.detach().clone() for parameter in stack)
                for name, stack in reference_stacks.items()
            }
            set_grads(
                current_redistributed,
                current_stacks,
                redistributed_grad,
                stack_grads,
            )
            reference_redistributed.grad = redistributed_grad.clone()
            for name, references in reference_stacks.items():
                for parameter, grad in zip(
                    references,
                    stack_grads[name].view(stack_shapes[name]),
                    strict=True,
                ):
                    parameter.grad = grad.clone()

            current_optimizer.step()
            reference_optimizer.step()
            assert_matches_reference(
                current_redistributed,
                current_stacks,
                reference_redistributed,
                reference_stacks,
                stacks_before,
                reference_stacks_before,
            )

        first_redistributed_grad = (
            torch.arange(1, 13, device=device).reshape(4, 3).float().div_(17)
        )
        first_stack_grads = {
            name: torch.arange(1, value.numel() + 1, device=device)
            .reshape_as(value)
            .float()
            .div_(19 + 2 * index)
            for index, (name, value) in enumerate(values.items())
        }
        first_stack_grads["layers.0.attention.wkv"] = (
            first_stack_grads["layers.0.attention.wkv"].flip(1).contiguous()
        )
        step_and_assert(
            optimizer,
            redistributed,
            stacks,
            first_redistributed_grad,
            first_stack_grads,
        )

        flat_state_dict = get_flat_optim_state_dict(optimizer)
        resumed_redistributed = make_parameter(redistributed.full_tensor().detach())
        resumed_stacks = {
            name: make_parameter(parameter.full_tensor().detach())
            for name, parameter in stacks.items()
        }
        resumed_optimizer = make_optimizer(
            resumed_redistributed,
            resumed_stacks,
            ns_steps=3,
        )
        init_optim_state(resumed_optimizer)
        load_flat_optim_state_dict(resumed_optimizer, flat_state_dict)

        step_and_assert(
            resumed_optimizer,
            resumed_redistributed,
            resumed_stacks,
            first_redistributed_grad.flip(0).contiguous(),
            {
                name: grad.flip(0).contiguous()
                for name, grad in first_stack_grads.items()
            },
        )


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestDistMuonLocalMatrixBatching(DTensorTestBase):
    """Batching follows local matrix compatibility, never FQN structure."""

    @property
    def world_size(self):
        return 2

    @property
    def device_type(self):
        return "cuda"

    def _mesh(self):
        return init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("dp_shard",),
        )

    def _stack(self, mesh, num_matrices, matrix_rows, columns):
        device = torch.device(self.device_type, self.rank)
        value = (
            torch.arange(num_matrices * matrix_rows * columns, device=device)
            .reshape(num_matrices * matrix_rows, columns)
            .float()
            .div_(11)
        )
        return torch.nn.Parameter(distribute_tensor(value, mesh, (Shard(0),)))

    @staticmethod
    def _block_layout(block_size):
        return ComputeLayout(
            shardings_by_mesh_axis={
                "dp_shard": BlockShard(dim=0, block_size=block_size)
            }
        )

    def _executions(self, optimizer):
        return [
            execution
            for plan in optimizer._local_execution_plans.values()
            for execution in plan
        ]

    def _build(self, params, layouts):
        return build_dist_muon(
            [{"params": list(params.values()), "param_names": list(params)}],
            lr=0.03,
            weight_decay=0.2,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
            compute_sharding_by_fqn=layouts,
            bucket_configs=[BucketConfig(name="layers", patterns=("layers.*",))],
        )

    @with_comms
    def test_batches_compatible_matrices_across_layers(self):
        # One bucket spanning two layers. Sibling FQNs differ in their parent,
        # so an FQN-derived grouping would leave all four unbatched.
        fqns = [
            "layers.1.attention.wq",
            "layers.2.attention.wq",
            "layers.1.attention.wkv_b",
            "layers.2.attention.wkv_b",
        ]
        mesh = self._mesh()
        params = {fqn: self._stack(mesh, 4, 5, 3) for fqn in fqns}
        optimizer = self._build(params, {fqn: self._block_layout(5) for fqn in fqns})

        executions = self._executions(optimizer)
        batches = [e for e in executions if isinstance(e, _LocalMatrixBatch)]
        self.assertEqual(len(batches), 1)
        self.assertEqual(sorted(s.layout.fqn for s in batches[0].slices), sorted(fqns))
        self.assertEqual(
            [e.fqn for e in executions if not isinstance(e, _LocalMatrixBatch)], []
        )

    @with_comms
    def test_separates_incompatible_matrix_shapes(self):
        mesh = self._mesh()
        wide = ("layers.1.attention.wq", "layers.2.attention.wq")
        narrow = ("layers.1.attention.wkv_b", "layers.2.attention.wkv_b")
        params = {
            **{fqn: self._stack(mesh, 4, 5, 3) for fqn in wide},
            **{fqn: self._stack(mesh, 4, 4, 3) for fqn in narrow},
        }
        optimizer = self._build(
            params,
            {
                **{fqn: self._block_layout(5) for fqn in wide},
                **{fqn: self._block_layout(4) for fqn in narrow},
            },
        )

        batches = [
            e for e in self._executions(optimizer) if isinstance(e, _LocalMatrixBatch)
        ]
        self.assertEqual(len(batches), 2)
        self.assertEqual(
            sorted(
                tuple(sorted(s.layout.fqn for s in batch.slices)) for batch in batches
            ),
            sorted((tuple(sorted(wide)), tuple(sorted(narrow)))),
        )

    @with_comms
    def test_batches_do_not_span_buckets(self):
        # Local work is executed one bucket at a time, so compatible tensors in
        # different buckets are never resident together and cannot share a call.
        mesh = self._mesh()
        first = ("layers.1.attention.wq", "layers.1.attention.wkv_b")
        second = ("layers.2.attention.wq", "layers.2.attention.wkv_b")
        params = {fqn: self._stack(mesh, 4, 5, 3) for fqn in (*first, *second)}
        optimizer = build_dist_muon(
            [{"params": list(params.values()), "param_names": list(params)}],
            lr=0.03,
            ns_steps=2,
            compute_sharding_by_fqn={fqn: self._block_layout(5) for fqn in params},
            bucket_configs=[
                BucketConfig(name="layers.1", patterns=("layers.1.*",)),
                BucketConfig(name="layers.2", patterns=("layers.2.*",)),
            ],
        )

        batches = [
            e for e in self._executions(optimizer) if isinstance(e, _LocalMatrixBatch)
        ]
        self.assertEqual(len(batches), 2)
        self.assertEqual(
            sorted(
                tuple(sorted(s.layout.fqn for s in batch.slices)) for batch in batches
            ),
            sorted((tuple(sorted(first)), tuple(sorted(second)))),
        )


@unittest.skipUnless(torch.cuda.device_count() >= 4, "requires four CUDA devices")
class TestDistMuonInitialExpertStorageContract(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @property
    def device_type(self):
        return "cuda"

    @with_comms
    def test_preserves_ep_shard_during_efsdp_redistribution(self):
        lr = 0.03
        weight_decay = 0.2
        mesh = init_device_mesh(
            self.device_type,
            (2, 2),
            mesh_dim_names=("efsdp", "ep"),
        )
        num_experts = 3
        self.assertLess(
            num_experts,
            mesh["efsdp"].size() * mesh["ep"].size(),
        )
        device = torch.device(self.device_type, self.rank)
        value = (
            torch.arange(num_experts * 5 * 3, device=device)
            .reshape(num_experts, 5, 3)
            .float()
            .div_(13)
        )
        storage_placements = (Shard(1), Shard(0))
        parameter = torch.nn.Parameter(
            distribute_tensor(value.clone(), mesh, storage_placements)
        )
        fqn = "layers.0.routed_experts.inner_experts.w1_EFD"

        def make_optimizer(param, shard_order_by_tensor_dim):
            return build_dist_muon(
                [{"params": [param], "param_names": [fqn]}],
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.0,
                nesterov=False,
                ns_steps=2,
                compute_sharding_by_fqn={
                    fqn: ComputeLayout(
                        shardings_by_mesh_axis={
                            "efsdp": Shard(0),
                            "ep": Shard(0),
                        },
                        shard_order_by_tensor_dim=shard_order_by_tensor_dim,
                    )
                },
                bucket_configs=[BucketConfig(patterns=(fqn,))],
            )

        expected_shard_order = {0: ("ep", "efsdp")}
        # The storage-mesh order shards over EFSDP first, which loses the exact
        # EP-axis ownership that the redistribution has to preserve.
        for default_order in ({}, {0: ("efsdp", "ep")}):
            with self.assertRaisesRegex(
                ValueError,
                r"must declare shard_order_by_tensor_dim=\{0: \('ep', 'efsdp'\)\}",
            ):
                make_optimizer(parameter, default_order)

        optimizer = make_optimizer(parameter, expected_shard_order)
        grad = (
            torch.arange(value.numel(), device=device)
            .reshape_as(value)
            .float()
            .mul_(0.37)
            .add_(0.2)
            .sin_()
        )
        parameter.grad = distribute_tensor(grad.clone(), mesh, storage_placements)

        mesh_coordinate = mesh.get_coordinate()
        assert mesh_coordinate is not None
        efsdp_coordinate, ep_coordinate = mesh_coordinate
        ep_num_experts, ep_offset = Shard.local_shard_size_and_offset(
            num_experts,
            mesh["ep"].size(),
            ep_coordinate,
        )
        efsdp_num_experts, efsdp_offset = Shard.local_shard_size_and_offset(
            ep_num_experts,
            mesh["efsdp"].size(),
            efsdp_coordinate,
        )
        compute_offset = ep_offset + efsdp_offset
        expected_compute = grad.narrow(
            0,
            compute_offset,
            efsdp_num_experts,
        ).contiguous()
        expected_direction = grad.clone().mul_(0.5).add_(0.25)
        expected_parameter = value.clone().mul_(1 - lr * weight_decay)
        expected_parameter.add_(
            expected_direction,
            alpha=-_adjust_muon_learning_rate(lr, None, value.shape[1:]),
        )
        captured_compute = None

        def capture_compute(_compute_layout, compute):
            nonlocal captured_compute
            captured_compute = compute.clone()
            compute.mul_(0.5).add_(0.25)

        with mock.patch.object(
            optimizer,
            "_compute_update",
            side_effect=capture_compute,
        ):
            optimizer.step()

        if expected_compute.numel():
            self.assertIsNotNone(captured_compute)
            torch.testing.assert_close(
                captured_compute,
                expected_compute,
                rtol=0,
                atol=0,
            )
        else:
            self.assertIsNone(captured_compute)

        torch.testing.assert_close(
            parameter.full_tensor(),
            expected_parameter,
            rtol=0,
            atol=0,
        )
        self.assertEqual(parameter.placements, storage_placements)

        compute_ready_local = value.narrow(
            0,
            compute_offset,
            efsdp_num_experts,
        ).contiguous()
        # The compute-ready storage already carries the declared shard order.
        compute_ready_parameter = torch.nn.Parameter(
            DTensor.from_local(
                compute_ready_local,
                mesh,
                (_StridedShard(0, split_factor=mesh["ep"].size()), Shard(0)),
                shape=value.shape,
                stride=value.stride(),
                run_check=False,
            )
        )
        compute_ready_optimizer = make_optimizer(
            compute_ready_parameter,
            expected_shard_order,
        )
        compute_ready_layout = compute_ready_optimizer._parameter_compute_layouts[0]
        self.assertTrue(compute_ready_layout.storage_is_compute_ready)


if __name__ == "__main__":
    unittest.main()
