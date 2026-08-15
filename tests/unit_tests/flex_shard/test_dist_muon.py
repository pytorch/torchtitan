# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.components.checkpoint_utils import (
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
    def test_explicit_replicated_compute(self):
        lr = 0.03
        weight_decay = 0.2
        mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("dp_shard",),
        )
        device = torch.device(self.device_type, self.rank)
        value = torch.arange(12, device=device).reshape(4, 3).float().div_(10)
        parameter = torch.nn.Parameter(
            distribute_tensor(value.clone(), mesh, (Replicate(),))
        )
        fqn = "layers.0.replicated"
        optimizer = build_dist_muon(
            [{"params": [parameter], "param_names": [fqn]}],
            compute_sharding_by_fqn={
                fqn: ComputeLayout(
                    shardings_by_mesh_axis={"dp_shard": Replicate()},
                )
            },
            bucket_configs=[BucketConfig(patterns=(fqn,))],
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
        )

        reference = torch.nn.Parameter(value.clone())
        reference_optimizer = torch.optim.Muon(
            [reference],
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
        )
        grad = torch.arange(1, 13, device=device).reshape(4, 3).float().div_(17)
        parameter.grad = distribute_tensor(grad.clone(), mesh, (Replicate(),))
        reference.grad = grad.clone()

        optimizer.step()
        reference_optimizer.step()

        torch.testing.assert_close(
            parameter.to_local(),
            reference,
            rtol=0,
            atol=0,
        )

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
            # These aligned siblings remain local and share one batched NS call.
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


@unittest.skipUnless(torch.cuda.device_count() >= 4, "requires four CUDA devices")
class TestDistMuonMultiMesh(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @property
    def device_type(self):
        return "cuda"

    @with_comms
    def test_matrix_batch_uses_one_active_mesh_axis(self):
        mesh = init_device_mesh(
            self.device_type,
            (2, 2),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        device = torch.device(self.device_type, self.rank)
        value = torch.arange(36, device=device, dtype=torch.float32).reshape(12, 3)
        fqn = "layers.0.attention.wq.weight"

        def make_optimizer(
            placements: tuple[Replicate | Shard, Replicate | Shard],
            compute_layout: ComputeLayout,
        ) -> DistMuon:
            parameter = torch.nn.Parameter(
                distribute_tensor(value.clone(), mesh, placements)
            )
            return build_dist_muon(
                [{"params": [parameter], "param_names": [fqn]}],
                compute_sharding_by_fqn={fqn: compute_layout},
                bucket_configs=[BucketConfig(patterns=(fqn,))],
            )

        optimizer = make_optimizer(
            (Replicate(), Shard(0)),
            ComputeLayout(
                shardings_by_mesh_axis={
                    "dp_shard": BlockShard(dim=0, block_size=4),
                }
            ),
        )
        self.assertIs(type(optimizer), DistMuon)
        compute_layout = optimizer._parameter_compute_layouts[0]
        self.assertFalse(compute_layout.storage_is_compute_ready)
        self.assertEqual(compute_layout.redistribution_storage_mesh_axis, 1)

        with self.assertRaisesRegex(
            NotImplementedError,
            "multiple active mesh axes.*only one active BlockShard axis",
        ):
            make_optimizer(
                (Shard(0), Shard(0)),
                ComputeLayout(
                    shardings_by_mesh_axis={
                        "dp_replicate": BlockShard(dim=0, block_size=4),
                        "dp_shard": BlockShard(dim=0, block_size=4),
                    }
                ),
            )

    @with_comms
    def test_preserves_ep_shard_in_mixed_logical_bucket(self):
        lr = 0.03
        weight_decay = 0.2
        dense_mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("dp_shard",),
        )
        sparse_mesh = init_device_mesh(
            self.device_type,
            (2, 2),
            mesh_dim_names=("efsdp", "ep"),
        )
        repeated_shard_mesh = init_device_mesh(
            self.device_type,
            (2, 2),
            mesh_dim_names=("ep", "efsdp"),
        )
        device = torch.device(self.device_type, self.rank)

        dense_value = (
            torch.arange(24, device=device).reshape(8, 3).float().div_(11).add_(1)
        )
        sparse_values = {
            "layers.0.routed_experts.sharded": (
                torch.arange(60, device=device)
                .reshape(4, 5, 3)
                .float()
                .div_(13)
                .add_(2)
            ),
            "layers.0.routed_experts.replicated": (
                torch.arange(60, 120, device=device)
                .reshape(4, 5, 3)
                .float()
                .div_(17)
                .add_(3)
            ),
            "layers.0.routed_experts.repeated_shard": (
                torch.arange(120, 180, device=device)
                .reshape(4, 5, 3)
                .float()
                .div_(19)
                .add_(4)
            ),
        }
        sparse_storage_layouts = {
            "layers.0.routed_experts.sharded": (
                sparse_mesh,
                (Shard(1), Shard(0)),
            ),
            "layers.0.routed_experts.replicated": (
                sparse_mesh,
                (Shard(1), Shard(0)),
            ),
            "layers.0.routed_experts.repeated_shard": (
                repeated_shard_mesh,
                (Shard(0), Shard(0)),
            ),
        }
        dense = torch.nn.Parameter(
            distribute_tensor(dense_value.clone(), dense_mesh, (Shard(0),))
        )
        sparse = {
            fqn: torch.nn.Parameter(
                distribute_tensor(
                    value.clone(),
                    *sparse_storage_layouts[fqn],
                )
            )
            for fqn, value in sparse_values.items()
        }
        dense_fqn = "layers.0.dense"
        optimizer = build_dist_muon(
            [
                {
                    "params": [dense, *sparse.values()],
                    "param_names": [dense_fqn, *sparse],
                }
            ],
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
            compute_sharding_by_fqn={
                dense_fqn: ComputeLayout(
                    shardings_by_mesh_axis={
                        "dp_shard": Owned(),
                    },
                ),
                "layers.0.routed_experts.sharded": ComputeLayout(
                    shardings_by_mesh_axis={"efsdp": Shard(0)},
                ),
                "layers.0.routed_experts.replicated": ComputeLayout(
                    shardings_by_mesh_axis={
                        "efsdp": Replicate(),
                        "ep": Shard(0),
                    },
                ),
                "layers.0.routed_experts.repeated_shard": ComputeLayout(
                    shardings_by_mesh_axis={"efsdp": Replicate()},
                ),
            },
            bucket_configs=[BucketConfig(patterns=("layers.0.*",), name="layers.0")],
        )
        transport_groups = {
            frozenset(plan.group.participants)
            for plan in optimizer._bucket_plans
            if hasattr(plan, "group")
        }
        self.assertEqual(
            transport_groups,
            {
                frozenset(range(self.world_size)),
                frozenset(sparse_mesh["efsdp"].mesh.flatten().tolist()),
                frozenset(repeated_shard_mesh["efsdp"].mesh.flatten().tolist()),
            },
        )

        dense_grad = torch.arange(1, 25, device=device).reshape(8, 3).float().div_(19)
        sparse_grads = {
            fqn: torch.arange(60, device=device)
            .reshape_as(value)
            .float()
            .mul_(0.37 + 0.11 * index)
            .add_(0.2 + index)
            .sin_()
            for index, (fqn, value) in enumerate(sparse_values.items())
        }
        dense.grad = distribute_tensor(dense_grad.clone(), dense_mesh, (Shard(0),))
        for fqn, parameter in sparse.items():
            storage_mesh, placements = sparse_storage_layouts[fqn]
            parameter.grad = distribute_tensor(
                sparse_grads[fqn].clone(),
                storage_mesh,
                placements,
            )

        reference_dense = torch.nn.Parameter(dense_value.clone())
        reference_sparse = {
            fqn: tuple(torch.nn.Parameter(matrix.clone()) for matrix in value)
            for fqn, value in sparse_values.items()
        }
        reference_optimizer = torch.optim.Muon(
            [
                reference_dense,
                *(
                    parameter
                    for matrices in reference_sparse.values()
                    for parameter in matrices
                ),
            ],
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
        )
        reference_dense.grad = dense_grad.clone()
        for fqn, matrices in reference_sparse.items():
            for parameter, grad in zip(
                matrices,
                sparse_grads[fqn],
                strict=True,
            ):
                parameter.grad = grad.clone()

        optimizer.step()
        reference_optimizer.step()

        decay = 1 - lr * weight_decay
        dense_adjusted_lr = _adjust_muon_learning_rate(lr, None, dense_value.shape)
        torch.testing.assert_close(
            (dense_value * decay - dense.full_tensor()) / dense_adjusted_lr,
            (dense_value * decay - reference_dense) / dense_adjusted_lr,
            rtol=0,
            atol=2e-2,
        )
        for fqn, parameter in sparse.items():
            expected = torch.stack(
                [reference.detach() for reference in reference_sparse[fqn]]
            )
            adjusted_lr = _adjust_muon_learning_rate(
                lr,
                None,
                reference_sparse[fqn][0].shape,
            )
            actual_update = (
                sparse_values[fqn] * decay - parameter.full_tensor()
            ) / adjusted_lr
            expected_update = (sparse_values[fqn] * decay - expected) / adjusted_lr
            torch.testing.assert_close(
                actual_update,
                expected_update,
                rtol=0,
                atol=2e-2,
            )
            self.assertEqual(parameter.placements, sparse_storage_layouts[fqn][1])

        unsupported_parameter = torch.nn.Parameter(
            distribute_tensor(
                sparse_values["layers.0.routed_experts.sharded"].clone(),
                sparse_mesh,
                (Shard(1), Shard(0)),
            )
        )
        unsupported_fqn = "layers.0.routed_experts.unsupported"
        with self.assertRaisesRegex(NotImplementedError, "multiple mesh axes"):
            build_dist_muon(
                [
                    {
                        "params": [unsupported_parameter],
                        "param_names": [unsupported_fqn],
                    }
                ],
                compute_sharding_by_fqn={
                    unsupported_fqn: ComputeLayout(
                        shardings_by_mesh_axis={
                            "efsdp": Replicate(),
                            "ep": Replicate(),
                        },
                    )
                },
                bucket_configs=[BucketConfig(patterns=(unsupported_fqn,))],
            )


if __name__ == "__main__":
    unittest.main()
