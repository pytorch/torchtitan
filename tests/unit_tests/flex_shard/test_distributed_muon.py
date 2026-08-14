# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Any, cast

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard
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
    BucketConfig,
    build_distributed_muon,
    ComputeLayout,
    Owned,
)
from torchtitan.distributed.flex_shard._optimizer_reshard_schedule import _TensorRegion
from torchtitan.distributed.flex_shard.distributed_muon import (
    _adjust_muon_learning_rate,
    _build_matrix_batch_redistribution_plan,
    _MatrixBatchView,
    DistributedMuon,
)


def _build_single_parameter_muon(
    parameter, shardings_by_mesh_axis, *, num_stacked_matrices=None, **kwargs
):
    fqn = "layers.0.attention.wq.weight"
    return build_distributed_muon(
        [{"params": [parameter], "param_names": [fqn]}],
        compute_sharding_by_fqn={
            fqn: ComputeLayout(
                shardings_by_mesh_axis=shardings_by_mesh_axis,
            )
        },
        num_stacked_matrices_by_fqn=(
            {} if num_stacked_matrices is None else {fqn: num_stacked_matrices}
        ),
        bucket_configs=[BucketConfig(patterns=(fqn,))],
        **kwargs,
    )


class TestStackedMatrixConfiguration(unittest.TestCase):
    def test_num_stacked_matrices_must_be_positive(self):
        parameter = torch.nn.Parameter(torch.ones(2, 2))
        for num_stacked_matrices in (True, 0, -1, 1.5):
            with self.subTest(num_stacked_matrices=num_stacked_matrices):
                with self.assertRaisesRegex(
                    ValueError,
                    "num_stacked_matrices_by_fqn.*must be a positive integer",
                ):
                    _build_single_parameter_muon(
                        parameter,
                        {"dp_shard": Shard(0)},
                        num_stacked_matrices=cast(Any, num_stacked_matrices),
                    )

    def test_rejects_invalid_configuration_maps(self):
        parameter = torch.nn.Parameter(torch.ones(2, 2))
        fqn = "layers.0.attention.wq.weight"
        params = [{"params": [parameter], "param_names": [fqn]}]
        compute_layout = ComputeLayout(shardings_by_mesh_axis={"dp_shard": Shard(0)})

        with self.assertRaisesRegex(
            ValueError,
            "compute_sharding_by_fqn values must be ComputeLayout",
        ):
            build_distributed_muon(
                params,
                compute_sharding_by_fqn={fqn: cast(Any, object())},
                num_stacked_matrices_by_fqn={},
                bucket_configs=[],
            )

        with self.assertRaisesRegex(
            ValueError,
            "num_stacked_matrices_by_fqn.*must be a positive integer",
        ):
            build_distributed_muon(
                params,
                compute_sharding_by_fqn={fqn: compute_layout},
                num_stacked_matrices_by_fqn={fqn: cast(Any, object())},
                bucket_configs=[],
            )

        with self.assertRaisesRegex(
            ValueError,
            "num_stacked_matrices_by_fqn entries must also appear in "
            "compute_sharding_by_fqn",
        ):
            build_distributed_muon(
                params,
                compute_sharding_by_fqn={},
                num_stacked_matrices_by_fqn={fqn: 2},
                bucket_configs=[],
            )

    def test_compute_input_view_is_zero_copy(self):
        matrix_rows = 3
        matrix_columns = 2
        compute_view = _MatrixBatchView(
            matrix_rows=matrix_rows,
            matrix_columns=matrix_columns,
        )

        for num_matrices in (0, 2):
            with self.subTest(num_matrices=num_matrices):
                compute_input = torch.arange(
                    num_matrices * matrix_rows * matrix_columns,
                    dtype=torch.float32,
                ).reshape(num_matrices * matrix_rows, matrix_columns)
                compute = compute_view.view_compute_input(compute_input)

                self.assertEqual(
                    compute.shape,
                    (num_matrices, matrix_rows, matrix_columns),
                )
                self.assertEqual(
                    compute.stride(),
                    (matrix_rows * matrix_columns, matrix_columns, 1),
                )
                self.assertEqual(compute.data_ptr(), compute_input.data_ptr())
                if compute.numel():
                    compute[0, 0, 0] = -1
                    self.assertEqual(compute_input[0, 0].item(), -1)

    def test_redistribution_destinations_are_flat_compute_inputs(self):
        plan = _build_matrix_batch_redistribution_plan(
            (
                ((0,), _TensorRegion(offsets=(0, 0), shape=(6, 2))),
                ((1,), _TensorRegion(offsets=(6, 0), shape=(6, 2))),
            ),
            participants=(0, 1),
            storage_shape=(12, 2),
            compute_shape=(4, 3, 2),
        )

        self.assertEqual(
            tuple(partition.tensor_shape for partition in plan.compute_partitions),
            ((6, 2), (6, 2)),
        )
        self.assertTrue(
            all(
                len(route.destination.tensor_region.shape) == 2
                for route in plan.storage_to_compute_routes
            )
        )


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestDistributedMuon(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @property
    def device_type(self):
        return "cuda"

    @with_comms
    def test_aligned_storage_uses_flat_compute_input_and_local_view(self):
        num_heads = 4
        matrix_rows = 3
        matrix_columns = 2
        mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("dp_shard",),
        )
        value = torch.arange(
            num_heads * matrix_rows * matrix_columns,
            device=torch.device(self.device_type, self.rank),
            dtype=torch.float32,
        ).reshape(num_heads * matrix_rows, matrix_columns)
        parameter = torch.nn.Parameter(distribute_tensor(value, mesh, (Shard(0),)))

        optimizer = _build_single_parameter_muon(
            parameter,
            {"dp_shard": Shard(0)},
            num_stacked_matrices=num_heads,
        )

        compute_layout = optimizer._parameter_compute_layouts[0]
        self.assertTrue(compute_layout.storage_is_compute_ready)
        local_compute_input = compute_layout.local_compute_input
        self.assertIsNotNone(local_compute_input)
        self.assertEqual(
            local_compute_input.shape,
            (num_heads // self.world_size * matrix_rows, matrix_columns),
        )
        self.assertEqual(
            local_compute_input.data_ptr(),
            parameter.to_local().data_ptr(),
        )
        compute_view = compute_layout.compute_view
        self.assertIsNotNone(compute_view)
        compute = compute_view.view_compute_input(local_compute_input)
        self.assertEqual(
            compute.shape,
            (num_heads // self.world_size, matrix_rows, matrix_columns),
        )
        self.assertEqual(compute.data_ptr(), local_compute_input.data_ptr())

    @with_comms
    def test_rejects_oversharded_matrix_batch_storage(self):
        num_heads = 3
        matrix_rows = 4
        matrix_columns = 2
        mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("dp_shard",),
        )
        value = torch.arange(
            num_heads * matrix_rows * matrix_columns,
            device=torch.device(self.device_type, self.rank),
            dtype=torch.float32,
        ).reshape(num_heads * matrix_rows, matrix_columns)
        parameter = torch.nn.Parameter(distribute_tensor(value, mesh, (Shard(0),)))

        with self.assertRaisesRegex(
            ValueError,
            "matrix-batch storage shards are not aligned to matrix rows of size 4",
        ):
            _build_single_parameter_muon(
                parameter,
                {"dp_shard": Shard(0)},
                num_stacked_matrices=num_heads,
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

        def make_parameter(value: torch.Tensor) -> torch.nn.Parameter:
            return torch.nn.Parameter(
                distribute_tensor(value.clone(), mesh, (Shard(0),))
            )

        def make_optimizer(
            redistributed: torch.nn.Parameter,
            local_blocks: torch.nn.Parameter,
            ns_steps: int = 2,
        ):
            redistributed_fqn = "layers.0.redistributed"
            local_blocks_fqn = "layers.0.local_blocks"
            return build_distributed_muon(
                [
                    {
                        "params": [redistributed, local_blocks],
                        "param_names": [redistributed_fqn, local_blocks_fqn],
                    }
                ],
                compute_sharding_by_fqn={
                    redistributed_fqn: ComputeLayout(
                        shardings_by_mesh_axis={
                            "dp_shard": Owned(),
                        },
                    ),
                    local_blocks_fqn: ComputeLayout(
                        shardings_by_mesh_axis={"dp_shard": Shard(0)},
                    ),
                },
                num_stacked_matrices_by_fqn={local_blocks_fqn: self.world_size},
                bucket_configs=[
                    BucketConfig(
                        patterns=("layers.0.*",),
                        name="layers.0",
                    )
                ],
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.8,
                nesterov=True,
                ns_steps=ns_steps,
            )

        redistributed_value = (
            torch.arange(12, device=device).reshape(4, 3).float().div_(10).add_(1)
        )
        local_blocks_value = (
            torch.arange(12, 24, device=device).reshape(4, 3).float().div_(10)
        )
        redistributed = make_parameter(redistributed_value)
        local_blocks = make_parameter(local_blocks_value)
        optimizer = make_optimizer(redistributed, local_blocks)
        self.assertIs(type(optimizer), DistributedMuon)
        with self.assertRaisesRegex(RuntimeError, "parameter groups are frozen"):
            optimizer.add_param_group({"params": []})

        reference_redistributed = torch.nn.Parameter(redistributed_value.clone())
        reference_local_blocks = tuple(
            torch.nn.Parameter(block.clone())
            for block in local_blocks_value.chunk(self.world_size, dim=0)
        )
        reference_optimizer = torch.optim.Muon(
            [reference_redistributed, *reference_local_blocks],
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
        )

        def step_and_assert(
            current_optimizer,
            current_redistributed: torch.nn.Parameter,
            current_local_blocks: torch.nn.Parameter,
            redistributed_grad: torch.Tensor,
            local_blocks_grad: torch.Tensor,
        ) -> None:
            local_blocks_before = current_local_blocks.to_local().clone()
            reference_local_blocks_before = tuple(
                parameter.detach().clone() for parameter in reference_local_blocks
            )
            current_redistributed.grad = distribute_tensor(
                redistributed_grad.clone(), mesh, (Shard(0),)
            )
            current_local_blocks.grad = distribute_tensor(
                local_blocks_grad.clone(), mesh, (Shard(0),)
            )
            reference_redistributed.grad = redistributed_grad.clone()
            for parameter, grad in zip(
                reference_local_blocks,
                local_blocks_grad.chunk(self.world_size, dim=0),
                strict=True,
            ):
                parameter.grad = grad.clone()

            current_optimizer.step()
            reference_optimizer.step()

            rank = mesh.get_local_rank()
            expected_redistributed = reference_redistributed.detach().chunk(
                self.world_size, dim=0
            )[rank]
            torch.testing.assert_close(
                current_redistributed.to_local(),
                expected_redistributed,
                rtol=0,
                atol=0,
            )

            expected_local_blocks = reference_local_blocks[rank].detach()
            decay = 1 - lr * weight_decay
            adjusted_lr = _adjust_muon_learning_rate(
                lr, None, expected_local_blocks.shape
            )
            actual_update = (
                local_blocks_before * decay - current_local_blocks.to_local()
            ) / adjusted_lr
            expected_update = (
                reference_local_blocks_before[rank] * decay - expected_local_blocks
            ) / adjusted_lr
            # Batched BF16 Newton-Schulz can differ slightly across GEMM schedules.
            torch.testing.assert_close(
                actual_update,
                expected_update,
                rtol=0,
                atol=2e-2,
            )

        first_redistributed_grad = (
            torch.arange(1, 13, device=device).reshape(4, 3).float().div_(17)
        )
        first_local_blocks_grad = (
            torch.arange(13, 25, device=device).reshape(4, 3).float().div_(19)
        )
        step_and_assert(
            optimizer,
            redistributed,
            local_blocks,
            first_redistributed_grad,
            first_local_blocks_grad,
        )

        flat_state_dict = get_flat_optim_state_dict(optimizer)
        resumed_redistributed = make_parameter(redistributed.full_tensor().detach())
        resumed_local_blocks = make_parameter(local_blocks.full_tensor().detach())
        resumed_optimizer = make_optimizer(
            resumed_redistributed,
            resumed_local_blocks,
            ns_steps=3,
        )
        init_optim_state(resumed_optimizer)
        load_flat_optim_state_dict(resumed_optimizer, flat_state_dict)

        second_redistributed_grad = first_redistributed_grad.flip(0).contiguous()
        second_local_blocks_grad = first_local_blocks_grad.flip(0).contiguous()
        step_and_assert(
            resumed_optimizer,
            resumed_redistributed,
            resumed_local_blocks,
            second_redistributed_grad,
            second_local_blocks_grad,
        )


@unittest.skipUnless(torch.cuda.device_count() >= 4, "requires four CUDA devices")
class TestDistributedMuonInitialExpertStorageContract(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @property
    def device_type(self):
        return "cuda"

    @with_comms
    def test_rejects_insufficient_expert_storage_layout(self):
        mesh = init_device_mesh(
            self.device_type,
            (2, 2),
            mesh_dim_names=("efsdp", "ep"),
        )
        num_experts = 2
        self.assertGreater(mesh["efsdp"].size() * mesh["ep"].size(), num_experts)
        device = torch.device(self.device_type, self.rank)
        value = torch.arange(
            num_experts * 16,
            device=device,
            dtype=torch.float32,
        ).reshape(num_experts, 4, 4)
        parameter = torch.nn.Parameter(
            distribute_tensor(value, mesh, (Shard(1), Shard(0)))
        )
        fqn = "layers.0.routed_experts.inner_experts.w1_EFD"

        with self.assertRaisesRegex(
            NotImplementedError,
            "cannot redistribute storage on mesh axis 'efsdp'.*"
            "preserving Shard\\(0\\) storage on mesh axis 'ep'.*"
            "orthogonal-shard redistribution is not implemented",
        ):
            build_distributed_muon(
                [{"params": [parameter], "param_names": [fqn]}],
                compute_sharding_by_fqn={
                    fqn: ComputeLayout(
                        shardings_by_mesh_axis={
                            "efsdp": Shard(0),
                            "ep": Shard(0),
                        },
                    )
                },
                num_stacked_matrices_by_fqn={},
                bucket_configs=[BucketConfig(patterns=(fqn,))],
            )


if __name__ == "__main__":
    unittest.main()
