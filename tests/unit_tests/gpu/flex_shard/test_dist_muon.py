# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# @lint-ignore-every CITRINE

import copy
import unittest
from unittest import mock

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
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
    DistMuon,
)


pytestmark = pytest.mark.multi_gpu


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
        num_matrices = 3
        matrix_rows = 4
        # Two storage shards each own six rows, so their boundary splits the
        # middle four-row matrix and exercises overshard redistribution.
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
            return build_dist_muon(
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
                        shardings_by_mesh_axis={
                            "dp_shard": BlockShard(
                                dim=0,
                                block_size=matrix_rows,
                            )
                        },
                    ),
                },
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
            torch.arange(12, 48, device=device)
            .reshape(num_matrices * matrix_rows, 3)
            .float()
            .div_(10)
        )
        redistributed = make_parameter(redistributed_value)
        local_blocks = make_parameter(local_blocks_value)
        optimizer = make_optimizer(redistributed, local_blocks)
        self.assertIs(type(optimizer), DistMuon)
        with self.assertRaisesRegex(RuntimeError, "parameter groups are frozen"):
            optimizer.add_param_group({"params": []})

        reference_redistributed = torch.nn.Parameter(redistributed_value.clone())
        reference_local_blocks = tuple(
            torch.nn.Parameter(block.clone())
            for block in local_blocks_value.view(num_matrices, matrix_rows, 3)
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
                local_blocks_grad.view(num_matrices, matrix_rows, 3),
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

            expected_local_blocks = torch.cat(
                tuple(parameter.detach() for parameter in reference_local_blocks)
            ).chunk(self.world_size)[rank]
            expected_local_blocks_before = torch.cat(
                reference_local_blocks_before
            ).chunk(self.world_size)[rank]
            decay = 1 - lr * weight_decay
            adjusted_lr = _adjust_muon_learning_rate(
                lr, None, reference_local_blocks[0].shape
            )
            actual_update = (
                local_blocks_before * decay - current_local_blocks.to_local()
            ) / adjusted_lr
            expected_update = (
                expected_local_blocks_before * decay - expected_local_blocks
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
            torch.arange(13, 49, device=device)
            .reshape(num_matrices * matrix_rows, 3)
            .float()
            .div_(19)
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


@unittest.skipUnless(torch.cuda.device_count() >= 4, "requires four CUDA devices")
class TestDistMuonHSDPOwnedCompute(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @property
    def device_type(self):
        return "cuda"

    @with_comms
    def test_multi_axis_owned_compute_uses_packed_redistribution(self):
        mesh = init_device_mesh(
            self.device_type,
            (2, 2),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        device = torch.device(self.device_type, self.rank)
        shapes = ((4, 3), (1, 4, 3), (4, 3), (1, 4, 3))
        values = tuple(
            torch.arange(12, device=device)
            .reshape(shape)
            .float()
            .div_(10)
            .add_(parameter_index + 1)
            for parameter_index, shape in enumerate(shapes)
        )
        gradients = tuple(
            torch.arange(1, 13, device=device)
            .reshape(shape)
            .float()
            .add_(parameter_index)
            .div_(17)
            for parameter_index, shape in enumerate(shapes)
        )
        placements = (Replicate(), Shard(0))
        fqns = tuple(f"layers.{index}.weight" for index in range(4))

        def run_steps(*, deduplicate_replicas):
            parameters = tuple(
                torch.nn.Parameter(distribute_tensor(value.clone(), mesh, placements))
                for value in values
            )
            owned_mesh_axes = (
                {"dp_replicate": Owned(), "dp_shard": Owned()}
                if deduplicate_replicas
                else {"dp_shard": Owned()}
            )
            optimizer = build_dist_muon(
                [{"params": parameters, "param_names": fqns}],
                compute_sharding_by_fqn={
                    fqn: ComputeLayout(
                        shardings_by_mesh_axis=owned_mesh_axes,
                    )
                    for fqn in fqns
                },
                bucket_configs=[BucketConfig(patterns=("layers.*",))],
                lr=0.03,
                weight_decay=0.2,
                momentum=0.8,
                nesterov=True,
                ns_steps=2,
            )
            num_local_compute_calls = 0
            num_local_all_to_all_calls = 0
            compute_update = optimizer._compute_update
            all_to_all_single = dist.all_to_all_single

            def count_compute(compute_layout, compute):
                nonlocal num_local_compute_calls
                num_local_compute_calls += 1
                compute_update(compute_layout, compute)

            def count_all_to_all(*args, **kwargs):
                nonlocal num_local_all_to_all_calls
                num_local_all_to_all_calls += 1
                return all_to_all_single(*args, **kwargs)

            with (
                mock.patch.object(
                    optimizer,
                    "_compute_update",
                    side_effect=count_compute,
                ),
                mock.patch.object(
                    dist,
                    "all_to_all_single",
                    side_effect=count_all_to_all,
                ),
            ):
                for reverse_rows in (False, True):
                    for parameter, gradient in zip(parameters, gradients, strict=True):
                        step_gradient = (
                            gradient.flip(-2).contiguous() if reverse_rows else gradient
                        )
                        parameter.grad = distribute_tensor(
                            step_gradient.clone(), mesh, placements
                        )
                    optimizer.step()

            num_global_compute_calls = torch.tensor(
                num_local_compute_calls,
                device=device,
                dtype=torch.int64,
            )
            dist.all_reduce(num_global_compute_calls)
            return (
                tuple(
                    parameter.to_local().detach().clone() for parameter in parameters
                ),
                tuple(
                    optimizer.state[parameter]["momentum_buffer"]
                    .to_local()
                    .detach()
                    .clone()
                    for parameter in parameters
                ),
                num_local_compute_calls,
                int(num_global_compute_calls.item()),
                num_local_all_to_all_calls,
                optimizer,
            )

        (
            baseline_params,
            baseline_momenta,
            _,
            baseline_compute_calls,
            _,
            _,
        ) = run_steps(deduplicate_replicas=False)
        (
            deduplicated_params,
            deduplicated_momenta,
            deduplicated_local_compute_calls,
            deduplicated_compute_calls,
            deduplicated_all_to_all_calls,
            deduplicated_optimizer,
        ) = run_steps(deduplicate_replicas=True)

        self.assertEqual(baseline_compute_calls, 16)
        self.assertEqual(deduplicated_compute_calls, 8)
        # Four equal-cost whole-matrix tasks are assigned one per rank.
        self.assertEqual(deduplicated_local_compute_calls, 2)
        self.assertEqual(deduplicated_all_to_all_calls, 4)
        bucket_plan = deduplicated_optimizer._bucket_plans[0]
        self.assertEqual(len(bucket_plan.group.participants), self.world_size)
        self.assertTrue(bucket_plan.storage_to_compute_schedule.has_remote_transfers)
        self.assertTrue(bucket_plan.compute_to_storage_schedule.has_remote_transfers)
        for actual, expected in zip(deduplicated_params, baseline_params, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(
            deduplicated_momenta, baseline_momenta, strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @with_comms
    def test_balances_heterogeneous_whole_matrix_tasks(self):
        mesh = init_device_mesh(
            self.device_type,
            (2, 2),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        device = torch.device(self.device_type, self.rank)
        shapes = ((9, 3), (1, 8, 3), (7, 3), (1, 6, 3), (5, 3), (4, 3))
        placements = (Replicate(), Shard(0))
        parameters = tuple(
            torch.nn.Parameter(
                distribute_tensor(torch.ones(shape, device=device), mesh, placements)
            )
            for shape in shapes
        )
        fqns = tuple(f"layers.{index}.weight" for index in range(len(parameters)))
        optimizer = build_dist_muon(
            [{"params": parameters, "param_names": fqns}],
            compute_sharding_by_fqn={
                fqn: ComputeLayout(
                    shardings_by_mesh_axis={
                        "dp_replicate": Owned(),
                        "dp_shard": Owned(),
                    }
                )
                for fqn in fqns
            },
            bucket_configs=[BucketConfig(patterns=("layers.*",))],
        )
        bucket_plan = optimizer._bucket_plans[0]
        owner_ranks = []
        for redistribution_plan in bucket_plan.redistribution_plans:
            owners = [
                partition.participant
                for partition in redistribution_plan.compute_partitions
                if partition.tensor_shape != (0,)
            ]
            self.assertEqual(len(owners), 1)
            owner_ranks.extend(owners)

        self.assertEqual(set(owner_ranks), set(bucket_plan.group.participants))
        self.assertEqual(
            sorted(owner_ranks.count(rank) for rank in bucket_plan.group.participants),
            [1, 1, 2, 2],
        )


@unittest.skipUnless(torch.cuda.device_count() >= 4, "requires four CUDA devices")
class TestDistMuonHSDPShardCompute(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @property
    def device_type(self):
        return "cuda"

    @with_comms
    def test_replica_sharding_preserves_storage_and_compute_order(self):
        device = torch.device(self.device_type, self.rank)
        for replica_first in (True, False):
            names = (
                ("dp_replicate", "dp_shard")
                if replica_first
                else ("dp_shard", "dp_replicate")
            )
            mesh = init_device_mesh(self.device_type, (2, 2), mesh_dim_names=names)
            placements = tuple(
                Replicate() if name == "dp_replicate" else Shard(0) for name in names
            )
            for num_matrices in (8, 3, 1):
                with self.subTest(replica_first=replica_first, matrices=num_matrices):
                    value = (
                        torch.arange(
                            num_matrices * 5 * 3, device=device, dtype=torch.float32
                        )
                        .reshape(num_matrices, 5, 3)
                        .div_(13)
                    )
                    parameter = torch.nn.Parameter(
                        distribute_tensor(value.clone(), mesh, placements)
                    )

                    def make_optimizer(order, parameter=parameter):
                        return build_dist_muon(
                            [{"params": [parameter], "param_names": ["weight"]}],
                            compute_sharding_by_fqn={
                                "weight": ComputeLayout(
                                    shardings_by_mesh_axis={
                                        "dp_replicate": Shard(0),
                                        "dp_shard": Shard(0),
                                    },
                                    shard_order_by_tensor_dim=order,
                                )
                            },
                            bucket_configs=[BucketConfig(patterns=("weight",))],
                            lr=0.03,
                            weight_decay=0.2,
                            momentum=0.0,
                            nesterov=False,
                        )

                    if replica_first:
                        with self.assertRaisesRegex(
                            ValueError, "must declare shard_order"
                        ):
                            make_optimizer({})
                    optimizer = make_optimizer(
                        {0: ("dp_shard", "dp_replicate")} if replica_first else {}
                    )
                    coordinate = mesh.get_coordinate()
                    shard_coordinate = coordinate[names.index("dp_shard")]
                    replica_coordinate = coordinate[names.index("dp_replicate")]
                    shard_size, shard_offset = Shard.local_shard_size_and_offset(
                        num_matrices, 2, shard_coordinate
                    )
                    compute_size, compute_offset = Shard.local_shard_size_and_offset(
                        shard_size, 2, replica_coordinate
                    )
                    gradient = value.sin()
                    parameter.grad = distribute_tensor(
                        gradient.clone(), mesh, placements
                    )
                    captured = []

                    def capture_compute(_layout, compute, captured=captured):
                        captured.append(compute.clone())
                        compute.mul_(0.5).add_(0.25)

                    with mock.patch.object(
                        optimizer, "_compute_update", side_effect=capture_compute
                    ):
                        optimizer.step()
                    if compute_size:
                        self.assertEqual(len(captured), 1)
                        torch.testing.assert_close(
                            captured[0],
                            gradient.narrow(
                                0, shard_offset + compute_offset, compute_size
                            ),
                            rtol=0,
                            atol=0,
                        )
                    else:
                        self.assertEqual(captured, [])
                    expected = value * (1 - 0.03 * 0.2)
                    expected.add_(
                        gradient * 0.5 + 0.25,
                        alpha=-_adjust_muon_learning_rate(0.03, None, value.shape),
                    )
                    torch.testing.assert_close(
                        parameter.full_tensor(), expected, rtol=0, atol=0
                    )
                    self.assertEqual(parameter.placements, placements)
                    bucket = optimizer._bucket_plans[0]
                    self.assertEqual(
                        set(bucket.group.participants),
                        set(mesh["dp_replicate"].mesh.tolist()),
                    )

    @with_comms
    def test_replica_sharding_matches_muon_across_checkpoint(self):
        device = torch.device(self.device_type, self.rank)
        mesh = init_device_mesh(
            self.device_type, (2, 2), mesh_dim_names=("dp_replicate", "dp_shard")
        )
        placements = (Replicate(), Shard(0))
        value = torch.arange(3 * 5 * 3, device=device).reshape(3, 5, 3).float().div_(31)
        for nesterov in (False, True):
            parameters = [
                torch.nn.Parameter(distribute_tensor(value.clone(), mesh, placements))
                for _ in range(2)
            ]

            def make_optimizer(parameter, shard_replicas, nesterov=nesterov):
                return build_dist_muon(
                    [{"params": [parameter], "param_names": ["weight"]}],
                    compute_sharding_by_fqn={
                        "weight": ComputeLayout(
                            shardings_by_mesh_axis=(
                                {"dp_replicate": Shard(0), "dp_shard": Shard(0)}
                                if shard_replicas
                                else {"dp_shard": Shard(0)}
                            ),
                            shard_order_by_tensor_dim=(
                                {0: ("dp_shard", "dp_replicate")}
                                if shard_replicas
                                else {}
                            ),
                        )
                    },
                    bucket_configs=[BucketConfig(patterns=("weight",))],
                    lr=0.03,
                    weight_decay=0.2,
                    momentum=0.8,
                    nesterov=nesterov,
                    ns_steps=2,
                )

            optimizers = [
                make_optimizer(param, bool(index))
                for index, param in enumerate(parameters)
            ]
            compute_numel = [0, 0]
            for step in range(10):
                gradient = (value * 0.37 + 0.2 + step).sin()
                before = [param.to_local().detach().clone() for param in parameters]
                for index, (param, optimizer) in enumerate(
                    zip(parameters, optimizers, strict=True)
                ):
                    param.grad = distribute_tensor(gradient.clone(), mesh, placements)
                    with mock.patch.object(
                        optimizer, "_compute_update", wraps=optimizer._compute_update
                    ) as compute_update:
                        optimizer.step()
                    if index < 2:
                        compute_numel[index] += sum(
                            call.args[1].numel()
                            for call in compute_update.call_args_list
                        )
                # BF16 batched kernels may differ with batch size. Check updates
                # as well as parameters so large initial values cannot hide error.
                torch.testing.assert_close(
                    parameters[1].to_local() - before[1],
                    parameters[0].to_local() - before[0],
                    rtol=2e-2,
                    atol=2e-2,
                )
                torch.testing.assert_close(
                    parameters[1].to_local(),
                    parameters[0].to_local(),
                    rtol=2e-2,
                    atol=2e-2,
                )
                torch.testing.assert_close(
                    optimizers[1].state[parameters[1]]["momentum_buffer"].to_local(),
                    optimizers[0].state[parameters[0]]["momentum_buffer"].to_local(),
                    rtol=0,
                    atol=0,
                )
                if step > 4:
                    torch.testing.assert_close(
                        parameters[2].to_local(),
                        parameters[1].to_local(),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        optimizers[2]
                        .state[parameters[2]]["momentum_buffer"]
                        .to_local(),
                        optimizers[1]
                        .state[parameters[1]]["momentum_buffer"]
                        .to_local(),
                        rtol=0,
                        atol=0,
                    )
                if step == 4:
                    state = copy.deepcopy(optimizers[1].state_dict())
                    restored_parameter = torch.nn.Parameter(
                        parameters[1].detach().clone()
                    )
                    restored = make_optimizer(restored_parameter, True)
                    restored.load_state_dict(state)
                    parameters.append(restored_parameter)
                    optimizers.append(restored)
            counts = torch.tensor(compute_numel, device=device)
            dist.all_reduce(counts)
            self.assertEqual(counts.tolist(), [20 * value.numel(), 10 * value.numel()])


if __name__ == "__main__":
    unittest.main()
