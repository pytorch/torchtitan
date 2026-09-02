# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# @lint-ignore-every CITRINE

import unittest
from unittest import mock

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
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
class TestDistMuonHSDPReplicaDeduplication(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @property
    def device_type(self):
        return "cuda"

    @with_comms
    def test_deduplicates_compute_and_preserves_momentum(self):
        mesh = init_device_mesh(
            self.device_type,
            (2, 2),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        device = torch.device(self.device_type, self.rank)
        value = torch.arange(12, device=device).reshape(4, 3).float().div_(10).add_(1)
        gradient = torch.arange(1, 13, device=device).reshape(4, 3).float().div_(17)
        placements = (Replicate(), Shard(0))
        fqn = "layers.0.weight"

        def run_steps(deduplicate_compute_mesh_axis):
            parameter = torch.nn.Parameter(
                distribute_tensor(value.clone(), mesh, placements)
            )
            optimizer = build_dist_muon(
                [{"params": [parameter], "param_names": [fqn]}],
                compute_sharding_by_fqn={
                    fqn: ComputeLayout(
                        shardings_by_mesh_axis={"dp_shard": Owned()},
                    )
                },
                bucket_configs=[BucketConfig(patterns=(fqn,))],
                lr=0.03,
                weight_decay=0.2,
                momentum=0.8,
                nesterov=True,
                ns_steps=2,
                deduplicate_compute_mesh_axis=deduplicate_compute_mesh_axis,
            )
            num_local_compute_calls = 0
            compute_update = optimizer._compute_update

            def count_compute(compute_layout, compute):
                nonlocal num_local_compute_calls
                num_local_compute_calls += 1
                compute_update(compute_layout, compute)

            with mock.patch.object(
                optimizer,
                "_compute_update",
                side_effect=count_compute,
            ):
                for step_gradient in (gradient, gradient.flip(0).contiguous()):
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
            momentum = optimizer.state[parameter]["momentum_buffer"]
            return (
                parameter.to_local().detach().clone(),
                momentum.to_local().detach().clone(),
                num_local_compute_calls,
                int(num_global_compute_calls.item()),
            )

        baseline_param, baseline_momentum, _, baseline_compute_calls = run_steps(None)
        (
            deduplicated_param,
            deduplicated_momentum,
            deduplicated_local_compute_calls,
            deduplicated_compute_calls,
        ) = run_steps("dp_replicate")

        self.assertEqual(baseline_compute_calls, 4)
        self.assertEqual(deduplicated_compute_calls, 2)
        mesh_coordinate = mesh.get_coordinate()
        assert mesh_coordinate is not None
        if mesh_coordinate[0] != 0:
            self.assertEqual(deduplicated_local_compute_calls, 0)
        torch.testing.assert_close(
            deduplicated_param,
            baseline_param,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            deduplicated_momentum,
            baseline_momentum,
            rtol=0,
            atol=0,
        )

    @with_comms
    def test_selects_replica_owner_by_mesh_coordinate(self):
        mesh = init_device_mesh(
            self.device_type,
            (2, 2),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        device = torch.device(self.device_type, self.rank)
        parameter = torch.nn.Parameter(
            distribute_tensor(
                torch.ones(4, 3, device=device),
                mesh,
                (Replicate(), Shard(0)),
            )
        )
        fqn = "layers.0.weight"

        with mock.patch.object(
            DeviceMesh,
            "get_local_rank",
            side_effect=AssertionError(
                "replica ownership must not depend on process-group rank order"
            ),
        ):
            optimizer = build_dist_muon(
                [{"params": [parameter], "param_names": [fqn]}],
                compute_sharding_by_fqn={
                    fqn: ComputeLayout(
                        shardings_by_mesh_axis={"dp_shard": Owned()},
                    )
                },
                bucket_configs=[BucketConfig(patterns=(fqn,))],
                deduplicate_compute_mesh_axis="dp_replicate",
            )

        replica_deduplication = optimizer._replica_compute_deduplication
        assert replica_deduplication is not None
        mesh_coordinate = mesh.get_coordinate()
        assert mesh_coordinate is not None
        self.assertEqual(
            replica_deduplication.is_owner,
            mesh_coordinate[0] == 0,
        )
        (broadcast_group,) = replica_deduplication.parameter_broadcast_groups
        expected_source_rank = int(mesh.mesh[0, mesh_coordinate[1]].item())
        self.assertEqual(broadcast_group.source_rank, expected_source_rank)

    @with_comms
    def test_waits_before_switching_replica_process_groups(self):
        dense_mesh = init_device_mesh(
            self.device_type,
            (2, 2),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        # Keep the same replica owners while forming distinct replica groups.
        sparse_mesh = DeviceMesh(
            self.device_type,
            torch.tensor(((1, 0), (2, 3))),
            mesh_dim_names=("dp_replicate", "efsdp"),
        )
        device = torch.device(self.device_type, self.rank)
        placements = (Replicate(), Shard(0))
        dense_parameter = torch.nn.Parameter(
            distribute_tensor(
                torch.ones(4, 3, device=device),
                dense_mesh,
                placements,
            )
        )
        sparse_parameter = torch.nn.Parameter(
            distribute_tensor(
                torch.ones(4, 3, device=device),
                sparse_mesh,
                placements,
            )
        )
        dense_fqn = "layers.0.attention.weight"
        sparse_fqn = "layers.0.experts.weight"
        optimizer = build_dist_muon(
            [
                {
                    "params": [dense_parameter, sparse_parameter],
                    "param_names": [dense_fqn, sparse_fqn],
                }
            ],
            compute_sharding_by_fqn={
                dense_fqn: ComputeLayout(
                    shardings_by_mesh_axis={"dp_shard": Owned()},
                ),
                sparse_fqn: ComputeLayout(
                    shardings_by_mesh_axis={"efsdp": Owned()},
                ),
            },
            bucket_configs=[
                BucketConfig(patterns=(dense_fqn,)),
                BucketConfig(patterns=(sparse_fqn,)),
            ],
            deduplicate_compute_mesh_axis="dp_replicate",
        )
        replica_deduplication = optimizer._replica_compute_deduplication
        assert replica_deduplication is not None
        broadcast_groups = replica_deduplication.parameter_broadcast_groups
        self.assertEqual(len(broadcast_groups), 2)

        events: list[tuple[str, int]] = []

        class RecordingWork:
            def __init__(self, process_group: dist.ProcessGroup):
                self.process_group = process_group

            def wait(self):
                events.append(("wait", id(self.process_group)))

        def record_broadcast(tensor, *, src, group, async_op):
            self.assertTrue(tensor.numel())
            self.assertIsInstance(src, int)
            self.assertTrue(async_op)
            events.append(("broadcast", id(group)))
            return RecordingWork(group)

        with mock.patch.object(dist, "broadcast", side_effect=record_broadcast):
            optimizer._broadcast_replica_parameters(replica_deduplication)

        expected_events = []
        for broadcast_group in broadcast_groups:
            process_group_id = id(broadcast_group.process_group)
            expected_events.extend(
                ("broadcast", process_group_id) for _ in broadcast_group.params
            )
            expected_events.extend(
                ("wait", process_group_id) for _ in broadcast_group.params
            )
        self.assertEqual(events, expected_events)


if __name__ == "__main__":
    unittest.main()
