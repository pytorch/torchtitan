# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.components.distributed_optimizers.bucketed_redistribution import (
    BucketConfig,
    BucketSpec,
)
from torchtitan.components.distributed_optimizers.muon import (
    _has_dim0_sharded_storage,
    _has_replicated_storage,
    DistributedMuon,
    Owned,
)
from torchtitan.components.checkpoint_utils import (
    get_flat_optim_state_dict,
    init_optim_state,
)
from torchtitan.components.distributed_optimizers.muon_parameter_prep import (
    build_distributed_muon,
    BatchedMatrixComputeView,
    MuonComputeSharding,
)


class TestDistributedMuonStoragePolicy(unittest.TestCase):
    def test_rejects_placement_subclasses(self):
        class UnsupportedShard(Shard):
            pass

        class UnsupportedReplicate(Replicate):
            pass

        class FakeParameter:
            ndim = 3

        parameter = FakeParameter()
        parameter.placements = (UnsupportedShard(0),)
        self.assertFalse(_has_dim0_sharded_storage(parameter))
        parameter.placements = (UnsupportedReplicate(),)
        self.assertFalse(_has_replicated_storage(parameter))

class _DistributedMuonTestBase(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @property
    def device_type(self):
        return "cuda"

    @property
    def mesh(self):
        if not hasattr(self, "_mesh"):
            self._mesh = init_device_mesh(
                self.device_type,
                (self.world_size,),
                mesh_dim_names=("dp_shard",),
            )
        return self._mesh

    @property
    def redistribution_mesh(self):
        if not hasattr(self, "_redistribution_mesh"):
            self._redistribution_mesh = self.mesh._flatten("optimizer")
        return self._redistribution_mesh

    @property
    def device(self):
        return torch.device("cuda", self.rank)

    def _parameter(self, value: torch.Tensor) -> torch.nn.Parameter:
        return torch.nn.Parameter(
            distribute_tensor(value.clone(), self.mesh, (Shard(0),))
        )

    def _optimizer(
        self,
        redistributed: torch.nn.Parameter,
        local_blocks: torch.nn.Parameter,
    ) -> DistributedMuon:
        return build_distributed_muon(
            [
                {
                    "params": [redistributed],
                    "param_names": ["layers.0.redistributed"],
                    "compute_sharding": MuonComputeSharding(placement=Owned()),
                },
                {
                    "params": [local_blocks],
                    "param_names": ["layers.0.local_blocks"],
                    "compute_sharding": MuonComputeSharding(
                        view_before_placement=BatchedMatrixComputeView(
                            num_matrices=2, matrices_flattened_into_dim=0
                        ),
                        placement=Shard(0),
                    ),
                },
            ],
            bucket_configs=[
                BucketConfig(
                    patterns=("layers.0.*",),
                    owner_rank_by_fqn={"layers.0.redistributed": 1},
                    mesh_axis="dp_shard",
                    name="layers.0",
                )
            ],
            lr=0.03,
            weight_decay=0.2,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
        )

    def _set_grads(
        self,
        redistributed: torch.nn.Parameter,
        local_blocks: torch.nn.Parameter,
        redistributed_grad: torch.Tensor,
        local_blocks_grad: torch.Tensor,
    ) -> None:
        redistributed.grad = distribute_tensor(
            redistributed_grad.clone(), self.mesh, (Shard(0),)
        )
        local_blocks.grad = distribute_tensor(
            local_blocks_grad.clone(), self.mesh, (Shard(0),)
        )

    def _assert_matches_reference(
        self,
        optimizer: DistributedMuon,
        redistributed: torch.nn.Parameter,
        local_blocks: torch.nn.Parameter,
        reference_optimizer: torch.optim.Muon,
        reference_redistributed: torch.nn.Parameter,
        reference_local_blocks: tuple[
            torch.nn.Parameter, torch.nn.Parameter
        ],
    ) -> None:
        rank = self.mesh.get_local_rank()
        expected_redistributed = reference_redistributed.detach().chunk(
            self.world_size, dim=0
        )[rank]
        expected_local_blocks = reference_local_blocks[rank].detach()
        torch.testing.assert_close(redistributed.to_local(), expected_redistributed)
        torch.testing.assert_close(local_blocks.to_local(), expected_local_blocks)

        for param in (redistributed, local_blocks):
            self.assertIsInstance(param, DTensor)
            self.assertEqual(param.placements, (Shard(0),))

        redistributed_momentum = optimizer.state[redistributed]["momentum_buffer"]
        self.assertIsInstance(redistributed_momentum, DTensor)
        self.assertEqual(redistributed_momentum.placements, (Shard(0),))
        expected_redistributed_momentum = reference_optimizer.state[
            reference_redistributed
        ]["momentum_buffer"].detach().chunk(self.world_size, dim=0)[rank]
        torch.testing.assert_close(
            redistributed_momentum.to_local(), expected_redistributed_momentum
        )

        local_blocks_momentum = optimizer.state[local_blocks]["momentum_buffer"]
        self.assertIsInstance(local_blocks_momentum, DTensor)
        self.assertEqual(local_blocks_momentum.placements, (Shard(0),))
        expected_local_blocks_momentum = reference_optimizer.state[
            reference_local_blocks[rank]
        ]["momentum_buffer"].detach()
        torch.testing.assert_close(
            local_blocks_momentum.to_local(), expected_local_blocks_momentum
        )


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestDistributedMuon(_DistributedMuonTestBase):
    @with_comms
    def test_constructor_strictly_validates_strided_storage_shards(self):
        mesh = init_device_mesh(self.device_type, (self.world_size, 1))

        def make_parameter(value, dim):
            placements = (
                _StridedShard(dim, split_factor=self.world_size),
                Shard(dim),
            )
            parameter = torch.nn.Parameter(
                distribute_tensor(value, mesh, placements)
            )
            self.assertEqual(parameter.placements, placements)
            return parameter

        def build(parameter, name, compute_placement, owner_rank=None):
            fqn = f"layers.0.{name}"
            owners = {} if owner_rank is None else {fqn: owner_rank}
            return build_distributed_muon(
                [
                    {
                        "params": [parameter],
                        "param_names": [fqn],
                        "compute_sharding": MuonComputeSharding(
                            placement=compute_placement
                        ),
                    }
                ],
                bucket_spec=[
                    BucketSpec(
                        patterns=("layers.0.*",),
                        owner_rank_by_fqn=owners,
                        mesh=self.mesh,
                    )
                ],
            )

        local_blocks = make_parameter(
            torch.arange(24, device=self.device).reshape(4, 2, 3).float(), 0
        )
        optimizer = build(local_blocks, "local_blocks", Shard(0))
        self.assertIs(
            optimizer._plans[0].local_items[0].param, local_blocks
        )
        local_blocks.grad = distribute_tensor(
            torch.ones(4, 2, 3, device=self.device),
            mesh,
            local_blocks.placements,
        )
        with patch(
            "torchtitan.components.distributed_optimizers.bucketed_redistribution.dist."
            "all_to_all_single"
        ) as collective:
            optimizer.step()
        collective.assert_not_called()
        self.assertEqual(
            optimizer.state[local_blocks]["momentum_buffer"].placements,
            local_blocks.placements,
        )

        dim1_sharded = make_parameter(
            torch.arange(24, device=self.device).reshape(2, 4, 3).float(), 1
        )
        with self.assertRaisesRegex(
            ValueError, "must already match storage sharding"
        ):
            build(dim1_sharded, "dim1_sharded", Shard(0))

        owned = make_parameter(
            torch.arange(12, device=self.device).reshape(4, 3).float(), 0
        )
        with self.assertRaisesRegex(
            ValueError, "requires replicated, 1D Shard"
        ):
            build(owned, "owned", Owned(), owner_rank=0)

    @with_comms
    def test_constructor_requires_exact_bucket_coverage_without_creating_state(self):
        redistributed = self._parameter(
            torch.arange(12, device=self.device).reshape(4, 3).float()
        )
        local_blocks = self._parameter(
            torch.arange(12, 24, device=self.device).reshape(4, 3).float()
        )

        optimizer = self._optimizer(redistributed, local_blocks)
        self.assertEqual(len(optimizer.state), 0)
        redistribution = optimizer._plans[0].redistribution_plans[0]
        self.assertTrue(
            all(
                route.destination_participants == (1,)
                for route in redistribution.storage_to_compute_routes
            )
        )
        redistributed_before = redistributed.to_local().clone()
        redistributed.grad = distribute_tensor(
            torch.ones(4, 3, device=self.device), self.mesh, (Shard(0),)
        )
        with self.assertRaisesRegex(RuntimeError, "every configured gradient"):
            optimizer.step()
        self.assertEqual(len(optimizer.state), 0)
        torch.testing.assert_close(redistributed.to_local(), redistributed_before)
        redistributed.grad = None

        with self.assertRaisesRegex(ValueError, "must match one bucket"):
            build_distributed_muon(
                [
                    {
                        "params": [redistributed, local_blocks],
                        "param_names": [
                            "layers.0.redistributed",
                            "layers.0.local_blocks",
                        ],
                        "compute_sharding": MuonComputeSharding(
                            view_before_placement=BatchedMatrixComputeView(
                                num_matrices=2, matrices_flattened_into_dim=0
                            ),
                            placement=Shard(0),
                        ),
                    }
                ],
                bucket_spec=[
                    BucketSpec(
                        patterns=("*.redistributed",),
                        owner_rank_by_fqn={},
                        mesh=self.mesh,
                    )
                ],
            )

        with self.assertRaisesRegex(ValueError, "must match one bucket"):
            build_distributed_muon(
                [
                    {
                        "params": [redistributed, local_blocks],
                        "param_names": [
                            "layers.0.redistributed",
                            "layers.0.local_blocks",
                        ],
                        "compute_sharding": MuonComputeSharding(
                            view_before_placement=BatchedMatrixComputeView(
                                num_matrices=2, matrices_flattened_into_dim=0
                            ),
                            placement=Shard(0),
                        ),
                    }
                ],
                bucket_spec=[
                    BucketSpec(
                        patterns=("layers.0.*",),
                        owner_rank_by_fqn={},
                        mesh=self.mesh,
                    ),
                    BucketSpec(
                        patterns=("*.local_blocks",),
                        owner_rank_by_fqn={},
                        mesh=self.mesh,
                    ),
                ],
            )

        local_blocks_before = local_blocks.to_local().clone()
        init_optim_state(optimizer)
        self.assertEqual(len(optimizer.state), 2)
        torch.testing.assert_close(redistributed.to_local(), redistributed_before)
        torch.testing.assert_close(local_blocks.to_local(), local_blocks_before)
        flat_state = get_flat_optim_state_dict(optimizer)
        self.assertIn(
            "state.layers.0.redistributed.momentum_buffer", flat_state
        )
        self.assertIn("state.layers.0.local_blocks.momentum_buffer", flat_state)

    @with_comms
    def test_constructor_rejects_storage_shards_that_split_matrices(self):
        parameter = self._parameter(
            torch.arange(36, device=self.device).reshape(12, 3).float()
        )
        with self.assertRaisesRegex(ValueError, "not aligned"):
            build_distributed_muon(
                [
                    {
                        "params": [parameter],
                        "param_names": ["layers.0.wq.weight"],
                        "compute_sharding": MuonComputeSharding(
                            view_before_placement=BatchedMatrixComputeView(
                                num_matrices=3, matrices_flattened_into_dim=0
                            ),
                            placement=Shard(0),
                        ),
                    }
                ],
                bucket_spec=[
                    BucketSpec(
                        patterns=("layers.0.*",),
                        owner_rank_by_fqn={},
                        mesh=self.mesh,
                    )
                ],
            )

    @with_comms
    def test_constructor_requires_valid_owner_assignments(self):
        with self.assertRaises(TypeError):
            Owned(0)

        first = self._parameter(
            torch.arange(12, device=self.device).reshape(4, 3).float()
        )
        second = self._parameter(
            torch.arange(12, 24, device=self.device).reshape(4, 3).float()
        )
        params = [
            {
                "params": [first, second],
                "param_names": ["layers.0.first", "layers.0.second"],
                "compute_sharding": MuonComputeSharding(placement=Owned()),
            }
        ]

        with self.assertRaisesRegex(TypeError, "compute_sharding"):
            build_distributed_muon(
                [{"params": [first], "param_names": ["layers.0.first"]}],
                bucket_spec=[
                    BucketSpec(
                        patterns=("layers.0.*",),
                        owner_rank_by_fqn={},
                        mesh=self.mesh,
                    )
                ],
            )

        with self.assertRaisesRegex(ValueError, "batch of complete Muon matrices"):
            build_distributed_muon(
                [
                    {
                        "params": [first],
                        "param_names": ["layers.0.first"],
                        "compute_sharding": MuonComputeSharding(
                            placement=Shard(0)
                        ),
                    }
                ],
                bucket_spec=[
                    BucketSpec(
                        patterns=("layers.0.*",),
                        owner_rank_by_fqn={},
                        mesh=self.mesh,
                    )
                ],
            )

        with self.assertRaisesRegex(ValueError, "owned Muon parameter"):
            build_distributed_muon(
                [
                    {
                        "params": [first],
                        "param_names": ["layers.0.first"],
                        "compute_sharding": MuonComputeSharding(
                            view_before_placement=BatchedMatrixComputeView(
                                num_matrices=2
                            ),
                            placement=Owned(),
                        ),
                    }
                ],
                bucket_spec=[
                    BucketSpec(
                        patterns=("layers.0.*",),
                        owner_rank_by_fqn={"layers.0.first": 0},
                        mesh=self.mesh,
                    )
                ],
            )

        with self.assertRaisesRegex(ValueError, "exactly cover"):
            build_distributed_muon(
                params,
                bucket_spec=[
                    BucketSpec(
                        patterns=("layers.0.*",),
                        owner_rank_by_fqn={"layers.0.first": 0},
                        mesh=self.mesh,
                    )
                ],
            )
        self.assertIn("compute_sharding", params[0])

        with self.assertRaisesRegex(ValueError, "outside its process group"):
            build_distributed_muon(
                params,
                bucket_spec=[
                    BucketSpec(
                        patterns=("layers.0.*",),
                        owner_rank_by_fqn={
                            "layers.0.first": 0,
                            "layers.0.second": self.world_size,
                        },
                        mesh=self.mesh,
                    )
                ],
            )

    @with_comms
    def test_constructor_rejects_cross_rank_plan_mismatch(self):
        redistributed = self._parameter(
            torch.arange(12, device=self.device).reshape(4, 3).float()
        )
        with self.assertRaisesRegex(RuntimeError, "plans differ across ranks"):
            build_distributed_muon(
                [
                    {
                        "params": [redistributed],
                        "param_names": ["layers.0.redistributed"],
                        "compute_sharding": MuonComputeSharding(
                            placement=Owned()
                        ),
                    }
                ],
                bucket_spec=[
                    BucketSpec(
                        patterns=("layers.0.*",),
                        owner_rank_by_fqn={"layers.0.redistributed": self.rank},
                        mesh=self.mesh,
                    )
                ],
            )

        with self.assertRaisesRegex(RuntimeError, "plans differ across ranks"):
            build_distributed_muon(
                [
                    {
                        "params": [redistributed],
                        "param_names": ["layers.0.redistributed"],
                        "compute_sharding": MuonComputeSharding(
                            placement=Owned()
                        ),
                    }
                ],
                bucket_spec=[
                    BucketSpec(
                        patterns=("layers.0.*",),
                        owner_rank_by_fqn={"layers.0.redistributed": 0},
                        mesh=self.mesh,
                    )
                ],
                lr=0.01 if self.rank == 0 else 0.02,
            )

    @with_comms
    def test_constructor_accepts_uneven_storage_shards(self):
        redistributed = self._parameter(
            torch.arange(15, device=self.device).reshape(5, 3).float()
        )
        optimizer = build_distributed_muon(
            [
                {
                    "params": [redistributed],
                    "param_names": ["layers.0.redistributed"],
                    "compute_sharding": MuonComputeSharding(placement=Owned()),
                }
            ],
            bucket_spec=[
                BucketSpec(
                    patterns=("layers.0.*",),
                    owner_rank_by_fqn={"layers.0.redistributed": 0},
                    mesh=self.mesh,
                )
            ],
        )

        schedule = optimizer._plans[0].storage_to_compute_schedule
        self.assertEqual(
            schedule.input_buffer_numel, 9 if self.rank == 0 else 6
        )

    @with_comms
    def test_replicated_storage_matches_plain_muon_without_redistribution(self):
        values = [
            torch.arange(offset, offset + 12, device=self.device)
            .reshape(4, 3)
            .float()
            .div_(10)
            for offset in (1, 13)
        ]
        owned, batched = (
            torch.nn.Parameter(
                distribute_tensor(value.clone(), self.mesh, (Replicate(),))
            )
            for value in values
        )
        optimizer = build_distributed_muon(
            [
                {
                    "params": [owned],
                    "param_names": ["layers.0.owned"],
                    "compute_sharding": MuonComputeSharding(placement=Owned()),
                },
                {
                    "params": [batched],
                    "param_names": ["layers.0.batched"],
                    "compute_sharding": MuonComputeSharding(
                        view_before_placement=BatchedMatrixComputeView(
                            num_matrices=2, matrices_flattened_into_dim=0
                        ),
                        placement=Shard(0),
                    ),
                },
            ],
            bucket_spec=[
                BucketSpec(
                    patterns=("layers.0.*",),
                    owner_rank_by_fqn={},
                    mesh=self.mesh,
                )
            ],
            lr=0.03,
            weight_decay=0.2,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
        )

        grads = [value.flip(0).contiguous() for value in values]
        for param, grad in zip((owned, batched), grads, strict=True):
            param.grad = distribute_tensor(grad, self.mesh, (Replicate(),))

        references = [
            torch.nn.Parameter(values[0].clone()),
            torch.nn.Parameter(values[1].view(2, 2, 3).clone()),
        ]
        reference_optimizer = torch.optim.Muon(
            references,
            lr=0.03,
            weight_decay=0.2,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
        )
        references[0].grad = grads[0]
        references[1].grad = grads[1].view(2, 2, 3)

        all_to_all_single = dist.all_to_all_single
        with patch(
            "torchtitan.components.distributed_optimizers.bucketed_redistribution.dist."
            "all_to_all_single",
            wraps=all_to_all_single,
        ) as collective:
            optimizer.step()
        reference_optimizer.step()

        collective.assert_not_called()
        for param, reference in zip(
            (owned, batched), references, strict=True
        ):
            self.assertEqual(param.placements, (Replicate(),))
            self.assertEqual(param.grad.placements, (Replicate(),))
            momentum = optimizer.state[param]["momentum_buffer"]
            self.assertEqual(momentum.placements, (Replicate(),))
            torch.testing.assert_close(
                param.to_local(), reference.view(param.shape)
            )
            torch.testing.assert_close(
                momentum.to_local(),
                reference_optimizer.state[reference]["momentum_buffer"].view(
                    param.shape
                ),
            )

    @with_comms
    def test_step_matches_plain_muon_and_continues_from_state_dict(self):
        redistributed_value = (
            torch.arange(12, device=self.device)
            .reshape(4, 3)
            .float()
            .div_(10)
            .add_(1)
        )
        local_blocks_value = (
            torch.arange(12, 24, device=self.device)
            .reshape(4, 3)
            .float()
            .div_(10)
        )
        redistributed = self._parameter(redistributed_value)
        local_blocks = self._parameter(local_blocks_value)
        optimizer = self._optimizer(redistributed, local_blocks)
        self.assertEqual(len(optimizer.state), 0)

        reference_redistributed = torch.nn.Parameter(redistributed_value.clone())
        reference_local_blocks = tuple(
            torch.nn.Parameter(block.clone())
            for block in local_blocks_value.chunk(self.world_size, dim=0)
        )
        reference_optimizer = torch.optim.Muon(
            [reference_redistributed, *reference_local_blocks],
            lr=0.03,
            weight_decay=0.2,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
        )

        first_redistributed_grad = (
            torch.arange(1, 13, device=self.device)
            .reshape(4, 3)
            .float()
            .div_(17)
        )
        first_local_blocks_grad = (
            torch.arange(13, 25, device=self.device)
            .reshape(4, 3)
            .float()
            .div_(19)
        )
        self._set_grads(
            redistributed,
            local_blocks,
            first_redistributed_grad,
            first_local_blocks_grad,
        )
        reference_redistributed.grad = first_redistributed_grad.clone()
        for parameter, grad in zip(
            reference_local_blocks,
            first_local_blocks_grad.chunk(self.world_size, dim=0),
        ):
            parameter.grad = grad.clone()

        all_to_all_single = dist.all_to_all_single
        with patch(
            "torchtitan.components.distributed_optimizers.bucketed_redistribution.dist."
            "all_to_all_single",
            wraps=all_to_all_single,
        ) as collective:
            optimizer.step()
        self.assertEqual(collective.call_count, 2)
        reference_optimizer.step()
        self._assert_matches_reference(
            optimizer,
            redistributed,
            local_blocks,
            reference_optimizer,
            reference_redistributed,
            reference_local_blocks,
        )

        state_dict = optimizer.state_dict()
        self.assertTrue(
            all(
                "compute_sharding" not in group
                and "_compute_placement" not in group
                for group in state_dict["param_groups"]
            )
        )
        resumed_redistributed = self._parameter(reference_redistributed.detach())
        resumed_local_blocks = self._parameter(
            torch.cat([parameter.detach() for parameter in reference_local_blocks])
        )
        resumed_optimizer = self._optimizer(
            resumed_redistributed, resumed_local_blocks
        )
        resumed_optimizer.load_state_dict(state_dict)

        second_redistributed_grad = first_redistributed_grad.flip(0).contiguous()
        second_local_blocks_grad = first_local_blocks_grad.flip(0).contiguous()
        self._set_grads(
            resumed_redistributed,
            resumed_local_blocks,
            second_redistributed_grad,
            second_local_blocks_grad,
        )
        reference_redistributed.grad = second_redistributed_grad.clone()
        for parameter, grad in zip(
            reference_local_blocks,
            second_local_blocks_grad.chunk(self.world_size, dim=0),
        ):
            parameter.grad = grad.clone()

        with patch(
            "torchtitan.components.distributed_optimizers.bucketed_redistribution.dist."
            "all_to_all_single",
            wraps=all_to_all_single,
        ) as collective:
            resumed_optimizer.step()
        self.assertEqual(collective.call_count, 2)
        reference_optimizer.step()
        self._assert_matches_reference(
            resumed_optimizer,
            resumed_redistributed,
            resumed_local_blocks,
            reference_optimizer,
            reference_redistributed,
            reference_local_blocks,
        )


@unittest.skipUnless(torch.cuda.device_count() >= 4, "requires four CUDA devices")
class TestTensorParallelDistributedMuon(_DistributedMuonTestBase):
    @property
    def world_size(self):
        return 4

    @property
    def mesh(self):
        if not hasattr(self, "_mesh"):
            self._mesh = init_device_mesh(
                self.device_type,
                (2, 2),
                mesh_dim_names=("fsdp", "tp"),
            )
        return self._mesh

    @with_comms
    def test_shard0_shard1_bucket_matches_plain_muon(self):
        placements = (Shard(0), Shard(1))
        values = [
            torch.arange(35, device=self.device).reshape(5, 7).float().div_(10),
            torch.arange(35, 65, device=self.device).reshape(6, 5).float().div_(10),
        ]
        params = [
            torch.nn.Parameter(
                distribute_tensor(value.clone(), self.mesh, placements)
            )
            for value in values
        ]
        names = ["layers.0.first", "layers.0.second"]
        optimizer = build_distributed_muon(
            [
                {
                    "params": params,
                    "param_names": names,
                    "compute_sharding": MuonComputeSharding(placement=Owned()),
                }
            ],
            bucket_spec=[
                BucketSpec(
                    patterns=("layers.0.*",),
                    owner_rank_by_fqn={names[0]: 1, names[1]: 3},
                    mesh=self.redistribution_mesh,
                )
            ],
            lr=0.03,
            weight_decay=0.2,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
        )

        grads = [value.flip((0, 1)).contiguous() for value in values]
        for param, grad in zip(params, grads, strict=True):
            param.grad = distribute_tensor(grad, self.mesh, placements)

        references = [torch.nn.Parameter(value.clone()) for value in values]
        reference_optimizer = torch.optim.Muon(
            references,
            lr=0.03,
            weight_decay=0.2,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
        )
        for reference, grad in zip(references, grads, strict=True):
            reference.grad = grad.clone()

        all_to_all_single = dist.all_to_all_single
        with patch(
            "torchtitan.components.distributed_optimizers.bucketed_redistribution.dist.all_to_all_single",
            wraps=all_to_all_single,
        ) as collective:
            optimizer.step()
        reference_optimizer.step()

        self.assertEqual(collective.call_count, 2)
        for param, reference in zip(params, references, strict=True):
            expected_param = distribute_tensor(
                reference.detach(), self.mesh, placements
            )
            expected_momentum = distribute_tensor(
                reference_optimizer.state[reference]["momentum_buffer"],
                self.mesh,
                placements,
            )
            momentum = optimizer.state[param]["momentum_buffer"]
            self.assertEqual(param.placements, placements)
            self.assertEqual(momentum.placements, placements)
            torch.testing.assert_close(param.to_local(), expected_param.to_local())
            torch.testing.assert_close(
                momentum.to_local(), expected_momentum.to_local()
            )

    @with_comms
    def test_distinct_bucket_meshes_use_mesh_local_owners(self):
        fsdp_mesh = self.mesh["fsdp"]
        tp_mesh = self.mesh["tp"]
        meshes = (fsdp_mesh, tp_mesh)
        values = (
            torch.arange(15, device=self.device).reshape(5, 3).float().div_(10),
            torch.arange(20, device=self.device).reshape(4, 5).float().div_(10),
        )
        params = [
            torch.nn.Parameter(
                distribute_tensor(value.clone(), mesh, (Shard(0),))
            )
            for value, mesh in zip(values, meshes, strict=True)
        ]
        names = ("layers.0.fsdp", "layers.1.tp")
        optimizer = build_distributed_muon(
            [
                {
                    "params": [param],
                    "param_names": [name],
                    "compute_sharding": MuonComputeSharding(placement=Owned()),
                }
                for param, name in zip(params, names, strict=True)
            ],
            bucket_spec=[
                BucketSpec(
                    patterns=(name,),
                    owner_rank_by_fqn={name: 1},
                    mesh=mesh,
                )
                for name, mesh in zip(names, meshes, strict=True)
            ],
            ns_steps=1,
        )

        for param, value, mesh in zip(params, values, meshes, strict=True):
            param.grad = distribute_tensor(torch.ones_like(value), mesh, (Shard(0),))

        all_to_all_single = dist.all_to_all_single
        with patch(
            "torchtitan.components.distributed_optimizers.bucketed_redistribution.dist."
            "all_to_all_single",
            wraps=all_to_all_single,
        ) as collective:
            optimizer.step()

        for plan, mesh in zip(optimizer._plans, meshes, strict=True):
            participants = tuple(dist.get_process_group_ranks(mesh.get_group()))
            route = plan.redistribution_plans[0].storage_to_compute_routes[0]
            self.assertEqual(route.destination_participants, (participants[1],))
            self.assertEqual(
                sum(
                    call.kwargs["group"] is mesh.get_group()
                    for call in collective.call_args_list
                ),
                2,
            )


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestDistributedMuonPipeline(_DistributedMuonTestBase):
    @with_comms
    def test_local_only_bucket_does_not_reuse_inflight_slot(self):
        values = [
            torch.arange(offset, offset + 12, device=self.device)
            .reshape(4, 3)
            .float()
            .div_(10)
            for offset in (0, 12, 24)
        ]
        distributed_0, local_blocks, distributed_2 = map(self._parameter, values)
        optimizer = build_distributed_muon(
            [
                {
                    "params": [distributed_0, distributed_2],
                    "param_names": [
                        "layers.0.redistributed",
                        "layers.2.redistributed",
                    ],
                    "compute_sharding": MuonComputeSharding(placement=Owned()),
                },
                {
                    "params": [local_blocks],
                    "param_names": ["layers.1.local_blocks"],
                    "compute_sharding": MuonComputeSharding(
                        view_before_placement=BatchedMatrixComputeView(
                            num_matrices=2, matrices_flattened_into_dim=0
                        ),
                        placement=Shard(0),
                    ),
                },
            ],
            bucket_spec=[
                BucketSpec(
                    patterns=("layers.0.*",),
                    owner_rank_by_fqn={"layers.0.redistributed": 0},
                    mesh=self.mesh,
                ),
                BucketSpec(
                    patterns=("layers.1.*",),
                    owner_rank_by_fqn={},
                    mesh=self.mesh,
                ),
                BucketSpec(
                    patterns=("layers.2.*",),
                    owner_rank_by_fqn={"layers.2.redistributed": 0},
                    mesh=self.mesh,
                ),
            ],
            lr=0.03,
            momentum=0.8,
            ns_steps=1,
        )
        grads = [torch.full_like(value, index + 1) for index, value in enumerate(values)]
        for param, grad in zip(
            (distributed_0, local_blocks, distributed_2), grads, strict=True
        ):
            param.grad = distribute_tensor(grad, self.mesh, (Shard(0),))

        rank = self.mesh.get_local_rank()
        references = [
            torch.nn.Parameter(values[0].clone()),
            torch.nn.Parameter(values[1].chunk(self.world_size, dim=0)[rank].clone()),
            torch.nn.Parameter(values[2].clone()),
        ]
        reference = torch.optim.Muon(
            references, lr=0.03, momentum=0.8, ns_steps=1
        )
        references[0].grad = grads[0].clone()
        references[1].grad = grads[1].chunk(self.world_size, dim=0)[rank].clone()
        references[2].grad = grads[2].clone()

        all_to_all_single = dist.all_to_all_single
        with patch(
            "torchtitan.components.distributed_optimizers.bucketed_redistribution.dist."
            "all_to_all_single",
            wraps=all_to_all_single,
        ) as collective:
            optimizer.step()
        reference.step()

        self.assertEqual(collective.call_count, 4)
        splits = [
            (
                tuple(call.kwargs["input_split_sizes"]),
                tuple(call.kwargs["output_split_sizes"]),
            )
            for call in collective.call_args_list
        ]
        self.assertEqual(splits[0], splits[1])
        self.assertEqual(splits[2], splits[3])
        self.assertNotEqual(splits[0], splits[2])
        torch.testing.assert_close(
            distributed_0.to_local(), references[0].chunk(self.world_size, dim=0)[rank]
        )
        torch.testing.assert_close(local_blocks.to_local(), references[1])
        torch.testing.assert_close(
            distributed_2.to_local(), references[2].chunk(self.world_size, dim=0)[rank]
        )


if __name__ == "__main__":
    unittest.main()
