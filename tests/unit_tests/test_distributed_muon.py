# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.components.checkpoint_utils import (
    get_flat_optim_state_dict,
    init_optim_state,
    load_flat_optim_state_dict,
)
from torchtitan.components.distributed_optimizers.flex_optimizer_reshard import (
    BucketConfig,
    BucketSpec,
)
from torchtitan.components.distributed_optimizers.muon import (
    _has_dim0_sharded_storage,
    _has_replicated_storage,
    DistributedMuon,
    Owned,
)
from torchtitan.components.distributed_optimizers.muon_parameter_prep import (
    BatchedMatrixComputeView,
    build_distributed_muon,
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

    def test_rejects_nan_hyperparameters(self):
        valid_group = {
            "lr": 0.03,
            "weight_decay": 0.2,
            "momentum": 0.8,
            "nesterov": True,
            "ns_coefficients": (3.4445, -4.7750, 2.0315),
            "eps": 1e-7,
            "ns_steps": 2,
            "adjust_lr_fn": None,
            "fused": False,
            "foreach": False,
        }
        optimizer = object.__new__(DistributedMuon)
        for name in ("lr", "weight_decay", "momentum", "eps"):
            with self.subTest(name=name):
                optimizer.param_groups = [{**valid_group, name: float("nan")}]
                with self.assertRaisesRegex(
                    ValueError, "unsupported DistributedMuon group 0"
                ):
                    optimizer._validate_groups()


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
        *,
        local_num_matrices: int = 2,
        owner_rank: int = 1,
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
                            num_matrices=local_num_matrices,
                            matrices_flattened_into_dim=0,
                        ),
                        placement=Shard(0),
                    ),
                },
            ],
            bucket_configs=[
                BucketConfig(
                    patterns=("layers.0.*",),
                    owner_rank_by_fqn={"layers.0.redistributed": owner_rank},
                    mesh_axes=("dp_shard",),
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
        reference_local_blocks: tuple[torch.nn.Parameter, torch.nn.Parameter],
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
        expected_redistributed_momentum = (
            reference_optimizer.state[reference_redistributed]["momentum_buffer"]
            .detach()
            .chunk(self.world_size, dim=0)[rank]
        )
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

    def _build_split_head_optimizer(
        self,
        value: torch.Tensor,
        *,
        num_heads: int,
    ) -> tuple[torch.nn.Parameter, DistributedMuon]:
        parameter = self._parameter(value)
        optimizer = build_distributed_muon(
            [
                {
                    "params": [parameter],
                    "param_names": ["layers.0.wq.weight"],
                    "compute_sharding": MuonComputeSharding(
                        view_before_placement=BatchedMatrixComputeView(
                            num_matrices=num_heads,
                            matrices_flattened_into_dim=0,
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
            lr=0.03,
            weight_decay=0.2,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
        )
        return parameter, optimizer

    def _build_per_head_reference(
        self,
        value: torch.Tensor,
        *,
        num_heads: int,
    ) -> tuple[tuple[torch.nn.Parameter, ...], torch.optim.Muon]:
        head_rows = value.shape[0] // num_heads
        parameters = tuple(
            torch.nn.Parameter(head.clone())
            for head in value.view(num_heads, head_rows, value.shape[1])
        )
        optimizer = torch.optim.Muon(
            parameters,
            lr=0.03,
            weight_decay=0.2,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
        )
        return parameters, optimizer

    def _set_split_head_grads(
        self,
        parameter: torch.nn.Parameter,
        reference_parameters: tuple[torch.nn.Parameter, ...],
        grad: torch.Tensor,
    ) -> None:
        parameter.grad = distribute_tensor(grad.clone(), self.mesh, (Shard(0),))
        head_rows = grad.shape[0] // len(reference_parameters)
        for reference_parameter, head_grad in zip(
            reference_parameters,
            grad.view(len(reference_parameters), head_rows, grad.shape[1]),
            strict=True,
        ):
            reference_parameter.grad = head_grad.clone()

    def _assert_split_head_matches_reference(
        self,
        optimizer: DistributedMuon,
        parameter: torch.nn.Parameter,
        reference_optimizer: torch.optim.Muon,
        reference_parameters: tuple[torch.nn.Parameter, ...],
    ) -> None:
        expected_parameter = torch.cat(
            [value.detach() for value in reference_parameters], dim=0
        )
        local_rows, row_offset = Shard.local_shard_size_and_offset(
            expected_parameter.shape[0],
            self.world_size,
            self.mesh.get_local_rank(),
        )
        torch.testing.assert_close(
            parameter.to_local(),
            expected_parameter.narrow(0, row_offset, local_rows),
        )
        self.assertEqual(parameter.placements, (Shard(0),))

        momentum = optimizer.state[parameter]["momentum_buffer"]
        expected_momentum = torch.cat(
            [
                reference_optimizer.state[value]["momentum_buffer"]
                for value in reference_parameters
            ],
            dim=0,
        )
        self.assertIsInstance(momentum, DTensor)
        self.assertEqual(momentum.placements, (Shard(0),))
        torch.testing.assert_close(
            momentum.to_local(),
            expected_momentum.narrow(0, row_offset, local_rows),
        )


@unittest.skipUnless(torch.cuda.device_count() >= 1, "requires one CUDA device")
class TestDistributedMuonSingleRank(_DistributedMuonTestBase):
    @property
    def world_size(self):
        return 1

    @with_comms
    def test_owned_compute_accepts_static_owner_for_replicated_storage(self):
        value = torch.arange(12, device=self.device).reshape(4, 3).float()
        parameter = torch.nn.Parameter(
            distribute_tensor(value.clone(), self.mesh, (Replicate(),))
        )
        optimizer = build_distributed_muon(
            [
                {
                    "params": [parameter],
                    "param_names": ["layers.0.weight"],
                    "compute_sharding": MuonComputeSharding(placement=Owned()),
                }
            ],
            bucket_spec=[
                BucketSpec(
                    patterns=("layers.0.*",),
                    owner_rank_by_fqn={"layers.0.weight": 0},
                    mesh=self.mesh,
                )
            ],
            ns_steps=1,
        )
        parameter.grad = distribute_tensor(
            torch.ones_like(value), self.mesh, (Replicate(),)
        )

        all_to_all_single = dist.all_to_all_single
        with patch(
            "torchtitan.components.distributed_optimizers.flex_optimizer_reshard.dist."
            "all_to_all_single",
            wraps=all_to_all_single,
        ) as collective:
            optimizer.step()

        collective.assert_not_called()


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
            parameter = torch.nn.Parameter(distribute_tensor(value, mesh, placements))
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
        self.assertIs(optimizer._plans[0].local_items[0].param, local_blocks)
        local_blocks.grad = distribute_tensor(
            torch.ones(4, 2, 3, device=self.device),
            mesh,
            local_blocks.placements,
        )
        with patch(
            "torchtitan.components.distributed_optimizers.flex_optimizer_reshard.dist."
            "all_to_all_single"
        ) as collective:
            optimizer.step()
        collective.assert_not_called()
        self.assertEqual(
            optimizer.state[local_blocks]["momentum_buffer"].placements,
            local_blocks.placements,
        )
        local_blocks._local_tensor = local_blocks.to_local().clone()
        with self.assertRaisesRegex(RuntimeError, "local storage changed"):
            optimizer.step()

        backing = torch.arange(24, device=self.device).float()
        first_region = backing[:12].view(2, 2, 3)
        second_region = backing[12:].view(2, 2, 3)
        shared_storage = torch.nn.Parameter(
            DTensor.from_local(
                first_region,
                self.mesh,
                (Shard(0),),
                run_check=False,
            )
        )
        shared_optimizer = build(shared_storage, "shared_storage", Shard(0))
        shared_storage.grad = DTensor.from_local(
            torch.ones_like(first_region),
            self.mesh,
            (Shard(0),),
            run_check=False,
        )
        self.assertEqual(
            shared_storage.to_local().untyped_storage().data_ptr(),
            second_region.untyped_storage().data_ptr(),
        )
        self.assertNotEqual(
            shared_storage.to_local().data_ptr(), second_region.data_ptr()
        )
        shared_storage._local_tensor = second_region
        with self.assertRaisesRegex(RuntimeError, "local storage changed"):
            shared_optimizer.step()

        dim1_sharded = make_parameter(
            torch.arange(24, device=self.device).reshape(2, 4, 3).float(), 1
        )
        with self.assertRaisesRegex(ValueError, "storage-to-compute layout"):
            build(dim1_sharded, "dim1_sharded", Shard(0))

        owned = make_parameter(
            torch.arange(12, device=self.device).reshape(4, 3).float(), 0
        )
        with self.assertRaisesRegex(ValueError, "storage-to-compute layout"):
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
        with patch("torch.distributed.all_reduce") as validation_collective:
            with self.assertRaisesRegex(RuntimeError, "layers.0.local_blocks"):
                optimizer.step()
        validation_collective.assert_not_called()
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
        self.assertIn("state.layers.0.redistributed.momentum_buffer", flat_state)
        self.assertIn("state.layers.0.local_blocks.momentum_buffer", flat_state)

    @with_comms
    def test_flat_state_dict_loads_after_group_membership_changes(self):
        values = {
            "layers.0.a": torch.arange(12, device=self.device).reshape(4, 3).float(),
            "layers.0.b": torch.arange(12, 24, device=self.device)
            .reshape(4, 3)
            .float(),
        }

        def build(names, *, compute_locally=False):
            parameters = [
                torch.nn.Parameter(
                    distribute_tensor(values[name].clone(), self.mesh, (Shard(0),))
                )
                for name in names
            ]
            compute_sharding = (
                MuonComputeSharding(
                    view_before_placement=BatchedMatrixComputeView(num_matrices=2),
                    placement=Shard(0),
                )
                if compute_locally
                else MuonComputeSharding(placement=Owned())
            )
            optimizer = build_distributed_muon(
                [
                    {
                        "params": parameters,
                        "param_names": names,
                        "compute_sharding": compute_sharding,
                    }
                ],
                bucket_spec=[
                    BucketSpec(
                        patterns=("layers.0.*",),
                        owner_rank_by_fqn=(
                            {} if compute_locally else dict.fromkeys(names, 0)
                        ),
                        mesh=self.mesh,
                    )
                ],
                ns_steps=2,
            )
            return parameters, optimizer

        source_parameters, source_optimizer = build(("layers.0.a", "layers.0.b"))
        for name, parameter in zip(
            ("layers.0.a", "layers.0.b"), source_parameters, strict=True
        ):
            parameter.grad = distribute_tensor(
                torch.ones_like(values[name]),
                self.mesh,
                (Shard(0),),
            )
        source_optimizer.step()
        flat_state_dict = get_flat_optim_state_dict(source_optimizer)
        self.assertIn(
            "state.layers.0.a._distributed_muon_layout_fingerprint",
            flat_state_dict,
        )

        target_parameters, target_optimizer = build(("layers.0.a",))
        init_optim_state(target_optimizer)
        load_flat_optim_state_dict(target_optimizer, flat_state_dict)
        torch.testing.assert_close(
            target_optimizer.state[target_parameters[0]]["momentum_buffer"].to_local(),
            source_optimizer.state[source_parameters[0]]["momentum_buffer"].to_local(),
        )

        _, changed_layout_optimizer = build(("layers.0.a",), compute_locally=True)
        init_optim_state(changed_layout_optimizer)
        with self.assertRaisesRegex(ValueError, "compute layout"):
            load_flat_optim_state_dict(
                changed_layout_optimizer,
                flat_state_dict,
            )

    @with_comms
    def test_split_head_shard0_matches_plain_muon_and_continues_from_state_dict(
        self,
    ):
        num_heads = 3
        value = (
            torch.arange(36, device=self.device).reshape(12, 3).float().div_(10).add_(1)
        )
        parameter, optimizer = self._build_split_head_optimizer(
            value,
            num_heads=num_heads,
        )
        reference_parameters, reference_optimizer = self._build_per_head_reference(
            value,
            num_heads=num_heads,
        )

        first_grad = (
            torch.arange(1, 37, device=self.device).reshape(12, 3).float().div_(17)
        )
        self._set_split_head_grads(parameter, reference_parameters, first_grad)
        all_to_all_single = dist.all_to_all_single
        with patch(
            "torchtitan.components.distributed_optimizers.flex_optimizer_reshard."
            "dist.all_to_all_single",
            wraps=all_to_all_single,
        ) as collective:
            optimizer.step()
        self.assertEqual(collective.call_count, 2)
        reference_optimizer.step()
        self._assert_split_head_matches_reference(
            optimizer,
            parameter,
            reference_optimizer,
            reference_parameters,
        )

        state_dict = optimizer.state_dict()
        flat_state_dict = get_flat_optim_state_dict(optimizer)
        resumed_value = torch.cat(
            [reference.detach() for reference in reference_parameters], dim=0
        )
        _, changed_view_optimizer = self._build_split_head_optimizer(
            resumed_value,
            num_heads=2,
        )
        with self.assertRaisesRegex(ValueError, "compute layout"):
            changed_view_optimizer.load_state_dict(state_dict)

        resumed_parameter, resumed_optimizer = self._build_split_head_optimizer(
            resumed_value,
            num_heads=num_heads,
        )
        init_optim_state(resumed_optimizer)
        load_flat_optim_state_dict(resumed_optimizer, flat_state_dict)
        self._assert_split_head_matches_reference(
            resumed_optimizer,
            resumed_parameter,
            reference_optimizer,
            reference_parameters,
        )

        second_grad = first_grad.flip((0, 1)).contiguous()
        self._set_split_head_grads(
            resumed_parameter,
            reference_parameters,
            second_grad,
        )
        with patch(
            "torchtitan.components.distributed_optimizers.flex_optimizer_reshard."
            "dist.all_to_all_single",
            wraps=all_to_all_single,
        ) as collective:
            resumed_optimizer.step()
        self.assertEqual(collective.call_count, 2)
        reference_optimizer.step()
        self._assert_split_head_matches_reference(
            resumed_optimizer,
            resumed_parameter,
            reference_optimizer,
            reference_parameters,
        )

    @with_comms
    def test_split_head_compute_with_actual_fsdp2_parameter(self):
        num_heads = 3
        value = (
            torch.arange(36, device=self.device).reshape(12, 3).float().div_(10).add_(1)
        )
        module = torch.nn.Linear(3, 12, bias=False, device=self.device)
        with torch.no_grad():
            module.weight.copy_(value)
        fully_shard(module, mesh=self.mesh, reshard_after_forward=True)
        parameter = module.weight
        optimizer = build_distributed_muon(
            [
                {
                    "params": [parameter],
                    "param_names": ["layers.0.wq.weight"],
                    "compute_sharding": MuonComputeSharding(
                        view_before_placement=BatchedMatrixComputeView(
                            num_matrices=num_heads,
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
            lr=0.03,
            weight_decay=0.2,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
        )

        input = torch.tensor(
            ((1.0, 2.0, 3.0), (4.0, 1.0, 2.0)),
            device=self.device,
        )
        output_scales = torch.arange(1, 13, device=self.device).float()
        (module(input) * output_scales).sum().backward()
        full_grad = output_scales[:, None] * input.sum(dim=0)[None, :]

        reference_parameters, reference_optimizer = self._build_per_head_reference(
            value,
            num_heads=num_heads,
        )
        for reference_parameter, head_grad in zip(
            reference_parameters,
            full_grad.view(num_heads, 4, 3),
            strict=True,
        ):
            reference_parameter.grad = head_grad.clone()

        all_to_all_single = dist.all_to_all_single
        with patch(
            "torchtitan.components.distributed_optimizers.flex_optimizer_reshard."
            "dist.all_to_all_single",
            wraps=all_to_all_single,
        ) as collective:
            optimizer.step()
        reference_optimizer.step()

        local_rows, row_offset = Shard.local_shard_size_and_offset(
            value.shape[0],
            self.world_size,
            self.mesh.get_local_rank(),
        )
        self.assertIsInstance(parameter, DTensor)
        self.assertEqual(parameter.placements, (Shard(0),))
        self.assertEqual(parameter.to_local().shape, torch.Size((6, 3)))
        self.assertIsInstance(parameter.grad, DTensor)
        torch.testing.assert_close(
            parameter.grad.to_local(),
            full_grad.narrow(0, row_offset, local_rows),
        )
        self.assertEqual(collective.call_count, 2)
        self._assert_split_head_matches_reference(
            optimizer,
            parameter,
            reference_optimizer,
            reference_parameters,
        )

    @with_comms
    def test_constructor_requires_valid_owner_assignments(self):
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

        with self.assertRaisesRegex(ValueError, "storage-to-compute layout"):
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
                        "compute_sharding": MuonComputeSharding(placement=Owned()),
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
        self.assertEqual(schedule.input_buffer_numel, 9 if self.rank == 0 else 6)

    @with_comms
    def test_shard1_owned_matches_plain_muon(self):
        value = torch.arange(15, device=self.device).reshape(3, 5).float().div_(10)
        placement = (Shard(1),)
        parameter = torch.nn.Parameter(
            distribute_tensor(value.clone(), self.mesh, placement)
        )
        optimizer = build_distributed_muon(
            [
                {
                    "params": [parameter],
                    "param_names": ["layers.0.weight"],
                    "compute_sharding": MuonComputeSharding(placement=Owned()),
                }
            ],
            bucket_spec=[
                BucketSpec(
                    patterns=("layers.0.*",),
                    owner_rank_by_fqn={"layers.0.weight": 1},
                    mesh=self.mesh,
                )
            ],
            lr=0.03,
            momentum=0.8,
            ns_steps=2,
        )
        grad = value.flip((0, 1)).contiguous()
        parameter.grad = distribute_tensor(grad, self.mesh, placement)

        reference = torch.nn.Parameter(value.clone())
        reference.grad = grad.clone()
        reference_optimizer = torch.optim.Muon(
            [reference],
            lr=0.03,
            momentum=0.8,
            ns_steps=2,
        )

        all_to_all_single = dist.all_to_all_single
        with patch(
            "torchtitan.components.distributed_optimizers.flex_optimizer_reshard."
            "dist.all_to_all_single",
            wraps=all_to_all_single,
        ) as collective:
            optimizer.step()
        reference_optimizer.step()

        self.assertEqual(collective.call_count, 2)
        expected_parameter = distribute_tensor(reference.detach(), self.mesh, placement)
        expected_momentum = distribute_tensor(
            reference_optimizer.state[reference]["momentum_buffer"],
            self.mesh,
            placement,
        )
        momentum = optimizer.state[parameter]["momentum_buffer"]
        self.assertEqual(parameter.placements, placement)
        self.assertEqual(momentum.placements, placement)
        torch.testing.assert_close(parameter.to_local(), expected_parameter.to_local())
        torch.testing.assert_close(
            momentum.to_local(),
            expected_momentum.to_local(),
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
                        view_before_placement=BatchedMatrixComputeView(num_matrices=2),
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

        owned_reference = torch.nn.Parameter(values[0].clone())
        batched_references = [
            torch.nn.Parameter(matrix.clone())
            for matrix in values[1].view(2, 2, 3).unbind()
        ]
        references = [owned_reference, *batched_references]
        reference_optimizer = torch.optim.Muon(
            references,
            lr=0.03,
            weight_decay=0.2,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
        )
        owned_reference.grad = grads[0]
        for reference, grad in zip(
            batched_references,
            grads[1].view(2, 2, 3).unbind(),
            strict=True,
        ):
            reference.grad = grad

        all_to_all_single = dist.all_to_all_single
        with patch(
            "torchtitan.components.distributed_optimizers.flex_optimizer_reshard.dist."
            "all_to_all_single",
            wraps=all_to_all_single,
        ) as collective:
            optimizer.step()
        reference_optimizer.step()

        collective.assert_not_called()
        reference_values = (
            owned_reference,
            torch.stack(batched_references).view(batched.shape),
        )
        reference_momenta = (
            reference_optimizer.state[owned_reference]["momentum_buffer"],
            torch.stack(
                [
                    reference_optimizer.state[reference]["momentum_buffer"]
                    for reference in batched_references
                ]
            ).view(batched.shape),
        )
        for param, reference, reference_momentum in zip(
            (owned, batched),
            reference_values,
            reference_momenta,
            strict=True,
        ):
            self.assertEqual(param.placements, (Replicate(),))
            self.assertEqual(param.grad.placements, (Replicate(),))
            momentum = optimizer.state[param]["momentum_buffer"]
            self.assertEqual(momentum.placements, (Replicate(),))
            torch.testing.assert_close(param.to_local(), reference)
            torch.testing.assert_close(
                momentum.to_local(),
                reference_momentum,
            )

    @with_comms
    def test_step_rejects_gradient_with_reordered_mesh(self):
        value = torch.arange(12, device=self.device).reshape(4, 3).float()
        parameter = self._parameter(value)
        optimizer = build_distributed_muon(
            [
                {
                    "params": [parameter],
                    "param_names": ["layers.0.weight"],
                    "compute_sharding": MuonComputeSharding(placement=Owned()),
                }
            ],
            bucket_spec=[
                BucketSpec(
                    patterns=("layers.0.*",),
                    owner_rank_by_fqn={"layers.0.weight": 0},
                    mesh=self.mesh,
                )
            ],
            ns_steps=1,
        )
        reversed_mesh = DeviceMesh(
            self.device_type,
            torch.arange(self.world_size - 1, -1, -1),
            mesh_dim_names=("dp_shard",),
        )
        self.assertEqual(
            tuple(dist.get_process_group_ranks(reversed_mesh.get_group())),
            tuple(dist.get_process_group_ranks(self.mesh.get_group())),
        )
        self.assertNotEqual(
            tuple(reversed_mesh.mesh.flatten().tolist()),
            tuple(self.mesh.mesh.flatten().tolist()),
        )
        parameter.grad = distribute_tensor(
            value.flip(0).contiguous(), reversed_mesh, (Shard(0),)
        )
        parameter_before = parameter.to_local().clone()

        with patch(
            "torchtitan.components.distributed_optimizers.flex_optimizer_reshard."
            "dist.all_to_all_single"
        ) as collective:
            with self.assertRaisesRegex(
                RuntimeError, "gradient storage layout changed"
            ):
                optimizer.step()

        collective.assert_not_called()
        self.assertEqual(len(optimizer.state), 0)
        torch.testing.assert_close(parameter.to_local(), parameter_before)

    @with_comms
    def test_step_matches_plain_muon_and_continues_from_state_dict(self):
        redistributed_value = (
            torch.arange(12, device=self.device).reshape(4, 3).float().div_(10).add_(1)
        )
        local_blocks_value = (
            torch.arange(12, 24, device=self.device).reshape(4, 3).float().div_(10)
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
            torch.arange(1, 13, device=self.device).reshape(4, 3).float().div_(17)
        )
        first_local_blocks_grad = (
            torch.arange(13, 25, device=self.device).reshape(4, 3).float().div_(19)
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
            "torchtitan.components.distributed_optimizers.flex_optimizer_reshard.dist."
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

        local_blocks.grad = None
        with self.assertRaisesRegex(RuntimeError, "layers.0.local_blocks"):
            optimizer.step()

        state_dict = optimizer.state_dict()
        flat_state_dict = get_flat_optim_state_dict(optimizer)
        self.assertTrue(
            all(
                "compute_sharding" not in group and "_compute_placement" not in group
                for group in state_dict["param_groups"]
            )
        )
        resumed_redistributed = self._parameter(reference_redistributed.detach())
        resumed_local_blocks = self._parameter(
            torch.cat([parameter.detach() for parameter in reference_local_blocks])
        )
        changed_view_optimizer = self._optimizer(
            self._parameter(reference_redistributed.detach()),
            self._parameter(
                torch.cat([parameter.detach() for parameter in reference_local_blocks])
            ),
            local_num_matrices=4,
        )
        with self.assertRaisesRegex(ValueError, "compute layout"):
            changed_view_optimizer.load_state_dict(state_dict)

        resumed_optimizer = self._optimizer(
            resumed_redistributed,
            resumed_local_blocks,
            owner_rank=0,
        )
        init_optim_state(resumed_optimizer)
        load_flat_optim_state_dict(resumed_optimizer, flat_state_dict)

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
            "torchtitan.components.distributed_optimizers.flex_optimizer_reshard.dist."
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
class TestDistributedMuonUnevenShards(_DistributedMuonTestBase):
    @property
    def world_size(self):
        return 4

    def _run_split_head_case(self, *, num_heads: int, head_rows: int) -> None:
        num_columns = 3
        value = (
            torch.arange(
                num_heads * head_rows * num_columns,
                device=self.device,
            )
            .reshape(num_heads * head_rows, num_columns)
            .float()
            .div_(10)
            .add_(1)
        )
        parameter, optimizer = self._build_split_head_optimizer(
            value,
            num_heads=num_heads,
        )
        reference_parameters, reference_optimizer = self._build_per_head_reference(
            value,
            num_heads=num_heads,
        )
        grad = value.flip((0, 1)).contiguous().div_(17)
        self._set_split_head_grads(parameter, reference_parameters, grad)

        all_to_all_single = dist.all_to_all_single
        with patch(
            "torchtitan.components.distributed_optimizers.flex_optimizer_reshard."
            "dist.all_to_all_single",
            wraps=all_to_all_single,
        ) as collective:
            optimizer.step()
        self.assertEqual(collective.call_count, 2)

        local_rank = self.mesh.get_local_rank()
        local_num_heads, _ = Shard.local_shard_size_and_offset(
            num_heads,
            self.world_size,
            local_rank,
        )
        head_numel = head_rows * num_columns
        storage_to_compute, compute_to_storage = collective.call_args_list
        self.assertEqual(
            sum(storage_to_compute.kwargs["output_split_sizes"]),
            local_num_heads * head_numel,
        )
        self.assertEqual(
            sum(compute_to_storage.kwargs["input_split_sizes"]),
            local_num_heads * head_numel,
        )
        self.assertGreater(parameter.to_local().numel(), 0)

        reference_optimizer.step()
        self._assert_split_head_matches_reference(
            optimizer,
            parameter,
            reference_optimizer,
            reference_parameters,
        )

    @with_comms
    def test_uneven_heads_use_whole_head_shard0_compute(self):
        self._run_split_head_case(num_heads=5, head_rows=3)

    @with_comms
    def test_more_ranks_than_heads_supports_empty_compute_shards(self):
        # Eight storage rows keep every FSDP rank nonempty, while only the
        # first two ranks receive a complete head for Muon compute.
        self._run_split_head_case(num_heads=2, head_rows=4)


@unittest.skipUnless(torch.cuda.device_count() >= 4, "requires four CUDA devices")
class TestDistributedMuonBucketMeshes(_DistributedMuonTestBase):
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
    def test_bucket_config_mixes_dense_and_expert_storage_meshes(self):
        dense_mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("dp_shard",),
        )
        expert_mesh = init_device_mesh(
            self.device_type,
            (2, 2),
            mesh_dim_names=("efsdp", "ep"),
        )
        dense_value = (
            torch.arange(24, device=self.device).reshape(8, 3).float().div_(10)
        )
        qkv_value = (
            torch.arange(24, 60, device=self.device).reshape(12, 3).float().div_(10)
        )
        expert_value = (
            torch.arange(48, device=self.device).reshape(8, 2, 3).float().div_(10)
        )
        dense = torch.nn.Parameter(
            distribute_tensor(dense_value.clone(), dense_mesh, (Shard(0),))
        )
        qkv = torch.nn.Parameter(
            distribute_tensor(qkv_value.clone(), dense_mesh, (Shard(0),))
        )
        expert_placements = (
            _StridedShard(0, split_factor=expert_mesh["ep"].size()),
            Shard(0),
        )
        expert = torch.nn.Parameter(
            distribute_tensor(
                expert_value.clone(),
                expert_mesh,
                expert_placements,
            )
        )
        optimizer = build_distributed_muon(
            [
                {
                    "params": [dense],
                    "param_names": ["layers.0.dense.weight"],
                    "compute_sharding": MuonComputeSharding(placement=Owned()),
                },
                {
                    "params": [qkv],
                    "param_names": ["layers.0.qkv.weight"],
                    "compute_sharding": MuonComputeSharding(
                        view_before_placement=BatchedMatrixComputeView(
                            num_matrices=3,
                        ),
                        placement=Shard(0),
                    ),
                },
                {
                    "params": [expert],
                    "param_names": ["layers.0.experts.weight"],
                    "compute_sharding": MuonComputeSharding(placement=Shard(0)),
                },
            ],
            bucket_configs=[
                BucketConfig(
                    patterns=("layers.0.*",),
                    owner_rank_by_fqn={"layers.0.dense.weight": 1},
                    mesh_axes=("dp_shard",),
                    name="layers.0",
                )
            ],
            lr=0.03,
            weight_decay=0.2,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
        )

        dense_grad = (
            torch.arange(1, 25, device=self.device).reshape(8, 3).float().div_(17)
        )
        qkv_grad = qkv_value.flip((0, 1)).contiguous().div_(23)
        expert_grad = expert_value.flip((0, 1, 2)).contiguous().div_(19)
        dense.grad = distribute_tensor(dense_grad.clone(), dense_mesh, (Shard(0),))
        qkv.grad = distribute_tensor(qkv_grad.clone(), dense_mesh, (Shard(0),))
        expert.grad = distribute_tensor(
            expert_grad.clone(),
            expert_mesh,
            expert_placements,
        )

        reference_dense = torch.nn.Parameter(dense_value.clone())
        reference_qkv = tuple(
            torch.nn.Parameter(matrix.clone()) for matrix in qkv_value.view(3, 4, 3)
        )
        reference_experts = tuple(
            torch.nn.Parameter(matrix.clone()) for matrix in expert.to_local()
        )
        reference_dense.grad = dense_grad.clone()
        for parameter, grad in zip(
            reference_qkv,
            qkv_grad.view(3, 4, 3),
            strict=True,
        ):
            parameter.grad = grad.clone()
        for parameter, grad in zip(
            reference_experts,
            expert.grad.to_local(),
            strict=True,
        ):
            parameter.grad = grad.clone()
        reference_optimizer = torch.optim.Muon(
            [reference_dense, *reference_qkv, *reference_experts],
            lr=0.03,
            weight_decay=0.2,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
        )

        all_to_all_single = dist.all_to_all_single
        with patch(
            "torchtitan.components.distributed_optimizers.flex_optimizer_reshard."
            "dist.all_to_all_single",
            wraps=all_to_all_single,
        ) as collective:
            optimizer.step()
        reference_optimizer.step()

        rank = dense_mesh.get_local_rank()
        self.assertEqual(collective.call_count, 2)
        torch.testing.assert_close(
            dense.to_local(),
            reference_dense.detach().chunk(self.world_size, dim=0)[rank],
        )
        torch.testing.assert_close(
            qkv.to_local(),
            torch.cat([parameter.detach() for parameter in reference_qkv]).chunk(
                self.world_size,
                dim=0,
            )[rank],
        )
        torch.testing.assert_close(
            expert.to_local(),
            torch.stack([parameter.detach() for parameter in reference_experts]),
        )
        dense_momentum = optimizer.state[dense]["momentum_buffer"]
        reference_dense_momentum = reference_optimizer.state[reference_dense][
            "momentum_buffer"
        ]
        torch.testing.assert_close(
            dense_momentum.to_local(),
            reference_dense_momentum.chunk(self.world_size, dim=0)[rank],
        )
        qkv_momentum = optimizer.state[qkv]["momentum_buffer"]
        reference_qkv_momentum = torch.cat(
            [
                reference_optimizer.state[parameter]["momentum_buffer"]
                for parameter in reference_qkv
            ]
        )
        torch.testing.assert_close(
            qkv_momentum.to_local(),
            reference_qkv_momentum.chunk(self.world_size, dim=0)[rank],
        )
        expert_momentum = optimizer.state[expert]["momentum_buffer"]
        torch.testing.assert_close(
            expert_momentum.to_local(),
            torch.stack(
                [
                    reference_optimizer.state[parameter]["momentum_buffer"]
                    for parameter in reference_experts
                ]
            ),
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
            torch.nn.Parameter(distribute_tensor(value.clone(), mesh, (Shard(0),)))
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
            "torchtitan.components.distributed_optimizers.flex_optimizer_reshard.dist."
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
        grads = [
            torch.full_like(value, index + 1) for index, value in enumerate(values)
        ]
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
        reference = torch.optim.Muon(references, lr=0.03, momentum=0.8, ns_steps=1)
        references[0].grad = grads[0].clone()
        references[1].grad = grads[1].chunk(self.world_size, dim=0)[rank].clone()
        references[2].grad = grads[2].clone()

        all_to_all_single = dist.all_to_all_single
        with patch(
            "torchtitan.components.distributed_optimizers.flex_optimizer_reshard.dist."
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
