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
    _LocalBucketPlan,
    BucketConfig,
)
from torchtitan.components.distributed_optimizers.muon import (
    BatchedMatrixComputeView,
    build_distributed_muon,
    DistributedMuon,
    MuonComputeSharding,
    Owned,
)
from torchtitan.components.distributed_optimizers.muon.distributed_muon import (
    _adjust_muon_learning_rate,
)


# Allow a few BF16 quantization steps across different GEMM schedules.
_BATCHED_BF16_DIRECTION_ATOL = 2e-2


def _assert_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _assert_batched_muon_update_close(
    actual_before: torch.Tensor,
    actual_after: torch.Tensor,
    expected_before: torch.Tensor,
    expected_after: torch.Tensor,
    *,
    lr: float,
    weight_decay: float,
    compute_matrix_shape: torch.Size,
) -> None:
    decay = 1 - lr * weight_decay
    adjusted_lr = _adjust_muon_learning_rate(lr, None, compute_matrix_shape)
    actual_update = (actual_before * decay - actual_after) / adjusted_lr
    expected_update = (expected_before * decay - expected_after) / adjusted_lr
    torch.testing.assert_close(
        actual_update,
        expected_update,
        rtol=0,
        atol=_BATCHED_BF16_DIRECTION_ATOL,
    )


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
                        ),
                        placement=Shard(0),
                    ),
                },
            ],
            bucket_configs=[
                BucketConfig(
                    patterns=("layers.0.*",),
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
        reference_local_blocks: tuple[torch.nn.Parameter, torch.nn.Parameter],
        *,
        local_blocks_before: torch.Tensor,
        reference_local_blocks_before: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        rank = self.mesh.get_local_rank()
        expected_redistributed = reference_redistributed.detach().chunk(
            self.world_size, dim=0
        )[rank]
        expected_local_blocks = reference_local_blocks[rank].detach()
        _assert_exact(redistributed.to_local(), expected_redistributed)
        _assert_batched_muon_update_close(
            local_blocks_before,
            local_blocks.to_local(),
            reference_local_blocks_before[rank],
            expected_local_blocks,
            lr=0.03,
            weight_decay=0.2,
            compute_matrix_shape=expected_local_blocks.shape,
        )

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
        _assert_exact(
            redistributed_momentum.to_local(), expected_redistributed_momentum
        )

        local_blocks_momentum = optimizer.state[local_blocks]["momentum_buffer"]
        self.assertIsInstance(local_blocks_momentum, DTensor)
        self.assertEqual(local_blocks_momentum.placements, (Shard(0),))
        expected_local_blocks_momentum = reference_optimizer.state[
            reference_local_blocks[rank]
        ]["momentum_buffer"].detach()
        _assert_exact(local_blocks_momentum.to_local(), expected_local_blocks_momentum)

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
                        ),
                        placement=Shard(0),
                    ),
                }
            ],
            bucket_configs=[
                BucketConfig(
                    patterns=("layers.0.*",),
                    mesh_axis="dp_shard",
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
        *,
        parameter_before: torch.Tensor,
        reference_parameters_before: tuple[torch.Tensor, ...],
    ) -> None:
        expected_parameter = torch.cat(
            [value.detach() for value in reference_parameters], dim=0
        )
        expected_parameter_before = torch.cat(reference_parameters_before, dim=0)
        local_rows, row_offset = Shard.local_shard_size_and_offset(
            expected_parameter.shape[0],
            self.world_size,
            self.mesh.get_local_rank(),
        )
        _assert_batched_muon_update_close(
            parameter_before,
            parameter.to_local(),
            expected_parameter_before.narrow(0, row_offset, local_rows),
            expected_parameter.narrow(0, row_offset, local_rows),
            lr=0.03,
            weight_decay=0.2,
            compute_matrix_shape=reference_parameters[0].shape,
        )
        self.assertEqual(parameter.placements, (Shard(0),))

        self._assert_split_head_momentum_matches_reference(
            optimizer,
            parameter,
            reference_optimizer,
            reference_parameters,
        )

    def _assert_split_head_momentum_matches_reference(
        self,
        optimizer: DistributedMuon,
        parameter: torch.nn.Parameter,
        reference_optimizer: torch.optim.Muon,
        reference_parameters: tuple[torch.nn.Parameter, ...],
    ) -> None:
        local_rows, row_offset = Shard.local_shard_size_and_offset(
            parameter.shape[0],
            self.world_size,
            self.mesh.get_local_rank(),
        )

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
        _assert_exact(
            momentum.to_local(),
            expected_momentum.narrow(0, row_offset, local_rows),
        )


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestDistributedMuon(_DistributedMuonTestBase):
    @with_comms
    def test_replicated_storage_honors_shard_and_owned_compute(self):
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
            bucket_configs=[
                BucketConfig(
                    patterns=("layers.0.*",),
                    mesh_axis="dp_shard",
                )
            ],
            lr=0.03,
            weight_decay=0.2,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
        )
        plan = optimizer._plans[0]
        self.assertEqual(plan.unredistributed_items, ())
        redistribution_by_fqn = {
            layout.fqn: redistribution_plan
            for layout, redistribution_plan in zip(
                plan.redistributed_items,
                plan.redistribution_plans,
                strict=True,
            )
        }
        batched_partitions = redistribution_by_fqn[
            "layers.0.batched"
        ].compute_partitions
        self.assertEqual(
            tuple(partition.tensor_shape for partition in batched_partitions),
            ((1, 2, 3), (1, 2, 3)),
        )
        owned_partitions = tuple(
            partition
            for partition in redistribution_by_fqn["layers.0.owned"].compute_partitions
            if partition.logical_regions
        )
        self.assertEqual(len(owned_partitions), 1)
        self.assertEqual(owned_partitions[0].tensor_shape, (4, 3))

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
        batched_before = batched.to_local().clone()
        reference_batched_before = torch.stack(
            [reference.detach().clone() for reference in batched_references]
        ).view(batched.shape)
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

        self.assertEqual(collective.call_count, 1)
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
        _assert_exact(owned.to_local(), reference_values[0])
        _assert_batched_muon_update_close(
            batched_before,
            batched.to_local(),
            reference_batched_before,
            reference_values[1],
            lr=0.03,
            weight_decay=0.2,
            compute_matrix_shape=batched_references[0].shape,
        )
        for param, reference_momentum in zip(
            (owned, batched),
            reference_momenta,
            strict=True,
        ):
            self.assertEqual(param.placements, (Replicate(),))
            self.assertEqual(param.grad.placements, (Replicate(),))
            momentum = optimizer.state[param]["momentum_buffer"]
            self.assertEqual(momentum.placements, (Replicate(),))
            _assert_exact(
                momentum.to_local(),
                reference_momentum,
            )

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
        local_blocks_before = local_blocks.to_local().clone()
        reference_local_blocks_before = tuple(
            reference.detach().clone() for reference in reference_local_blocks
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
            local_blocks_before=local_blocks_before,
            reference_local_blocks_before=reference_local_blocks_before,
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
        resumed_redistributed_value = redistributed.full_tensor().detach()
        resumed_local_blocks_value = local_blocks.full_tensor().detach()
        resumed_redistributed = self._parameter(resumed_redistributed_value)
        resumed_local_blocks = self._parameter(resumed_local_blocks_value)
        changed_view_optimizer = self._optimizer(
            self._parameter(resumed_redistributed_value),
            self._parameter(resumed_local_blocks_value),
            local_num_matrices=4,
        )
        with self.assertRaisesRegex(ValueError, "compute layout"):
            changed_view_optimizer.load_state_dict(state_dict)

        resumed_optimizer = self._optimizer(
            resumed_redistributed,
            resumed_local_blocks,
        )
        init_optim_state(resumed_optimizer)
        load_flat_optim_state_dict(resumed_optimizer, flat_state_dict)
        _assert_exact(resumed_redistributed.to_local(), redistributed.to_local())
        _assert_exact(resumed_local_blocks.to_local(), local_blocks.to_local())

        second_redistributed_grad = first_redistributed_grad.flip(0).contiguous()
        second_local_blocks_grad = first_local_blocks_grad.flip(0).contiguous()
        local_blocks_before = resumed_local_blocks.to_local().clone()
        reference_local_blocks_before = tuple(
            reference.detach().clone() for reference in reference_local_blocks
        )
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
            local_blocks_before=local_blocks_before,
            reference_local_blocks_before=reference_local_blocks_before,
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
        parameter_before = parameter.to_local().clone()
        reference_parameters_before = tuple(
            reference.detach().clone() for reference in reference_parameters
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
            parameter_before=parameter_before,
            reference_parameters_before=reference_parameters_before,
        )

        state_dict = optimizer.state_dict()
        flat_state_dict = get_flat_optim_state_dict(optimizer)
        resumed_value = parameter.full_tensor().detach()
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
        _assert_exact(resumed_parameter.to_local(), parameter.to_local())
        self._assert_split_head_momentum_matches_reference(
            resumed_optimizer,
            resumed_parameter,
            reference_optimizer,
            reference_parameters,
        )

        second_grad = first_grad.flip((0, 1)).contiguous()
        resumed_parameter_before = resumed_parameter.to_local().clone()
        reference_parameters_before = tuple(
            reference.detach().clone() for reference in reference_parameters
        )
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
            parameter_before=resumed_parameter_before,
            reference_parameters_before=reference_parameters_before,
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
        parameter_before = parameter.to_local().clone()
        reference_parameters_before = tuple(
            reference.detach().clone() for reference in reference_parameters
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
            parameter_before=parameter_before,
            reference_parameters_before=reference_parameters_before,
        )

    @with_comms
    def test_uneven_and_empty_head_shards(self):
        with self.subTest("uneven heads"):
            self._run_split_head_case(num_heads=5, head_rows=3)
        # Eight storage rows keep every FSDP rank nonempty, while only the
        # first two ranks receive a complete head for Muon compute.
        with self.subTest("more ranks than heads"):
            self._run_split_head_case(num_heads=2, head_rows=4)


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
                        view_before_placement=BatchedMatrixComputeView(num_matrices=2),
                        placement=Shard(0),
                    ),
                },
            ],
            bucket_configs=[
                BucketConfig(
                    patterns=("layers.0.*",),
                    mesh_axis="dp_shard",
                ),
                BucketConfig(
                    patterns=("layers.1.*",),
                    mesh_axis="dp_shard",
                ),
                BucketConfig(
                    patterns=("layers.2.*",),
                    mesh_axis="dp_shard",
                ),
            ],
            num_pipeline_slots=3,
            lr=0.03,
            momentum=0.8,
            ns_steps=1,
        )
        self.assertIsInstance(optimizer._plans[1], _LocalBucketPlan)
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
        local_blocks_before = local_blocks.to_local().clone()
        reference_local_blocks_before = references[1].detach().clone()
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
        communicating_plans = (optimizer._plans[0], optimizer._plans[2])
        expected_splits = [
            (
                tuple(plan.storage_to_compute_schedule.input_split_sizes),
                tuple(plan.storage_to_compute_schedule.output_split_sizes),
            )
            for plan in communicating_plans
        ] + [
            (
                tuple(plan.compute_to_storage_schedule.input_split_sizes),
                tuple(plan.compute_to_storage_schedule.output_split_sizes),
            )
            for plan in communicating_plans
        ]
        self.assertEqual(splits, expected_splits)
        _assert_exact(
            distributed_0.to_local(), references[0].chunk(self.world_size, dim=0)[rank]
        )
        _assert_batched_muon_update_close(
            local_blocks_before,
            local_blocks.to_local(),
            reference_local_blocks_before,
            references[1],
            lr=0.03,
            weight_decay=0.1,
            compute_matrix_shape=references[1].shape,
        )
        _assert_exact(
            distributed_2.to_local(), references[2].chunk(self.world_size, dim=0)[rank]
        )


if __name__ == "__main__":
    unittest.main()
