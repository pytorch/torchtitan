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
from torch.distributed.tensor import distribute_tensor, DTensor, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.components.distributed_muon import (
    BucketSpec,
    DistributedMuon,
)
from torchtitan.components.checkpoint_utils import (
    get_flat_optim_state_dict,
    init_optim_state,
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
            self._mesh = init_device_mesh(self.device_type, (self.world_size,))
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
    ) -> DistributedMuon:
        return DistributedMuon(
            [
                {
                    "params": [redistributed],
                    "param_names": ["layers.0.redistributed"],
                },
                {
                    "params": [local_blocks],
                    "param_names": ["layers.0.local_blocks"],
                    "matrix_shape": (2, 3),
                },
            ],
            bucket_spec=[
                BucketSpec(
                    patterns=("layers.0.*",),
                    owner_rank_by_fqn={"layers.0.redistributed": 1},
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
    def test_constructor_requires_exact_bucket_coverage_without_creating_state(self):
        redistributed = self._parameter(
            torch.arange(12, device=self.device).reshape(4, 3).float()
        )
        local_blocks = self._parameter(
            torch.arange(12, 24, device=self.device).reshape(4, 3).float()
        )

        optimizer = self._optimizer(redistributed, local_blocks)
        self.assertEqual(len(optimizer.state), 0)
        self.assertEqual(
            optimizer._plans[0].distributed_bindings[0].owner_rank,
            1,
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
            DistributedMuon(
                [
                    {
                        "params": [redistributed, local_blocks],
                        "param_names": [
                            "layers.0.redistributed",
                            "layers.0.local_blocks",
                        ],
                        "matrix_shape": (2, 3),
                    }
                ],
                bucket_spec=[
                    BucketSpec(
                        patterns=("*.redistributed",),
                        owner_rank_by_fqn={},
                    )
                ],
            )

        with self.assertRaisesRegex(ValueError, "must match one bucket"):
            DistributedMuon(
                [
                    {
                        "params": [redistributed, local_blocks],
                        "param_names": [
                            "layers.0.redistributed",
                            "layers.0.local_blocks",
                        ],
                        "matrix_shape": (2, 3),
                    }
                ],
                bucket_spec=[
                    BucketSpec(
                        patterns=("layers.0.*",),
                        owner_rank_by_fqn={},
                    ),
                    BucketSpec(
                        patterns=("*.local_blocks",),
                        owner_rank_by_fqn={},
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
            }
        ]

        with self.assertRaisesRegex(ValueError, "exactly cover"):
            DistributedMuon(
                params,
                bucket_spec=[
                    BucketSpec(
                        patterns=("layers.0.*",),
                        owner_rank_by_fqn={"layers.0.first": 0},
                    )
                ],
            )

        with self.assertRaisesRegex(ValueError, "outside its process group"):
            DistributedMuon(
                params,
                bucket_spec=[
                    BucketSpec(
                        patterns=("layers.0.*",),
                        owner_rank_by_fqn={
                            "layers.0.first": 0,
                            "layers.0.second": self.world_size,
                        },
                    )
                ],
            )

        with self.assertRaisesRegex(ValueError, "invalid DistributedMuon group"):
            DistributedMuon(
                [
                    {
                        "params": [first],
                        "param_names": ["layers.0.first"],
                        "matrix_block_dim": 1,
                    }
                ],
                bucket_spec=[
                    BucketSpec(
                        patterns=("layers.0.*",),
                        owner_rank_by_fqn={},
                    )
                ],
            )

        with self.assertRaisesRegex(ValueError, "whole-matrix-owned"):
            DistributedMuon(
                [
                    {
                        "params": [first],
                        "param_names": ["layers.0.first"],
                        "matrix_shape": (4, 1),
                        "matrix_block_dim": 1,
                    }
                ],
                bucket_spec=[
                    BucketSpec(
                        patterns=("layers.0.*",),
                        owner_rank_by_fqn={"layers.0.first": 0},
                    )
                ],
            )

    @with_comms
    def test_constructor_rejects_cross_rank_plan_mismatch(self):
        redistributed = self._parameter(
            torch.arange(12, device=self.device).reshape(4, 3).float()
        )
        with self.assertRaisesRegex(RuntimeError, "plans differ across ranks"):
            DistributedMuon(
                [
                    {
                        "params": [redistributed],
                        "param_names": ["layers.0.redistributed"],
                    }
                ],
                bucket_spec=[
                    BucketSpec(
                        patterns=("layers.0.*",),
                        owner_rank_by_fqn={"layers.0.redistributed": self.rank},
                    )
                ],
            )

        with self.assertRaisesRegex(RuntimeError, "plans differ across ranks"):
            DistributedMuon(
                [
                    {
                        "params": [redistributed],
                        "param_names": ["layers.0.redistributed"],
                    }
                ],
                bucket_spec=[
                    BucketSpec(
                        patterns=("layers.0.*",),
                        owner_rank_by_fqn={"layers.0.redistributed": 0},
                    )
                ],
                lr=0.01 if self.rank == 0 else 0.02,
            )

    @with_comms
    def test_mixed_owned_and_head_sharded_bucket_matches_plain_muon(self):
        owned_value = (
            torch.arange(12, device=self.device).reshape(4, 3).float().div_(10)
        )
        wo_value = (
            torch.arange(24, device=self.device).reshape(4, 6).float().div_(13)
        )
        owned = self._parameter(owned_value)
        wo = self._parameter(wo_value)

        def make_optimizer(
            owned_param: torch.nn.Parameter, wo_param: torch.nn.Parameter
        ) -> DistributedMuon:
            return DistributedMuon(
                [
                    {
                        "params": [owned_param],
                        "param_names": ["layers.0.owned"],
                    },
                    {
                        "params": [wo_param],
                        "param_names": ["layers.0.wo"],
                        "matrix_shape": (4, 2),
                        "matrix_block_dim": 1,
                    },
                ],
                bucket_spec=[
                    BucketSpec(
                        patterns=("layers.0.*",),
                        owner_rank_by_fqn={"layers.0.owned": 1},
                    )
                ],
                lr=0.03,
                weight_decay=0.2,
                momentum=0.8,
                nesterov=True,
                ns_steps=2,
            )

        optimizer = make_optimizer(owned, wo)
        plan = optimizer._plans[0]
        self.assertEqual(plan.input_split_sizes, [8, 10])
        self.assertEqual(
            plan.output_split_sizes,
            [8, 8] if self.mesh.get_local_rank() == 0 else [10, 10],
        )

        reference_owned = torch.nn.Parameter(owned_value.clone())
        reference_heads = tuple(
            torch.nn.Parameter(head.clone())
            for head in wo_value.unflatten(1, (3, 2)).movedim(1, 0)
        )
        reference = torch.optim.Muon(
            [reference_owned, *reference_heads],
            lr=0.03,
            weight_decay=0.2,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
        )

        first_owned_grad = (
            torch.arange(1, 13, device=self.device)
            .reshape(4, 3)
            .float()
            .div_(17)
        )
        first_wo_grad = (
            torch.arange(1, 25, device=self.device)
            .reshape(4, 6)
            .float()
            .div_(19)
        )
        all_to_all_single = dist.all_to_all_single
        rank = self.mesh.get_local_rank()
        buffer_signature = None
        for owned_grad, wo_grad in (
            (first_owned_grad, first_wo_grad),
            (
                first_owned_grad.flip(0).contiguous(),
                first_wo_grad.flip(0).contiguous(),
            ),
        ):
            owned.grad = distribute_tensor(
                owned_grad.clone(), self.mesh, (Shard(0),)
            )
            wo.grad = distribute_tensor(wo_grad.clone(), self.mesh, (Shard(0),))
            reference_owned.grad = owned_grad.clone()
            for head, head_grad in zip(
                reference_heads,
                wo_grad.unflatten(1, (3, 2)).movedim(1, 0),
                strict=True,
            ):
                head.grad = head_grad.clone()

            with patch(
                "torchtitan.components.distributed_muon.dist.all_to_all_single",
                wraps=all_to_all_single,
            ) as collective:
                optimizer.step()
            self.assertEqual(collective.call_count, 2)
            reference.step()

            expected_owned = reference_owned.detach().chunk(
                self.world_size, dim=0
            )[rank]
            expected_wo = (
                torch.stack([head.detach() for head in reference_heads])
                .movedim(0, 1)
                .contiguous()
                .view(4, 6)
                .chunk(self.world_size, dim=0)[rank]
            )
            torch.testing.assert_close(owned.to_local(), expected_owned)
            torch.testing.assert_close(wo.to_local(), expected_wo)

            expected_owned_momentum = reference.state[reference_owned][
                "momentum_buffer"
            ].chunk(self.world_size, dim=0)[rank]
            expected_wo_momentum = (
                torch.stack(
                    [
                        reference.state[head]["momentum_buffer"]
                        for head in reference_heads
                    ]
                )
                .movedim(0, 1)
                .contiguous()
                .view(4, 6)
                .chunk(self.world_size, dim=0)[rank]
            )
            torch.testing.assert_close(
                optimizer.state[owned]["momentum_buffer"].to_local(),
                expected_owned_momentum,
            )
            torch.testing.assert_close(
                optimizer.state[wo]["momentum_buffer"].to_local(),
                expected_wo_momentum,
            )
            for param in (owned, wo):
                momentum = optimizer.state[param]["momentum_buffer"]
                self.assertIsInstance(param, DTensor)
                self.assertIsInstance(momentum, DTensor)
                self.assertEqual(param.placements, (Shard(0),))
                self.assertEqual(momentum.placements, (Shard(0),))
                self.assertTrue(param.to_local().is_contiguous())
                self.assertTrue(momentum.to_local().is_contiguous())

            context = optimizer._communication_context
            assert context is not None
            current_signature = tuple(
                tuple(
                    (
                        storage_name,
                        tuple(
                            (str(key), tensor.data_ptr(), tensor.numel())
                            for key, tensor in storage.items()
                        ),
                    )
                    for storage_name, storage in (
                        ("local", slot.local_storage),
                        ("routed", slot.routed_storage),
                        ("compute", slot.compute_storage),
                    )
                )
                for slot in context.slots
            )
            if buffer_signature is None:
                buffer_signature = current_signature
            else:
                self.assertEqual(current_signature, buffer_signature)

        state_dict = optimizer.state_dict()
        resumed_owned = self._parameter(reference_owned.detach())
        resumed_wo_value = (
            torch.stack([head.detach() for head in reference_heads])
            .movedim(0, 1)
            .contiguous()
            .view(4, 6)
        )
        resumed_wo = self._parameter(resumed_wo_value)
        resumed = make_optimizer(resumed_owned, resumed_wo)
        resumed.load_state_dict(state_dict)
        self.assertEqual(resumed.param_groups[1]["matrix_block_dim"], 1)
        torch.testing.assert_close(
            resumed.state[resumed_owned]["momentum_buffer"].to_local(),
            optimizer.state[owned]["momentum_buffer"].to_local(),
        )
        torch.testing.assert_close(
            resumed.state[resumed_wo]["momentum_buffer"].to_local(),
            optimizer.state[wo]["momentum_buffer"].to_local(),
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
            "torchtitan.components.distributed_muon.dist.all_to_all_single",
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
            "torchtitan.components.distributed_muon.dist.all_to_all_single",
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
        optimizer = DistributedMuon(
            [
                {
                    "params": [distributed_0, distributed_2],
                    "param_names": [
                        "layers.0.redistributed",
                        "layers.2.redistributed",
                    ],
                },
                {
                    "params": [local_blocks],
                    "param_names": ["layers.1.local_blocks"],
                    "matrix_shape": (2, 3),
                },
            ],
            bucket_spec=[
                BucketSpec(
                    patterns=("layers.0.*",),
                    owner_rank_by_fqn={"layers.0.redistributed": 0},
                ),
                BucketSpec(
                    patterns=("layers.1.*",),
                    owner_rank_by_fqn={},
                ),
                BucketSpec(
                    patterns=("layers.2.*",),
                    owner_rank_by_fqn={"layers.2.redistributed": 0},
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
            "torchtitan.components.distributed_muon.dist.all_to_all_single",
            wraps=all_to_all_single,
        ) as collective:
            optimizer.step()
        reference.step()

        self.assertEqual(collective.call_count, 4)
        torch.testing.assert_close(
            distributed_0.to_local(), references[0].chunk(self.world_size, dim=0)[rank]
        )
        torch.testing.assert_close(local_blocks.to_local(), references[1])
        torch.testing.assert_close(
            distributed_2.to_local(), references[2].chunk(self.world_size, dim=0)[rank]
        )


if __name__ == "__main__":
    unittest.main()
