# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import mock

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Shard
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
)
from torchtitan.components.distributed_optimizers.muon import (
    BatchedMatrixComputeView,
    build_distributed_muon,
    distributed_muon as distributed_muon_module,
    MuonComputeShardingConfig,
    Owned,
)
from torchtitan.components.distributed_optimizers.muon.distributed_muon import (
    _adjust_muon_learning_rate,
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
            "layers.0.oversharded": (3, 4, 3),
            # These aligned siblings remain local and share one batched NS call.
            "layers.0.local.w1": (4, 5, 3),
            "layers.0.local.w3": (4, 5, 3),
        }

        def make_parameter(value: torch.Tensor) -> torch.nn.Parameter:
            return torch.nn.Parameter(
                distribute_tensor(value.clone(), mesh, (Shard(0),))
            )

        def make_optimizer(
            redistributed: torch.nn.Parameter,
            stacks: dict[str, torch.nn.Parameter],
        ):
            return build_distributed_muon(
                [
                    {
                        "params": [redistributed],
                        "param_names": ["layers.0.redistributed"],
                        "compute_sharding": MuonComputeShardingConfig(
                            placement=Owned()
                        ),
                    },
                    {
                        "params": [stacks["layers.0.oversharded"]],
                        "param_names": ["layers.0.oversharded"],
                        "compute_sharding": MuonComputeShardingConfig(
                            view_before_placement=BatchedMatrixComputeView(
                                num_matrices=3,
                            ),
                            placement=Shard(0),
                        ),
                    },
                    {
                        "params": [
                            stacks["layers.0.local.w1"],
                            stacks["layers.0.local.w3"],
                        ],
                        "param_names": [
                            "layers.0.local.w1",
                            "layers.0.local.w3",
                        ],
                        "compute_sharding": MuonComputeShardingConfig(
                            view_before_placement=BatchedMatrixComputeView(
                                num_matrices=4,
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
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.8,
                nesterov=True,
                ns_steps=2,
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
            optimizer,
            redistributed: torch.nn.Parameter,
            stacks: dict[str, torch.nn.Parameter],
            reference_optimizer: torch.optim.Muon,
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

                momentum = optimizer.state[parameter]["momentum_buffer"]
                expected_momentum = torch.cat(
                    [
                        reference_optimizer.state[reference]["momentum_buffer"]
                        for reference in reference_blocks
                    ],
                    dim=0,
                ).narrow(0, row_offset, local_rows)
                torch.testing.assert_close(
                    momentum.to_local(),
                    expected_momentum,
                    rtol=0,
                    atol=0,
                )

            for parameter in (redistributed, *stacks.values()):
                self.assertIsInstance(parameter, DTensor)
                self.assertEqual(parameter.placements, (Shard(0),))

            redistributed_momentum = optimizer.state[redistributed]["momentum_buffer"]
            expected_redistributed_momentum = reference_optimizer.state[
                reference_redistributed
            ]["momentum_buffer"].chunk(self.world_size, dim=0)[rank]
            torch.testing.assert_close(
                redistributed_momentum.to_local(),
                expected_redistributed_momentum,
                rtol=0,
                atol=0,
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

            runtime = current_optimizer._redistribution_runtime
            self.assertIsNotNone(runtime._context)
            reserved = runtime._context.slots[0].buffers.buffers[
                (device, torch.float32)
            ]
            scratch = reserved.compute_scratch
            self.assertIsNotNone(scratch)
            expected_batch_numel = sum(
                current_stacks[name].to_local().numel()
                for name in ("layers.0.local.w1", "layers.0.local.w3")
            )
            self.assertGreaterEqual(scratch.numel(), expected_batch_numel)
            scratch_data_ptr = scratch.data_ptr()
            with mock.patch.object(
                distributed_muon_module,
                "_compute_muon_direction",
                wraps=distributed_muon_module._compute_muon_direction,
            ) as compute_muon_direction:
                current_optimizer.step()
            reference_optimizer.step()
            self.assertIs(reserved.compute_scratch, scratch)
            self.assertEqual(reserved.compute_scratch.data_ptr(), scratch_data_ptr)
            self.assertEqual(
                [
                    tuple(call.args[0].shape)
                    for call in compute_muon_direction.call_args_list
                    if tuple(call.args[0].shape[-2:]) == (5, 3)
                ],
                [(4, 5, 3)],
            )
            assert_matches_reference(
                current_optimizer,
                current_redistributed,
                current_stacks,
                reference_optimizer,
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
        first_stack_grads["layers.0.local.w3"] = (
            first_stack_grads["layers.0.local.w3"].flip(1).contiguous()
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
        resumed_optimizer = make_optimizer(resumed_redistributed, resumed_stacks)
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


if __name__ == "__main__":
    unittest.main()
