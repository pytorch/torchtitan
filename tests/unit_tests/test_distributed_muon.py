# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

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
    MuonComputeSharding,
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
        ):
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
                                num_matrices=num_matrices,
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
            local_blocks: torch.nn.Parameter,
            redistributed_grad: torch.Tensor,
            local_blocks_grad: torch.Tensor,
        ) -> None:
            redistributed.grad = distribute_tensor(
                redistributed_grad.clone(), mesh, (Shard(0),)
            )
            local_blocks.grad = distribute_tensor(
                local_blocks_grad.clone(), mesh, (Shard(0),)
            )

        def assert_matches_reference(
            optimizer,
            redistributed: torch.nn.Parameter,
            local_blocks: torch.nn.Parameter,
            reference_optimizer: torch.optim.Muon,
            reference_redistributed: torch.nn.Parameter,
            reference_local_blocks: tuple[torch.nn.Parameter, ...],
            local_blocks_before: torch.Tensor,
            reference_local_blocks_before: tuple[torch.Tensor, ...],
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

            expected_local_blocks = torch.cat(
                [parameter.detach() for parameter in reference_local_blocks], dim=0
            )
            expected_local_blocks_before = torch.cat(
                reference_local_blocks_before, dim=0
            )
            local_rows, row_offset = Shard.local_shard_size_and_offset(
                expected_local_blocks.shape[0], self.world_size, rank
            )
            expected_local_blocks = expected_local_blocks.narrow(
                0, row_offset, local_rows
            )
            expected_local_blocks_before = expected_local_blocks_before.narrow(
                0, row_offset, local_rows
            )
            decay = 1 - lr * weight_decay
            adjusted_lr = _adjust_muon_learning_rate(
                lr, None, reference_local_blocks[0].shape
            )
            actual_update = (
                local_blocks_before * decay - local_blocks.to_local()
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

            for parameter in (redistributed, local_blocks):
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

            local_blocks_momentum = optimizer.state[local_blocks]["momentum_buffer"]
            expected_local_blocks_momentum = torch.cat(
                [
                    reference_optimizer.state[parameter]["momentum_buffer"]
                    for parameter in reference_local_blocks
                ],
                dim=0,
            ).narrow(0, row_offset, local_rows)
            torch.testing.assert_close(
                local_blocks_momentum.to_local(),
                expected_local_blocks_momentum,
                rtol=0,
                atol=0,
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

        first_redistributed_grad = (
            torch.arange(1, 13, device=device).reshape(4, 3).float().div_(17)
        )
        first_local_blocks_grad = (
            torch.arange(13, 49, device=device)
            .reshape(num_matrices * matrix_rows, 3)
            .float()
            .div_(19)
        )
        local_blocks_before = local_blocks.to_local().clone()
        reference_local_blocks_before = tuple(
            parameter.detach().clone() for parameter in reference_local_blocks
        )
        set_grads(
            redistributed,
            local_blocks,
            first_redistributed_grad,
            first_local_blocks_grad,
        )
        reference_redistributed.grad = first_redistributed_grad.clone()
        for parameter, grad in zip(
            reference_local_blocks,
            first_local_blocks_grad.view(num_matrices, matrix_rows, 3),
            strict=True,
        ):
            parameter.grad = grad.clone()

        optimizer.step()
        reference_optimizer.step()
        assert_matches_reference(
            optimizer,
            redistributed,
            local_blocks,
            reference_optimizer,
            reference_redistributed,
            reference_local_blocks,
            local_blocks_before,
            reference_local_blocks_before,
        )

        flat_state_dict = get_flat_optim_state_dict(optimizer)
        resumed_redistributed = make_parameter(redistributed.full_tensor().detach())
        resumed_local_blocks = make_parameter(local_blocks.full_tensor().detach())
        resumed_optimizer = make_optimizer(
            resumed_redistributed,
            resumed_local_blocks,
        )
        init_optim_state(resumed_optimizer)
        load_flat_optim_state_dict(resumed_optimizer, flat_state_dict)

        second_redistributed_grad = first_redistributed_grad.flip(0).contiguous()
        second_local_blocks_grad = first_local_blocks_grad.flip(0).contiguous()
        local_blocks_before = resumed_local_blocks.to_local().clone()
        reference_local_blocks_before = tuple(
            parameter.detach().clone() for parameter in reference_local_blocks
        )
        set_grads(
            resumed_redistributed,
            resumed_local_blocks,
            second_redistributed_grad,
            second_local_blocks_grad,
        )
        reference_redistributed.grad = second_redistributed_grad.clone()
        for parameter, grad in zip(
            reference_local_blocks,
            second_local_blocks_grad.view(num_matrices, matrix_rows, 3),
            strict=True,
        ):
            parameter.grad = grad.clone()

        resumed_optimizer.step()
        reference_optimizer.step()
        assert_matches_reference(
            resumed_optimizer,
            resumed_redistributed,
            resumed_local_blocks,
            reference_optimizer,
            reference_redistributed,
            reference_local_blocks,
            local_blocks_before,
            reference_local_blocks_before,
        )


if __name__ == "__main__":
    unittest.main()
