# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for DistMuon DP x TP layouts and pipeline transport creation."""

from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.optim.optimizer import (
    DistMuon as ConfigurableDistMuon,
    OptimizersContainer,
)
from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
    Owned,
)
from torchtitan.distributed.flex_shard.dist_muon import (
    _adjust_muon_learning_rate,
    DistMuon,
)
from torchtitan.distributed.utils import build_transport_meshes


class TestDistMuonTensorParallelCPU(DTensorTestBase):
    mesh_shape = (2, 2)

    @property
    def world_size(self):
        return 4

    @property
    def device_type(self):
        return "cpu"

    @with_comms
    def test_owned_rejects_omitted_sharded_axis(self):
        mesh = init_device_mesh(
            self.device_type, self.mesh_shape, mesh_dim_names=("dp_shard", "tp")
        )
        parameter = torch.nn.Parameter(
            distribute_tensor(torch.ones(4, 4), mesh, (Shard(0), Shard(1)))
        )
        with self.assertRaisesRegex(ValueError, "non-replicated axes:.*tp"):
            DistMuon(
                [{"params": [parameter], "param_names": ["matrix"]}],
                compute_sharding_by_fqn={
                    "matrix": ComputeLayout(
                        shardings_by_mesh_axis={"dp_shard": Owned()}
                    )
                },
                bucket_configs=(BucketConfig(patterns=("matrix",)),),
            )

    @with_comms
    def test_mixed_block_row_scales_follow_sequential_and_strided_storage(self):
        mesh = init_device_mesh(
            self.device_type, self.mesh_shape, mesh_dim_names=("dp_shard", "tp")
        )
        block_sizes = (4, 2)
        reference_ratio = _adjust_muon_learning_rate(1.0, None, (6, 3))
        expected = torch.tensor(
            [
                _adjust_muon_learning_rate(1.0, None, (rows, 3)) / reference_ratio
                for rows in block_sizes * 2
                for _ in range(rows)
            ]
        ).reshape(12, 1)
        values = torch.arange(36).reshape(12, 3).float().div_(41)
        gradient = values.mul(0.37).add(0.2).sin()
        lr = 0.03
        (transport_mesh,) = build_transport_meshes(
            [(mesh, ("dp_shard", "tp"))], max_ctas=None
        )
        for storage_placements in (
            (Shard(0), Shard(0)),
            (_StridedShard(0, split_factor=2), Shard(0)),
        ):
            parameter = torch.nn.Parameter(
                distribute_tensor(values.clone(), mesh, storage_placements)
            )
            optimizer = DistMuon(
                [{"params": [parameter], "param_names": ["matrix"]}],
                compute_sharding_by_fqn={
                    "matrix": ComputeLayout(
                        shardings_by_mesh_axis={
                            "dp_shard": BlockShard(0, block_sizes),
                            "tp": BlockShard(0, block_sizes),
                        },
                        compute_dtype=torch.bfloat16,
                    )
                },
                bucket_configs=(
                    BucketConfig(patterns=("matrix",), mesh=transport_mesh),
                ),
                lr=lr,
                weight_decay=0,
                momentum=0,
                nesterov=False,
                ns_steps=2,
            )
            expected_local = distribute_tensor(
                expected.clone(), mesh, storage_placements
            ).to_local()
            torch.testing.assert_close(
                optimizer._direction_row_scale_by_fqn["matrix"],
                expected_local,
                rtol=0,
                atol=0,
            )
            parameter.grad = distribute_tensor(
                gradient.clone(), mesh, storage_placements
            )
            reference_parameters = [
                torch.nn.Parameter(block.clone())
                for block in values.split(block_sizes * 2)
            ]
            reference_optimizer = torch.optim.Muon(
                reference_parameters,
                lr=lr,
                weight_decay=0,
                momentum=0,
                nesterov=False,
                ns_steps=2,
            )
            for reference, grad in zip(
                reference_parameters, gradient.split(block_sizes * 2), strict=True
            ):
                reference.grad = grad.clone()
            optimizer.step()
            reference_optimizer.step()
            expected_parameter = torch.cat(
                [reference.detach() for reference in reference_parameters]
            )
            # Redistribution scales directions in BF16 before the FP32 add;
            # plain Muon applies its scalar alpha without that extra rounding.
            # Compare normalized updates, not near-equal resident weights.
            torch.testing.assert_close(
                (parameter.full_tensor() - values) / lr,
                (expected_parameter - values) / lr,
                rtol=0,
                atol=torch.finfo(torch.bfloat16).eps,
            )

    @with_comms
    def test_compute_ready_block_shard_accepts_empty_local_storage(self):
        mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp_shard",)
        )
        values = torch.arange(18).reshape(6, 3).float().div_(13)
        gradient = values.mul(0.37).add(0.2)
        parameter = torch.nn.Parameter(
            distribute_tensor(values.clone(), mesh, (Shard(0),))
        )
        parameter.grad = distribute_tensor(gradient.clone(), mesh, (Shard(0),))
        optimizer = DistMuon(
            [{"params": [parameter], "param_names": ["matrix"]}],
            compute_sharding_by_fqn={
                "matrix": ComputeLayout(
                    shardings_by_mesh_axis={"dp_shard": BlockShard(0, (2,))},
                    compute_dtype=torch.bfloat16,
                )
            },
            bucket_configs=(BucketConfig(patterns=("matrix",)),),
            lr=0.03,
            weight_decay=0,
            momentum=0,
            nesterov=False,
        )
        reference_parameters = [
            torch.nn.Parameter(block.clone()) for block in values.split(2)
        ]
        reference_optimizer = torch.optim.Muon(
            reference_parameters, lr=0.03, weight_decay=0, momentum=0, nesterov=False
        )
        for reference, grad in zip(
            reference_parameters, gradient.split(2), strict=True
        ):
            reference.grad = grad.clone()
        optimizer.step()
        reference_optimizer.step()
        expected = torch.cat([reference.detach() for reference in reference_parameters])
        torch.testing.assert_close(parameter.full_tensor(), expected, rtol=0, atol=0)
        if self.rank == 3:
            self.assertEqual(parameter.to_local().shape, (0, 3))


class TestDistMuonUnevenPipelineCPU(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    @property
    def device_type(self):
        return "cpu"

    @with_comms
    def test_uneven_pipeline_transport_creation_and_muon_parity(self):
        """Stages with different buckets and storage meshes create every group.

        Stage 0 holds only a dense matrix on (dp_shard, tp); stage 1 holds only
        experts on (edp_shard, ep), so neither has the other's storage mesh.
        """
        world_mesh = init_device_mesh(
            "cpu", (self.world_size,), mesh_dim_names=("world",)
        )
        dense_mesh = world_mesh._unflatten(
            0, (2, 2, 2), mesh_dim_names=("pp", "dp_shard", "tp")
        )["dp_shard", "tp"]
        expert_mesh = world_mesh._unflatten(
            0, (2, 2, 2), mesh_dim_names=("pp", "edp_shard", "ep")
        )["edp_shard", "ep"]
        name = "dense" if self.rank < 4 else "experts"
        mesh = dense_mesh if name == "dense" else expert_mesh
        value = (
            torch.arange(16).reshape(4, 4).float().div_(32)
            if name == "dense"
            else torch.arange(24).reshape(4, 3, 2).float().div_(32)
        )
        placements = (Shard(0), Shard(1))
        parameter = torch.nn.Parameter(
            distribute_tensor(value.clone(), mesh, placements)
        )
        model = torch.nn.Module()
        model.register_parameter(name, parameter)
        muon = ConfigurableDistMuon.Config(
            pattern=".*",
            compute_sharding_by_fqn={
                "dense": ComputeLayout(
                    {"dp_shard": Owned(), "tp": Owned()}, compute_dtype=torch.bfloat16
                ),
                "experts": ComputeLayout(
                    {"edp_shard": Shard(0), "ep": Shard(0)},
                    compute_dtype=torch.bfloat16,
                ),
            },
            bucket_configs=[
                BucketConfig(patterns=("dense",)),
                BucketConfig(patterns=("experts",)),
            ],
            lr=0.03,
            weight_decay=0.2,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
        )
        optimizer = OptimizersContainer.Config(optimizers=[muon]).build(
            model_parts=[model]
        )
        # Nonmember ranks can skip native first-use group creation without an
        # immediate Gloo hang. The next WORLD group exposes the desynchronization.
        dist.barrier()
        group = dist.new_group(
            ranks=list(range(self.world_size)), timeout=timedelta(seconds=5)
        )
        total = torch.ones(1)
        dist.all_reduce(total, group=group)
        self.assertEqual(total.item(), self.world_size)
        reference_parameters = [
            torch.nn.Parameter(matrix.clone())
            for matrix in (value.unsqueeze(0) if name == "dense" else value)
        ]
        reference = torch.optim.Muon(
            reference_parameters,
            lr=muon.lr,
            weight_decay=muon.weight_decay,
            momentum=muon.momentum,
            nesterov=muon.nesterov,
            ns_steps=muon.ns_steps,
        )
        for step in range(2):
            gradient = value.mul(0.37).add(0.2 + step).sin()
            parameter.grad = distribute_tensor(gradient.clone(), mesh, placements)
            for ref, grad in zip(
                reference_parameters,
                gradient.unsqueeze(0) if name == "dense" else gradient,
                strict=True,
            ):
                ref.grad = grad.clone()
            optimizer.step()
            reference.step()
            expected = (
                reference_parameters[0].detach()
                if name == "dense"
                else torch.stack([ref.detach() for ref in reference_parameters])
            )
            torch.testing.assert_close(
                parameter.full_tensor(), expected, rtol=0, atol=0
            )
        dist.destroy_process_group(group)
