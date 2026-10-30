# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Real CPU DP/TP routing checks shared with the CUDA transport tests."""

from unittest import mock

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    build_transport_mesh,
    ComputeLayout,
    Owned,
    redistribution_mesh_axes,
)
from torchtitan.distributed.flex_shard.dist_muon import (
    _adjust_muon_learning_rate,
    DistMuon,
)


class TensorParallelParity:
    """Shared exact routing checks for CPU and CUDA TP process groups."""

    def _assert_storage_update(self, *, capture_graph: bool = False):
        """Check whole matrices, head blocks, and expert matrices exactly once."""
        torch.manual_seed(42)
        mesh = init_device_mesh(
            self.device_type,
            self.mesh_shape,
            mesh_dim_names=("dp_shard", "tp"),
        )
        device = torch.device(self.device_type, self.rank)
        lr = 0.03
        weight_decay = 0.2
        matrix_rows = 2

        values = {
            "layers.0.per_head": (
                torch.arange(24, device=device).reshape(8, 3).float().div_(13)
            ),
            "layers.0.whole": (
                torch.arange(16, device=device).reshape(4, 4).float().div_(11)
            ),
            "layers.0.experts": (
                torch.arange(24, device=device).reshape(4, 3, 2).float().div_(7)
            ),
        }
        placements = {
            "layers.0.per_head": (
                _StridedShard(0, split_factor=mesh["tp"].size()),
                Shard(0),
            ),
            "layers.0.whole": (Shard(0), Shard(1)),
            "layers.0.experts": (Shard(0), Shard(1)),
        }
        parameters = {
            fqn: torch.nn.Parameter(
                distribute_tensor(value.clone(), mesh, placements[fqn])
            )
            for fqn, value in values.items()
        }
        gradients = {
            fqn: value.clone().mul_(0.37).add_(0.2).sin_()
            for fqn, value in values.items()
        }
        for fqn, parameter in parameters.items():
            parameter.grad = distribute_tensor(
                gradients[fqn].clone(),
                mesh,
                placements[fqn],
            )

        transport_mesh = build_transport_mesh(
            mesh,
            redistribution_mesh_axes(
                parameters["layers.0.whole"],
                ComputeLayout(
                    shardings_by_mesh_axis={"dp_shard": Owned(), "tp": Owned()}
                ),
            ),
        )
        optimizer = DistMuon(
            [
                {
                    "params": list(parameters.values()),
                    "param_names": list(parameters),
                }
            ],
            compute_sharding_by_fqn={
                "layers.0.per_head": ComputeLayout(
                    shardings_by_mesh_axis={
                        "dp_shard": BlockShard(0, (matrix_rows,)),
                        "tp": BlockShard(0, (matrix_rows,)),
                    },
                    compute_dtype=torch.bfloat16,
                ),
                "layers.0.whole": ComputeLayout(
                    shardings_by_mesh_axis={
                        "dp_shard": Owned(),
                        "tp": Owned(),
                    },
                    compute_dtype=torch.bfloat16,
                ),
                "layers.0.experts": ComputeLayout(
                    shardings_by_mesh_axis={
                        "dp_shard": Shard(0),
                        "tp": Shard(0),
                    },
                    compute_dtype=torch.bfloat16,
                ),
            },
            bucket_configs=[
                BucketConfig(
                    patterns=("layers.0.per_head", "layers.0.whole"),
                    mesh=transport_mesh,
                ),
                BucketConfig(patterns=("layers.0.experts",)),
            ],
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.0,
            nesterov=False,
            ns_steps=2,
        )
        transport_group = transport_mesh.get_group()
        plan = optimizer._bucket_plans[0]
        self.assertIs(plan.group.process_group, transport_group)
        self.assertIs(plan.storage_to_compute_schedule.process_group, transport_group)
        self.assertIs(plan.compute_to_storage_schedule.process_group, transport_group)

        captured_compute_by_fqn: dict[str, torch.Tensor] = {}

        def matrix_dependent_direction(tensor, matrix_shape):
            matrices = tensor.reshape(-1, *matrix_shape)
            return matrices.cumsum(dim=-2).cumsum(dim=-1).reshape_as(tensor)

        def make_direction(compute_layout, compute):
            matrix_shape = (
                (matrix_rows, compute_layout.param.shape[-1])
                if type(compute_layout.compute_sharding) is BlockShard
                else tuple(compute_layout.param.shape[-2:])
            )
            # Match the NS conversion even when the local scratch is FP32.
            prepared = compute.to(torch.bfloat16)
            captured_compute_by_fqn[compute_layout.fqn] = prepared.detach().clone()
            compute.copy_(matrix_dependent_direction(prepared, matrix_shape))

        with mock.patch.object(
            optimizer,
            "_compute_update",
            side_effect=make_direction,
        ):
            if capture_graph:
                optimizer.step()
                for fqn, parameter in parameters.items():
                    original = distribute_tensor(
                        values[fqn].clone(), mesh, placements[fqn]
                    )
                    with torch.no_grad():
                        parameter.to_local().copy_(original.to_local())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    optimizer.step()
                graph.replay()
            else:
                optimizer.step()

        compute_shapes = {
            "layers.0.per_head": (matrix_rows, values["layers.0.per_head"].shape[1]),
            "layers.0.whole": values["layers.0.whole"].shape,
            "layers.0.experts": values["layers.0.experts"].shape[1:],
        }
        for fqn, matrix_shape in compute_shapes.items():
            expected_matrices = (
                gradients[fqn].to(torch.bfloat16).reshape(-1, *matrix_shape)
            )
            matrix_match_counts = torch.zeros(
                expected_matrices.shape[0],
                dtype=torch.int64,
                device=device,
            )
            captured_compute = captured_compute_by_fqn.get(fqn)
            if captured_compute is not None and captured_compute.numel():
                captured_matrices = captured_compute.reshape(-1, *matrix_shape)
                matches = (
                    captured_matrices.flatten(1)[:, None]
                    == expected_matrices.flatten(1)[None]
                ).all(dim=-1)
                self.assertTrue(matches.any(dim=1).all())
                matrix_match_counts.add_(matches.sum(dim=0))
            dist.all_reduce(matrix_match_counts)
            torch.testing.assert_close(
                matrix_match_counts,
                torch.ones_like(matrix_match_counts),
            )

        for fqn, parameter in parameters.items():
            expected = values[fqn].mul(1 - lr * weight_decay)
            expected_direction = matrix_dependent_direction(
                gradients[fqn].to(torch.bfloat16), compute_shapes[fqn]
            ).to(values[fqn].dtype)
            expected.add_(
                expected_direction,
                alpha=-_adjust_muon_learning_rate(lr, None, compute_shapes[fqn]),
            )
            torch.testing.assert_close(
                parameter.full_tensor(),
                expected,
                rtol=0,
                atol=0,
            )


class TestDistMuonTensorParallelCPU(DTensorTestBase, TensorParallelParity):
    mesh_shape = (2, 2)

    @property
    def world_size(self):
        return 4

    @property
    def device_type(self):
        return "cpu"

    @with_comms
    def test_dp_tp_storage_matches_unsharded_update(self):
        self._assert_storage_update()

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
        for storage_placements in (
            (Shard(0), Shard(0)),
            (_StridedShard(0, split_factor=2), Shard(0)),
        ):
            parameter = torch.nn.Parameter(
                distribute_tensor(torch.ones(12, 3), mesh, storage_placements)
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
                bucket_configs=(BucketConfig(patterns=("matrix",)),),
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
        with mock.patch.object(
            optimizer, "_compute_update", side_effect=lambda layout, x: x.mul_(0.5)
        ):
            optimizer.step()
        expected = values.add(gradient.mul(0.5), alpha=-0.03)
        torch.testing.assert_close(parameter.full_tensor(), expected, rtol=0, atol=0)
        if self.rank == 3:
            self.assertEqual(parameter.to_local().shape, (0, 3))
