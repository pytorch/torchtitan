# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Real CPU DP/TP routing checks shared with the CUDA transport tests."""

from dataclasses import replace
from datetime import timedelta
from unittest import mock

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.optim.optimizer import (
    DistMuon as ConfigurableDistMuon,
    OptimizersContainer,
)
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
    Owned,
    redistribution_mesh_axes,
)
from torchtitan.distributed.flex_shard.dist_muon import (
    _adjust_muon_learning_rate,
    DistMuon,
)
from torchtitan.distributed.utils import build_transport_meshes
from torchtitan.models.kimi_k2_7.qk_clip import _scale_mla_heads
from torchtitan_recipes.tests.models.kimi_k2_7 import _per_expert_compute_layout


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
                torch.arange(36, device=device).reshape(12, 3).float().div_(13)
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

        transport_axes = redistribution_mesh_axes(
            "layers.0.whole",
            parameters["layers.0.whole"],
            ComputeLayout(shardings_by_mesh_axis={"dp_shard": Owned(), "tp": Owned()}),
        )
        (transport_mesh,) = build_transport_meshes(
            [(mesh, transport_axes)], max_ctas=None, storage_meshes={mesh}
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
                "layers.0.experts": _per_expert_compute_layout(
                    ParallelismConfig(
                        data_parallel_shard_degree=mesh["dp_shard"].size(),
                        tensor_parallel_degree=mesh["tp"].size(),
                        expert_parallel_degree=1,
                    )
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
    def test_dp_tp_matches_plain_muon(self):
        """Real redistribution and NS preserve whole, mixed-block and expert matrices."""
        mesh = init_device_mesh(
            self.device_type, self.mesh_shape, mesh_dim_names=("dp_shard", "tp")
        )
        values = {
            "whole": torch.arange(16).reshape(4, 4).float().div_(32),
            "blocks": torch.arange(96).reshape(12, 8).float().div_(128),
            "experts": torch.arange(24).reshape(4, 3, 2).float().div_(32),
        }
        placements = {
            "whole": (Shard(0), Shard(1)),
            "blocks": (_StridedShard(0, split_factor=2), Shard(0)),
            "experts": (Shard(0), Shard(1)),
        }
        parameters = {
            name: torch.nn.Parameter(
                distribute_tensor(value.clone(), mesh, placements[name])
            )
            for name, value in values.items()
        }
        settings = dict(
            compute_sharding_by_fqn={
                "whole": ComputeLayout(
                    {"dp_shard": Owned(), "tp": Owned()}, compute_dtype=torch.bfloat16
                ),
                "blocks": ComputeLayout(
                    {"dp_shard": BlockShard(0, (4, 2)), "tp": BlockShard(0, (4, 2))},
                    compute_dtype=torch.bfloat16,
                ),
                "experts": ComputeLayout(
                    {"dp_shard": Shard(0), "tp": Shard(0)}, compute_dtype=torch.bfloat16
                ),
            },
            bucket_configs=[
                BucketConfig(patterns=("whole", "blocks")),
                BucketConfig(patterns=("experts",)),
            ],
            lr=0.03,
            weight_decay=0.2,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
        )
        (transport_mesh,) = build_transport_meshes(
            [(mesh, ("dp_shard", "tp"))], max_ctas=None, storage_meshes={mesh}
        )
        param_groups = [
            {"params": list(parameters.values()), "param_names": list(parameters)}
        ]
        # FlexShard consumes caller-created groups; neither the missing-mesh
        # error nor explicit construction may create a communicator implicitly.
        with mock.patch.object(
            DeviceMesh,
            "_flatten",
            side_effect=AssertionError("caller owns group setup"),
        ):
            with self.assertRaisesRegex(ValueError, "explicit BucketConfig.mesh"):
                DistMuon(param_groups, **settings)
            settings["bucket_configs"][0] = replace(
                settings["bucket_configs"][0], mesh=transport_mesh
            )
            optimizer = DistMuon(param_groups, **settings)
        matrix_rows = (4, 2, 4, 2)
        references = {
            "whole": [torch.nn.Parameter(values["whole"].clone())],
            "blocks": [
                torch.nn.Parameter(block.clone())
                for block in values["blocks"].split(matrix_rows)
            ],
            "experts": [
                torch.nn.Parameter(matrix.clone()) for matrix in values["experts"]
            ],
        }
        reference_optimizer = torch.optim.Muon(
            [parameter for group in references.values() for parameter in group],
            lr=0.03,
            weight_decay=0.2,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
        )
        for step in range(2):
            gradients = {
                name: value.mul(0.37).add(0.2 + step).sin()
                for name, value in values.items()
            }
            reference_gradients = {
                "whole": [gradients["whole"]],
                "blocks": gradients["blocks"].split(matrix_rows),
                "experts": gradients["experts"].unbind(),
            }
            for name, parameter in parameters.items():
                parameter.grad = distribute_tensor(
                    gradients[name].clone(), mesh, placements[name]
                )
                for reference, gradient in zip(
                    references[name], reference_gradients[name], strict=True
                ):
                    reference.grad = gradient.clone()
            optimizer.step()
            reference_optimizer.step()
            expected = {
                "whole": references["whole"][0].detach(),
                "blocks": torch.cat(
                    [parameter.detach() for parameter in references["blocks"]]
                ),
                "experts": torch.stack(
                    [parameter.detach() for parameter in references["experts"]]
                ),
            }
            for name, parameter in parameters.items():
                torch.testing.assert_close(
                    parameter.full_tensor(), expected[name], rtol=0, atol=0
                )

    @with_comms
    def test_qk_clip_uses_tp_local_heads_with_fsdp_split_rows(self):
        mesh = init_device_mesh(
            self.device_type, self.mesh_shape, mesh_dim_names=("dp_shard", "tp")
        )
        values = torch.arange(48).float().reshape(24, 2)
        weight = distribute_tensor(
            values.clone(), mesh, (_StridedShard(0, split_factor=2), Shard(0))
        )
        scales_H = torch.tensor([0.04, 0.16, 0.36, 0.64, 0.81, 1.0])
        local_scales_H = scales_H.chunk(mesh["tp"].size())[mesh["tp"].get_local_rank()]
        # Three TP-local heads have twelve rows; FSDP's six-row shard splits
        # a head. Scale logical rows rather than reshaping the local weight.
        for remaining_exponent in (0.0, 1.0):
            weight.to_local().copy_(
                distribute_tensor(values.clone(), mesh, weight.placements).to_local()
            )
            _scale_mla_heads(
                weight,
                local_scales_H,
                rows_per_head=4,
                nope_rows_per_head=2,
                nope_scale_exponent=0.5,
                remaining_scale_exponent=remaining_exponent,
            )
            expected_scales_R1 = torch.cat(
                [
                    scales_H.pow(0.5)[:, None].expand(-1, 2),
                    scales_H.pow(remaining_exponent)[:, None].expand(-1, 2),
                ],
                dim=1,
            ).reshape(-1, 1)
            torch.testing.assert_close(
                weight.full_tensor(), values * expected_scales_R1, rtol=0, atol=0
            )

        for invalid_values, head_scales, placements in (
            (torch.tensor(1.0), local_scales_H, (Replicate(), Replicate())),
            (values[:, 0].contiguous(), local_scales_H, weight.placements),
            (values.unsqueeze(-1), local_scales_H, weight.placements),
            (values, local_scales_H[:-1], weight.placements),
        ):
            with self.subTest(
                weight_shape=tuple(invalid_values.shape), num_heads=head_scales.numel()
            ):
                invalid_weight = distribute_tensor(invalid_values, mesh, placements)
                with self.assertRaisesRegex(
                    ValueError, "QK clip scales do not match the MLA weight shape"
                ):
                    _scale_mla_heads(
                        invalid_weight,
                        head_scales,
                        rows_per_head=4,
                        nope_rows_per_head=2,
                        nope_scale_exponent=0.5,
                        remaining_scale_exponent=1.0,
                    )

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
            [(mesh, ("dp_shard", "tp"))], max_ctas=None, storage_meshes={mesh}
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
    def test_missing_transport_view_fails_collectively(self):
        """An expert-free stage rejects unavailable views before group creation."""
        root = init_device_mesh("cpu", (8,), mesh_dim_names=("world",))
        dense_mesh = root._unflatten(
            0, (2, 2, 2), mesh_dim_names=("pp", "dp_shard", "tp")
        )["dp_shard", "tp"]
        expert_mesh = root._unflatten(
            0, (2, 2, 2), mesh_dim_names=("pp", "edp_shard", "ep")
        )["edp_shard", "ep"]
        mesh = dense_mesh if self.rank < 4 else expert_mesh
        axes = mesh.mesh_dim_names
        parameter = torch.nn.Parameter(
            distribute_tensor(torch.ones(4, 4), mesh, (Shard(0), Shard(1)))
        )
        with self.assertRaisesRegex(ValueError, "have no storage mesh on some ranks"):
            ConfigurableDistMuon.Config(
                pattern=".*",
                compute_sharding_by_fqn={
                    "matrix": ComputeLayout(
                        {axis: Owned() for axis in axes},
                        compute_dtype=torch.bfloat16,
                    )
                },
                bucket_configs=[BucketConfig(patterns=("matrix",))],
            ).build(params=[{"params": [parameter], "param_names": ["matrix"]}])
        group = dist.new_group(
            ranks=list(range(self.world_size)), timeout=timedelta(seconds=5)
        )
        total = torch.ones(1)
        dist.all_reduce(total, group=group)
        self.assertEqual(total.item(), self.world_size)
        dist.destroy_process_group(group)

    @with_comms
    def test_uneven_pipeline_transport_creation_and_muon_parity(self):
        """Missing PP buckets do not desynchronize WORLD group creation."""
        world_mesh = init_device_mesh(
            "cpu", (self.world_size,), mesh_dim_names=("world",)
        )
        mesh = world_mesh._unflatten(
            0, (2, 2, 2), mesh_dim_names=("pp", "dp_shard", "tp")
        )["dp_shard", "tp"]
        name = "dense" if self.rank < 4 else "experts"
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
                    {"dp_shard": Shard(0), "tp": Shard(0)}, compute_dtype=torch.bfloat16
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
