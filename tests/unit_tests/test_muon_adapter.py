# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import datetime
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.distributed.tensor.placement_types as placement_types
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.components.flex_shard import (
    BucketSpec,
    build_layer_bucket_specs,
    flex_shard,
    get_flex_shard_assignments,
    Owned,
)
from torchtitan.components.muon_adapter import MuonAdapter


_StridedShard = getattr(placement_types, "_StridedShard", None)


def _run_bucketed_muon_parity(
    rank: int,
    world_size: int,
    store_path: str,
) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("fsdp",))
        full_values = [
            torch.arange(1, 25, dtype=torch.float32).reshape(6, 4) / 29,
            torch.arange(1, 16, dtype=torch.float32).reshape(5, 3) / 17,
            torch.arange(1, 9, dtype=torch.float32).reshape(4, 2) / 11,
            torch.arange(1, 16, dtype=torch.float32).reshape(3, 5) / 23,
            torch.arange(1, 7, dtype=torch.float32).reshape(3, 2) / 13,
        ]

        def to_dtensor(tensor: torch.Tensor) -> DTensor:
            rows, row_offset = Shard.local_shard_size_and_offset(
                tensor.shape[0], world_size, rank
            )
            local = tensor.narrow(0, int(row_offset), int(rows)).contiguous()
            return DTensor.from_local(
                local,
                device_mesh=mesh,
                placements=(Shard(0),),
                run_check=False,
                shape=tensor.shape,
                stride=tensor.stride(),
            )

        params = [to_dtensor(value.clone()).requires_grad_() for value in full_values]
        references = [
            torch.nn.Parameter(full_values[0].clone().view(2, 3, 4)),
            torch.nn.Parameter(full_values[1].clone()),
            torch.nn.Parameter(full_values[2].clone()),
            torch.nn.Parameter(full_values[3].clone()),
            torch.nn.Parameter(full_values[4].clone()),
        ]
        kwargs = {
            "lr": 0.03,
            "weight_decay": 0.2,
            "momentum": 0.8,
            "nesterov": True,
            "ns_steps": 2,
            "adjust_lr_fn": "match_rms_adamw",
        }
        bucket_specs = [
            BucketSpec(
                name="layer-0",
                patterns=("layers.0.*",),
                mesh=mesh,
            ),
            BucketSpec(
                name="layer-1",
                patterns=("layers.1.*",),
                mesh=mesh,
            ),
        ]

        def make_optimizer(optimizer_params, **overrides):
            optimizer = MuonAdapter(
                [
                    {
                        "params": [optimizer_params[0]],
                        "param_names": ["layers.0.attention.wq.weight"],
                        "matrix_shape": (3, 4),
                    },
                    {
                        "params": [optimizer_params[1]],
                        "param_names": ["layers.1.feed_forward.w1.weight"],
                    },
                    {
                        "params": [optimizer_params[2]],
                        "param_names": ["layers.0.feed_forward.w1.weight"],
                    },
                    {
                        "params": [optimizer_params[3]],
                        "param_names": ["layers.1.feed_forward.w2.weight"],
                    },
                    {
                        "params": [optimizer_params[4]],
                        "param_names": ["layers.1.feed_forward.w3.weight"],
                    },
                ],
                **{**kwargs, **overrides},
            )
            return flex_shard(optimizer, bucket_spec=bucket_specs)

        optimizer = make_optimizer(params)
        reference_optimizer = torch.optim.Muon(
            [
                {"params": [references[0]]},
                {"params": [references[1]]},
                {"params": [references[2]]},
                {"params": [references[3]]},
                {"params": [references[4]]},
            ],
            **kwargs,
        )

        all_to_all_calls = 0
        status_collective_calls = 0
        original_all_to_all = dist.all_to_all_single
        original_all_gather = dist.all_gather
        original_all_reduce = dist.all_reduce

        def counted_all_to_all(*args, **call_kwargs):
            nonlocal all_to_all_calls
            all_to_all_calls += 1
            return original_all_to_all(*args, **call_kwargs)

        def counted_all_gather(*args, **call_kwargs):
            nonlocal status_collective_calls
            status_collective_calls += 1
            return original_all_gather(*args, **call_kwargs)

        def counted_all_reduce(*args, **call_kwargs):
            nonlocal status_collective_calls
            status_collective_calls += 1
            return original_all_reduce(*args, **call_kwargs)

        def optimizer_step_without_status_collectives(step_optimizer):
            calls_before_step = status_collective_calls
            step_optimizer.step()
            assert status_collective_calls == calls_before_step

        persistent_params = list(params)
        param_ptrs = [param.to_local().data_ptr() for param in params]
        storage_placements = [param.placements for param in params]
        persistent_momentum = [None] * len(params)
        momentum_ptrs = [None] * len(params)
        momentum_versions = [None] * len(params)

        def assert_state_matches_reference():
            for index, (param, reference) in enumerate(
                zip(params, references, strict=True)
            ):
                rows, row_offset = Shard.local_shard_size_and_offset(
                    full_values[index].shape[0], world_size, rank
                )
                expected_param = reference.view(full_values[index].shape).narrow(
                    0, int(row_offset), int(rows)
                )
                torch.testing.assert_close(
                    param.to_local(), expected_param, rtol=1e-6, atol=1e-7
                )
                assert param is persistent_params[index]
                assert param.to_local().data_ptr() == param_ptrs[index]
                assert param.placements == storage_placements[index]

                grad = param.grad
                reference_grad = reference.grad
                if reference_grad is None:
                    assert grad is None
                else:
                    assert isinstance(grad, DTensor)
                    assert grad.placements == storage_placements[index]
                    expected_grad = reference_grad.view(
                        full_values[index].shape
                    ).narrow(0, int(row_offset), int(rows))
                    torch.testing.assert_close(
                        grad.to_local(), expected_grad, rtol=0, atol=0
                    )

                momentum = optimizer.state[param]["momentum_buffer"]
                reference_momentum = reference_optimizer.state[reference][
                    "momentum_buffer"
                ].view(full_values[index].shape)
                expected_momentum = reference_momentum.narrow(
                    0, int(row_offset), int(rows)
                )
                torch.testing.assert_close(
                    momentum.to_local(), expected_momentum, rtol=0, atol=0
                )
                if persistent_momentum[index] is None:
                    persistent_momentum[index] = momentum
                    momentum_ptrs[index] = momentum.to_local().data_ptr()
                else:
                    assert momentum is persistent_momentum[index]
                    assert momentum.to_local().data_ptr() == momentum_ptrs[index]
                    previous_version = momentum_versions[index]
                    assert previous_version is not None
                    assert momentum._version == previous_version + int(
                        param.grad is not None
                    )
                momentum_versions[index] = momentum._version

        with (
            patch.object(dist, "all_to_all_single", side_effect=counted_all_to_all),
            patch.object(dist, "all_gather", side_effect=counted_all_gather),
            patch.object(dist, "all_reduce", side_effect=counted_all_reduce),
        ):
            for step in range(3):
                torch.manual_seed(100 + step)
                full_grads = [torch.randn_like(value) for value in full_values]
                params[0].grad = to_dtensor(full_grads[0].clone())
                for index in range(1, len(params)):
                    params[index].grad = to_dtensor(full_grads[index].clone())
                references[0].grad = full_grads[0].view(2, 3, 4).clone()
                for index in range(1, len(references)):
                    references[index].grad = full_grads[index].clone()

                versions = [param._version for param in params]
                optimizer_step_without_status_collectives(optimizer)
                reference_optimizer.step()
                assert all(
                    param._version == version + 1
                    for param, version in zip(params, versions, strict=True)
                )
                assert_state_matches_reference()

            torch.manual_seed(200)
            active_grad = torch.randn_like(full_values[0])
            params[0].grad = to_dtensor(active_grad.clone())
            for param in params[1:]:
                param.grad = None
            references[0].grad = active_grad.view(2, 3, 4).clone()
            for reference in references[1:]:
                reference.grad = None
            versions = [param._version for param in params]
            optimizer_step_without_status_collectives(optimizer)
            reference_optimizer.step()
            assert params[0]._version == versions[0] + 1
            assert all(
                param._version == version
                for param, version in zip(params[1:], versions[1:], strict=True)
            )
            assert_state_matches_reference()

            saved_state = copy.deepcopy(optimizer.state_dict())
            assert all(
                set(param_state) == {"momentum_buffer"}
                for param_state in saved_state["state"].values()
            )
            restored_params = [
                DTensor.from_local(
                    param.to_local().detach().clone(),
                    device_mesh=mesh,
                    placements=(Shard(0),),
                    run_check=False,
                    shape=param.shape,
                    stride=param.stride(),
                ).requires_grad_()
                for param in params
            ]
            restored_optimizer = make_optimizer(restored_params)
            restored_optimizer.load_state_dict(saved_state)

            torch.manual_seed(300)
            continuation_grads = [torch.randn_like(value) for value in full_values]
            for index in range(len(params)):
                params[index].grad = to_dtensor(continuation_grads[index].clone())
                restored_params[index].grad = to_dtensor(
                    continuation_grads[index].clone()
                )
            references[0].grad = continuation_grads[0].view(2, 3, 4).clone()
            for index in range(1, len(references)):
                references[index].grad = continuation_grads[index].clone()
            optimizer_step_without_status_collectives(optimizer)
            optimizer_step_without_status_collectives(restored_optimizer)
            reference_optimizer.step()
            assert_state_matches_reference()
            for param, restored_param in zip(params, restored_params, strict=True):
                torch.testing.assert_close(
                    restored_param.to_local(), param.to_local(), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    restored_optimizer.state[restored_param][
                        "momentum_buffer"
                    ].to_local(),
                    optimizer.state[param]["momentum_buffer"].to_local(),
                    rtol=0,
                    atol=0,
                )

            mismatch_params = [
                to_dtensor(value.clone()).requires_grad_() for value in full_values
            ]
            with unittest.TestCase().assertRaisesRegex(
                RuntimeError, "FlexShard plans differ across ranks"
            ):
                make_optimizer(
                    mismatch_params,
                    lr=0.04 if rank == 0 else 0.03,
                )

            bad_local_rows = 2 if rank == 0 else 3
            bad_param = DTensor.from_local(
                torch.ones(bad_local_rows, 4),
                device_mesh=mesh,
                placements=(Shard(0),),
                run_check=False,
                shape=(6, 4),
                stride=(4, 1),
            ).requires_grad_()
            bad_optimizer = MuonAdapter(
                [
                    {
                        "params": [bad_param],
                        "param_names": ["layers.0.bad.weight"],
                    }
                ]
            )
            with unittest.TestCase().assertRaises((ValueError, RuntimeError)):
                flex_shard(
                    bad_optimizer,
                    bucket_spec=[
                        BucketSpec(
                            patterns=("layers.0.*",),
                            mesh=mesh,
                        )
                    ],
                )

        assert all_to_all_calls == 3 * 2 * 2 + 2 + 2 * 2 * 2
    finally:
        dist.destroy_process_group()


def _has_batched_muon() -> bool:
    try:
        torch.optim.Muon([torch.nn.Parameter(torch.randn(1, 2, 3))])
    except ValueError:
        return False
    return True


class TestMuonAdapter(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @property
    def device_type(self):
        return "cpu"

    @property
    def mesh(self):
        if not hasattr(self, "_mesh"):
            self._mesh = init_device_mesh(
                self.device_type,
                (self.world_size,),
                mesh_dim_names=("fsdp",),
            )
        return self._mesh

    def test_rejects_unsupported_implementation(self):
        param = torch.nn.Parameter(torch.randn(3, 4))
        with self.assertRaisesRegex(NotImplementedError, "fused or foreach"):
            MuonAdapter([{"params": [param], "fused": True}])

    def _sharded_dtensor(self, local, global_shape):
        stride = torch.empty(global_shape).stride()
        return DTensor.from_local(
            local,
            self.mesh,
            [Shard(0)],
            run_check=False,
            shape=global_shape,
            stride=stride,
        )

    def _dtensor_parameter(self, local, grad, global_shape):
        param = self._sharded_dtensor(local.clone(), global_shape)
        param.requires_grad_()
        param.grad = self._sharded_dtensor(grad, global_shape)
        return param

    def _bucketed_optimizer(self):
        global_shape = (4, 3)
        first = self._dtensor_parameter(
            torch.ones(2, 3), torch.ones(2, 3), global_shape
        )
        second = self._dtensor_parameter(
            torch.full((2, 3), 2.0), torch.ones(2, 3), global_shape
        )
        optimizer = MuonAdapter(
            [
                {
                    "params": [first, second],
                    "param_names": [
                        "layers.0.attention.wq.weight",
                        "layers.1.attention.wq.weight",
                    ],
                }
            ]
        )
        return optimizer, first, second

    @with_comms
    def test_compute_buckets_resolve_fqns_and_balance_layers(self):
        optimizer, _first, _second = self._bucketed_optimizer()

        bucket_spec = build_layer_bucket_specs(optimizer)
        self.assertEqual(
            [(spec.name, spec.patterns) for spec in bucket_spec],
            [
                ("layers.0", ("layers.0.attention.wq.weight",)),
                ("layers.1", ("layers.1.attention.wq.weight",)),
            ],
        )
        returned = flex_shard(optimizer, bucket_spec=bucket_spec)

        self.assertIs(returned, optimizer)
        self.assertEqual(
            [
                (assignment.bucket_name, assignment.fqn, assignment.owner_rank)
                for assignment in get_flex_shard_assignments(optimizer)
            ],
            [
                ("layers.0", "layers.0.attention.wq.weight", 0),
                ("layers.1", "layers.1.attention.wq.weight", 1),
            ],
        )
        with self.assertRaisesRegex(RuntimeError, "after flex_shard plan"):
            optimizer.add_param_group(
                {
                    "params": [torch.nn.Parameter(torch.ones(2, 2))],
                    "param_names": ["late.weight"],
                }
            )

    @with_comms
    def test_adamw_same_as_storage_preserves_storage_sharding(self):
        local_param = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)
        local_grad = torch.arange(6, 0, -1, dtype=torch.float32).reshape(2, 3)
        param = self._dtensor_parameter(
            local_param,
            local_grad.clone(),
            global_shape=(4, 3),
        )
        reference = torch.nn.Parameter(local_param.clone())
        reference.grad = local_grad.clone()
        kwargs = {
            "lr": 0.03,
            "weight_decay": 0.2,
            "foreach": False,
            "fused": False,
        }
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [param],
                    "param_names": ["layers.0.weight"],
                }
            ],
            **kwargs,
        )
        reference_optimizer = torch.optim.AdamW([reference], **kwargs)
        storage_placements = param.placements
        local_data_ptr = param.to_local().data_ptr()

        returned = flex_shard(
            optimizer,
            bucket_spec=[
                BucketSpec(
                    name="layers.0",
                    patterns=("layers.0.*",),
                    mesh=self.mesh,
                )
            ],
        )
        self.assertIs(returned, optimizer)
        self.assertEqual(
            [
                (assignment.bucket_name, assignment.fqn, assignment.owner_rank)
                for assignment in get_flex_shard_assignments(optimizer)
            ],
            [("layers.0", "layers.0.weight", None)],
        )

        optimizer.step()
        reference_optimizer.step()

        torch.testing.assert_close(param.to_local(), reference)
        self.assertEqual(param.placements, storage_placements)
        self.assertEqual(param.to_local().data_ptr(), local_data_ptr)
        self.assertIsInstance(optimizer.state[param]["exp_avg"], DTensor)
        self.assertEqual(
            optimizer.state[param]["exp_avg"].placements,
            storage_placements,
        )
        with self.assertRaisesRegex(RuntimeError, "after flex_shard plan"):
            optimizer.add_param_group(
                {
                    "params": [torch.nn.Parameter(torch.ones(2, 2))],
                    "param_names": ["late.weight"],
                }
            )

    @with_comms
    def test_compute_requirements_are_declarative(self):
        optimizer, first, _second = self._bucketed_optimizer()
        self.assertEqual(
            optimizer.flex_shard_compute_requirement(first, optimizer.param_groups[0]),
            Owned(trailing_dims=2),
        )

        with self.assertRaisesRegex(TypeError, "must implement"):
            flex_shard(
                torch.optim.SGD([torch.nn.Parameter(torch.ones(2, 2))]),
                bucket_spec=[],
            )

    @with_comms
    def test_compute_buckets_reject_orphan_and_overlap(self):
        optimizer, _first, _second = self._bucketed_optimizer()
        with self.assertRaisesRegex(ValueError, "not covered"):
            flex_shard(
                optimizer,
                bucket_spec=[
                    BucketSpec(
                        patterns=("layers.0.*",),
                        mesh=self.mesh,
                    )
                ],
            )

        optimizer, _first, _second = self._bucketed_optimizer()
        with self.assertRaisesRegex(ValueError, "matched multiple"):
            flex_shard(
                optimizer,
                bucket_spec=[
                    BucketSpec(
                        patterns=("layers.*",),
                        mesh=self.mesh,
                    ),
                    BucketSpec(
                        patterns=("layers.0.*", "layers.1.*"),
                        mesh=self.mesh,
                    ),
                ],
            )

    @with_comms
    def test_compute_buckets_reject_non_row_shard_and_invalid_ns(self):
        local = torch.ones(4, 2)
        param = DTensor.from_local(
            local,
            device_mesh=self.mesh,
            placements=(Shard(1),),
            run_check=False,
            shape=(4, 4),
            stride=(4, 1),
        ).requires_grad_()
        optimizer = MuonAdapter(
            [{"params": [param], "param_names": ["layers.0.weight"]}]
        )
        with self.assertRaisesRegex(ValueError, r"Shard\(0\)"):
            flex_shard(
                optimizer,
                bucket_spec=[
                    BucketSpec(
                        patterns=("layers.0.*",),
                        mesh=self.mesh,
                    )
                ],
            )

        optimizer, _first, _second = self._bucketed_optimizer()
        optimizer.param_groups[0]["ns_steps"] = 100
        with self.assertRaisesRegex(ValueError, "ns_steps"):
            flex_shard(
                optimizer,
                bucket_spec=[
                    BucketSpec(
                        patterns=("layers.*",),
                        mesh=self.mesh,
                    )
                ],
            )

    @unittest.skipUnless(_has_batched_muon(), "requires PyTorch PR #190597")
    def test_ordinary_matrix_matches_muon_for_two_steps(self):
        initial = torch.arange(1, 13, dtype=torch.bfloat16).reshape(3, 4) / 13
        grads = (
            torch.arange(1, 13, dtype=torch.bfloat16).reshape(3, 4) / 17,
            torch.arange(12, 0, -1, dtype=torch.bfloat16).reshape(3, 4) / 19,
        )
        param = torch.nn.Parameter(initial.clone())
        reference = torch.nn.Parameter(initial.clone())
        kwargs = {
            "lr": 0.03,
            "weight_decay": 0.2,
            "momentum": 0.8,
            "nesterov": False,
            "ns_steps": 2,
        }
        optimizer = MuonAdapter([param], **kwargs)
        reference_optimizer = torch.optim.Muon([reference], **kwargs)
        expected_momentum = torch.zeros_like(param)

        for grad in grads:
            param.grad = grad.clone()
            reference.grad = grad.clone()
            expected_momentum.lerp_(grad, 1 - kwargs["momentum"])

            optimizer.step()
            reference_optimizer.step()

            torch.testing.assert_close(param, reference)
            torch.testing.assert_close(
                optimizer.state[param]["momentum_buffer"], expected_momentum
            )

    @unittest.skipUnless(_has_batched_muon(), "requires PyTorch PR #190597")
    @with_comms
    def test_leading_expert_shard_matches_independent_muon_for_two_steps(self):
        local_experts, matrix_rows, matrix_cols = 2, 3, 4
        local_shape = (local_experts, matrix_rows, matrix_cols)
        global_shape = (self.world_size * local_experts, matrix_rows, matrix_cols)
        local_param = (
            torch.arange(1, 25, dtype=torch.bfloat16).reshape(local_shape) / 29
        )
        local_grads = (
            torch.arange(1, 25, dtype=torch.bfloat16).reshape(local_shape) / 31,
            torch.arange(24, 0, -1, dtype=torch.bfloat16).reshape(local_shape) / 37,
        )
        param = self._dtensor_parameter(
            local_param, local_grads[0].clone(), global_shape
        )
        reference_params = [
            torch.nn.Parameter(matrix.clone()) for matrix in local_param
        ]
        kwargs = {
            "lr": 0.03,
            "weight_decay": 0.2,
            "momentum": 0.8,
            "nesterov": False,
            "ns_steps": 2,
        }
        optimizer = MuonAdapter([param], **kwargs)
        reference_optimizer = torch.optim.Muon(reference_params, **kwargs)
        expected_momentum = torch.zeros_like(local_param)
        storage_placements = param.placements
        local_storage_ptr = param.to_local().data_ptr()
        persistent_momentum = None
        momentum_storage_ptr = None

        self.assertEqual(
            MuonAdapter._compute_placements(param, matrix_shape=None),
            storage_placements,
        )

        for local_grad in local_grads:
            param.grad = self._sharded_dtensor(local_grad.clone(), global_shape)
            for reference_param, reference_grad in zip(
                reference_params, local_grad, strict=True
            ):
                reference_param.grad = reference_grad.clone()
            expected_momentum.lerp_(local_grad, 1 - kwargs["momentum"])

            optimizer.step()
            reference_optimizer.step()

            self.assertIs(optimizer.param_groups[0]["params"][0], param)
            self.assertEqual(param.placements, storage_placements)
            self.assertEqual(param.to_local().data_ptr(), local_storage_ptr)
            torch.testing.assert_close(param.to_local(), torch.stack(reference_params))

            momentum = optimizer.state[param]["momentum_buffer"]
            reference_momentum = torch.stack(
                [
                    reference_optimizer.state[reference_param]["momentum_buffer"]
                    for reference_param in reference_params
                ]
            )
            self.assertIsInstance(momentum, DTensor)
            self.assertEqual(momentum.shape, param.shape)
            self.assertEqual(momentum.placements, storage_placements)
            if persistent_momentum is None:
                persistent_momentum = momentum
                momentum_storage_ptr = momentum.to_local().data_ptr()
            else:
                self.assertIs(momentum, persistent_momentum)
                self.assertEqual(momentum.to_local().data_ptr(), momentum_storage_ptr)
            torch.testing.assert_close(momentum.to_local(), expected_momentum)
            torch.testing.assert_close(momentum.to_local(), reference_momentum)

    @unittest.skipUnless(_has_batched_muon(), "requires PyTorch PR #190597")
    @with_comms
    def test_matrix_shape_matches_batched_muon_for_two_steps(self):
        head_dim, model_dim = 3, 4
        global_shape = (2 * head_dim, model_dim)
        local_param = (
            torch.arange(1, 13, dtype=torch.bfloat16).reshape(head_dim, model_dim) / 13
        )
        local_grads = (
            torch.arange(1, 13, dtype=torch.bfloat16).reshape(head_dim, model_dim) / 17,
            torch.arange(12, 0, -1, dtype=torch.bfloat16).reshape(head_dim, model_dim)
            / 19,
        )
        param = self._dtensor_parameter(
            local_param, local_grads[0].clone(), global_shape
        )
        kwargs = {
            "lr": 0.03,
            "weight_decay": 0.2,
            "momentum": 0.8,
            "nesterov": False,
            "ns_steps": 2,
        }
        optimizer = MuonAdapter(
            [{"params": [param], "matrix_shape": (head_dim, model_dim)}],
            **kwargs,
        )

        reference = torch.nn.Parameter(torch.stack([local_param, local_param]))
        reference_optimizer = torch.optim.Muon([reference], **kwargs)
        expected_momentum = torch.zeros_like(local_param)
        storage_placements = param.placements
        persistent_momentum = None

        for local_grad in local_grads:
            param.grad = self._sharded_dtensor(local_grad.clone(), global_shape)
            reference.grad = torch.stack([local_grad, local_grad])
            expected_momentum.lerp_(local_grad, 1 - kwargs["momentum"])

            optimizer.step()
            reference_optimizer.step()

            self.assertIs(optimizer.param_groups[0]["params"][0], param)
            self.assertEqual(param.placements, storage_placements)
            torch.testing.assert_close(param.to_local(), reference[0])

            momentum = optimizer.state[param]["momentum_buffer"]
            self.assertIsInstance(momentum, DTensor)
            self.assertEqual(momentum.shape, param.shape)
            self.assertEqual(momentum.placements, storage_placements)
            if persistent_momentum is None:
                persistent_momentum = momentum
            else:
                self.assertIs(momentum, persistent_momentum)
            torch.testing.assert_close(momentum.to_local(), expected_momentum)
            torch.testing.assert_close(
                momentum.to_local(),
                reference_optimizer.state[reference]["momentum_buffer"][0],
            )


class TestMuonAdapterStridedPolicy(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @property
    def device_type(self):
        return "cpu"

    @property
    def mesh(self):
        if not hasattr(self, "_mesh"):
            self._mesh = init_device_mesh(
                self.device_type,
                (2, 2),
                mesh_dim_names=("dp", "tp"),
            )
        return self._mesh

    @unittest.skipIf(_StridedShard is None, "PyTorch has no private strided shard")
    @with_comms
    def test_composed_strided_storage_uses_conservative_compute_layout(self):
        assert _StridedShard is not None
        storage_placements = (
            _StridedShard(0, split_factor=self.mesh["tp"].size()),
            Shard(0),
        )
        layout = DTensor.from_local(
            torch.zeros(2, 3, 4),
            self.mesh,
            storage_placements,
            run_check=False,
            shape=(8, 3, 4),
            stride=(12, 4, 1),
        )

        self.assertIs(type(storage_placements[0]), _StridedShard)
        self.assertEqual(
            MuonAdapter._compute_placements(layout, matrix_shape=None),
            (Replicate(), Shard(0)),
        )
        self.assertEqual(
            MuonAdapter._compute_placements(layout, matrix_shape=(3, 4)),
            (Replicate(), Replicate()),
        )


class TestDistributedBucketedMuonAdapter(unittest.TestCase):
    @unittest.skipUnless(_has_batched_muon(), "requires PyTorch PR #190597")
    def test_layer_buckets_match_full_muon(self):
        with tempfile.TemporaryDirectory() as store_dir:
            mp.spawn(
                _run_bucketed_muon_parity,
                args=(2, os.path.join(store_dir, "store")),
                nprocs=2,
                join=True,
            )
