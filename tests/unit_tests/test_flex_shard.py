# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import os
import tempfile
import unittest
from collections.abc import Mapping, MutableMapping
from typing import Any
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from torch.optim import Optimizer
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.components.flex_shard import (
    BucketSpec,
    flex_shard,
    Owned,
)


class _OwnedTestOptimizer(Optimizer):
    """Minimal complete-matrix optimizer used to test FlexShard itself."""

    def __init__(self, params, lr: float = 0.1) -> None:
        super().__init__(params, {"lr": lr})

    def flex_shard_compute_requirement(self, param, group):
        return Owned(trailing_dims=2)

    def flex_shard_init_state(self, param, grad, group):
        state = self.state[param]
        if not state:
            state["accumulator"] = torch.zeros_like(param)
        return state

    def flex_shard_prepare(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        state: Mapping[str, torch.Tensor],
        group: MutableMapping[str, Any],
        *,
        out: torch.Tensor,
    ) -> None:
        state["accumulator"].add_(grad)
        out.copy_(state["accumulator"])

    def flex_shard_compute(self, compute_input, group):
        # The global mean makes the result depend on correct full-matrix assembly.
        return compute_input + compute_input.mean()

    def flex_shard_finalize(self, param, update, group, *, out) -> None:
        if out is not param:
            out.copy_(param)
        out.add_(update, alpha=-group["lr"])


def _run_cuda_pipeline_lifetime(
    rank: int,
    world_size: int,
    store_path: str,
) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        device = torch.device("cuda", rank)
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
        full_values = [
            torch.arange(1, 2049, device=device, dtype=torch.float32).reshape(64, 32)
            / 2049,
            torch.arange(1, 1537, device=device, dtype=torch.float32).reshape(48, 32)
            / 1537,
            torch.arange(1, 1025, device=device, dtype=torch.float32).reshape(32, 32)
            / 1025,
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

        reference_values = [value.clone() for value in full_values]
        reference_accumulators = [torch.zeros_like(value) for value in full_values]
        pipeline_params = [
            to_dtensor(value.clone()).requires_grad_() for value in full_values
        ]
        bucket_specs = [
            BucketSpec(
                name=f"layer-{index}",
                patterns=(f"layers.{index}.weight",),
                mesh=mesh,
            )
            for index in range(len(full_values))
        ]

        pipeline_optimizer = flex_shard(
            _OwnedTestOptimizer(
                [
                    {
                        "params": [param],
                        "param_names": [f"layers.{index}.weight"],
                    }
                    for index, param in enumerate(pipeline_params)
                ],
                lr=0.03,
            ),
            bucket_spec=bucket_specs,
        )

        runtime = pipeline_optimizer.__dict__["_flex_shard_runtime"]
        original_reverse_bucket = runtime._reverse_bucket

        def delayed_reverse_bucket(work):
            torch.cuda._sleep(500_000)
            return original_reverse_bucket(work)

        runtime._reverse_bucket = delayed_reverse_bucket
        caller_stream = torch.cuda.Stream()
        active_bytes = []

        def forbid_record_stream(*args, **kwargs):
            raise AssertionError("optimizer FlexShard must not call record_stream")

        for step in range(3):
            torch.manual_seed(1000 + step)
            full_grads = [torch.randn_like(value) for value in full_values]
            for reference, accumulator, pipeline_param, grad in zip(
                reference_values,
                reference_accumulators,
                pipeline_params,
                full_grads,
                strict=True,
            ):
                accumulator.add_(grad)
                reference.add_(accumulator + accumulator.mean(), alpha=-0.03)
                pipeline_param.grad = to_dtensor(grad.clone())

            caller_stream.wait_stream(torch.cuda.current_stream(device))
            with (
                torch.cuda.stream(caller_stream),
                mock.patch.object(
                    torch.Tensor,
                    "record_stream",
                    new=forbid_record_stream,
                ),
            ):
                pipeline_optimizer.step()
                snapshots = [param._local_tensor.clone() for param in pipeline_params]
                poison = [
                    torch.empty_like(param._local_tensor).fill_(step + rank + 1)
                    for param in pipeline_params
                    for _ in range(8)
                ]
            caller_stream.synchronize()

            for snapshot, pipeline_param, reference, accumulator in zip(
                snapshots,
                pipeline_params,
                reference_values,
                reference_accumulators,
                strict=True,
            ):
                rows, row_offset = Shard.local_shard_size_and_offset(
                    reference.shape[0], world_size, rank
                )
                expected_param = reference.narrow(0, int(row_offset), int(rows))
                expected_state = accumulator.narrow(0, int(row_offset), int(rows))
                torch.testing.assert_close(snapshot, expected_param)
                torch.testing.assert_close(
                    pipeline_optimizer.state[pipeline_param][
                        "accumulator"
                    ]._local_tensor,
                    expected_state,
                )

            del full_grads, poison, snapshots
            torch.cuda.synchronize(device)
            active_bytes.append(torch.cuda.memory_allocated(device))

        if max(active_bytes[1:]) - min(active_bytes[1:]) > 1024 * 1024:
            raise AssertionError(
                f"pipelined FlexShard retained CUDA memory: {active_bytes}"
            )
    finally:
        dist.destroy_process_group()


class TestFlexShard(DTensorTestBase):
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

    def _sharded_dtensor(self, full: torch.Tensor) -> DTensor:
        rows, offset = Shard.local_shard_size_and_offset(
            full.shape[0], self.world_size, dist.get_rank()
        )
        local = full.narrow(0, int(offset), int(rows)).contiguous()
        return DTensor.from_local(
            local,
            self.mesh,
            (Shard(0),),
            run_check=False,
            shape=full.shape,
            stride=full.stride(),
        )

    def _parameter(self, full: torch.Tensor, grad: torch.Tensor) -> DTensor:
        param = self._sharded_dtensor(full.clone()).requires_grad_()
        param.grad = self._sharded_dtensor(grad.clone())
        return param

    def _owned_optimizer(self):
        full_values = (
            torch.arange(1, 25, dtype=torch.float32).reshape(6, 4) / 29,
            torch.arange(1, 16, dtype=torch.float32).reshape(5, 3) / 17,
            torch.arange(1, 9, dtype=torch.float32).reshape(4, 2) / 11,
        )
        full_grads = tuple(value.flip(0).div(7) for value in full_values)
        params = [
            self._parameter(value, grad)
            for value, grad in zip(full_values, full_grads, strict=True)
        ]
        optimizer = _OwnedTestOptimizer(
            [
                {
                    "params": params,
                    "param_names": [
                        "layers.0.first.weight",
                        "layers.0.second.weight",
                        "layers.0.third.weight",
                    ],
                }
            ],
            lr=0.2,
        )
        return optimizer, params, full_values, full_grads

    @with_comms
    def test_owned_runtime_uses_complete_matrices_and_preserves_storage(self):
        optimizer, params, full_values, full_grads = self._owned_optimizer()
        local_ptrs = [param.to_local().data_ptr() for param in params]
        placements = [param.placements for param in params]
        versions = [param._version for param in params]
        flex_shard(
            optimizer,
            bucket_spec=[
                BucketSpec(
                    name="layers.0",
                    patterns=("layers.0.*",),
                    mesh=self.mesh,
                )
            ],
        )
        with self.assertRaisesRegex(RuntimeError, "after flex_shard plan"):
            optimizer.add_param_group(
                {
                    "params": [torch.nn.Parameter(torch.ones(2, 2))],
                    "param_names": ["late.weight"],
                }
            )

        all_to_all_calls = 0
        status_collective_calls = 0
        original_all_to_all = dist.all_to_all_single
        original_all_gather = dist.all_gather
        original_all_reduce = dist.all_reduce

        def counted_all_to_all(*args, **kwargs):
            nonlocal all_to_all_calls
            all_to_all_calls += 1
            return original_all_to_all(*args, **kwargs)

        def counted_all_gather(*args, **kwargs):
            nonlocal status_collective_calls
            status_collective_calls += 1
            return original_all_gather(*args, **kwargs)

        def counted_all_reduce(*args, **kwargs):
            nonlocal status_collective_calls
            status_collective_calls += 1
            return original_all_reduce(*args, **kwargs)

        with (
            mock.patch.object(
                dist, "all_to_all_single", side_effect=counted_all_to_all
            ),
            mock.patch.object(
                dist, "all_gather", side_effect=counted_all_gather
            ),
            mock.patch.object(
                dist, "all_reduce", side_effect=counted_all_reduce
            ),
        ):
            optimizer.step()

        self.assertEqual(all_to_all_calls, 2)
        self.assertEqual(status_collective_calls, 0)
        rank = dist.get_rank()
        for index, (param, full_value, full_grad) in enumerate(
            zip(params, full_values, full_grads, strict=True)
        ):
            expected = full_value - 0.2 * (full_grad + full_grad.mean())
            rows, offset = Shard.local_shard_size_and_offset(
                full_value.shape[0], self.world_size, rank
            )
            torch.testing.assert_close(
                param.to_local(),
                expected.narrow(0, int(offset), int(rows)),
            )
            self.assertEqual(param.placements, placements[index])
            self.assertEqual(param.to_local().data_ptr(), local_ptrs[index])
            self.assertEqual(param._version, versions[index] + 1)
            accumulator = optimizer.state[param]["accumulator"]
            self.assertIsInstance(accumulator, DTensor)
            self.assertEqual(accumulator.placements, placements[index])
            torch.testing.assert_close(accumulator.to_local(), param.grad.to_local())

    @with_comms
    def test_pipeline_prefetches_one_bucket(self):
        optimizer, params, _full_values, _full_grads = self._owned_optimizer()
        optimizer.param_groups[0]["param_names"] = [
            "layers.0.first.weight",
            "layers.1.second.weight",
            "layers.2.third.weight",
        ]
        flex_shard(
            optimizer,
            bucket_spec=[
                BucketSpec(
                    name=f"layers.{index}",
                    patterns=(f"layers.{index}.*",),
                    mesh=self.mesh,
                )
                for index in range(3)
            ],
        )
        runtime = optimizer.__dict__["_flex_shard_runtime"]

        stages = []
        original_forward_bucket = runtime._forward_bucket
        original_reverse_bucket = runtime._reverse_bucket

        def record_forward_bucket(work):
            stages.append(f"F:{work.plan.spec.name}")
            return original_forward_bucket(work)

        def record_reverse_bucket(work):
            stages.append(f"R:{work.plan.spec.name}")
            return original_reverse_bucket(work)

        runtime._forward_bucket = record_forward_bucket
        runtime._reverse_bucket = record_reverse_bucket
        optimizer.step()
        self.assertEqual(
            stages,
            [
                "F:layers.0",
                "F:layers.1",
                "R:layers.0",
                "F:layers.2",
                "R:layers.1",
                "R:layers.2",
            ],
        )

    @with_comms
    def test_bucket_coverage_and_storage_layout_validation(self):
        optimizer, _params, _values, _grads = self._owned_optimizer()
        with self.assertRaisesRegex(ValueError, "not covered"):
            flex_shard(
                optimizer,
                bucket_spec=[
                    BucketSpec(patterns=("layers.1.*",), mesh=self.mesh)
                ],
            )

        optimizer, _params, _values, _grads = self._owned_optimizer()
        with self.assertRaisesRegex(ValueError, "matched multiple"):
            flex_shard(
                optimizer,
                bucket_spec=[
                    BucketSpec(patterns=("layers.*",), mesh=self.mesh),
                    BucketSpec(patterns=("layers.0.*",), mesh=self.mesh),
                ],
            )

        local = torch.ones(4, 2)
        param = DTensor.from_local(
            local,
            self.mesh,
            (Shard(1),),
            run_check=False,
            shape=(4, 4),
            stride=(4, 1),
        ).requires_grad_()
        optimizer = _OwnedTestOptimizer(
            [{"params": [param], "param_names": ["layers.0.weight"]}]
        )
        with self.assertRaisesRegex(ValueError, r"Shard\(0\)"):
            flex_shard(
                optimizer,
                bucket_spec=[
                    BucketSpec(patterns=("layers.0.*",), mesh=self.mesh)
                ],
            )

    @with_comms
    def test_requires_compute_requirement_provider(self):
        param = self._parameter(torch.ones(4, 2), torch.ones(4, 2))
        with self.assertRaisesRegex(TypeError, "must implement"):
            flex_shard(
                torch.optim.SGD(
                    [
                        {
                            "params": [param],
                            "param_names": ["layers.0.weight"],
                        }
                    ],
                    lr=0.1,
                ),
                bucket_spec=[
                    BucketSpec(
                        patterns=("layers.0.*",),
                        mesh=self.mesh,
                    )
                ],
            )


class TestDistributedFlexShard(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
    def test_pipeline_cuda_allocator_lifetime(self):
        with tempfile.TemporaryDirectory() as store_dir:
            mp.spawn(
                _run_cuda_pipeline_lifetime,
                args=(2, os.path.join(store_dir, "store")),
                nprocs=2,
                join=True,
            )
