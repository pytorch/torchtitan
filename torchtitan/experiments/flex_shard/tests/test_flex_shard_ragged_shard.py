#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    FSDPTest,
    FSDPTestMultiThread,
    get_devtype,
)
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.experiments.flex_shard import BucketSpec, flex_shard
from torchtitan.experiments.flex_shard.example import (
    GroupedRaggedShard,
    make_grouped_ragged_placement_fn,
    make_ragged_placement_fn,
    per_param_ragged_placements,
    RaggedShard,
)
from torchtitan.experiments.flex_shard.flex_shard.bucket_storage import (
    ParamInfo,
    ShardedBucketStorage,
)
from torchtitan.experiments.flex_shard.tests.common import single_rank_cpu_mesh


device_type = torch.device(get_devtype())


def _make_param_info(
    fqn: str,
    tensor: torch.Tensor,
    placement: RaggedShard,
    rank: int,
    world_size: int,
) -> ParamInfo:
    local_shape = placement.compute_local_shape(tensor.shape, rank, world_size)
    local_numel = placement.compute_local_numel(tensor.shape, rank, world_size)
    return ParamInfo(
        fqn=fqn,
        global_shape=tensor.shape,
        global_stride=tensor.stride(),
        dtype=tensor.dtype,
        requires_grad=True,
        placements=(placement,),
        local_shape=local_shape,
        local_numel=local_numel,
        storage_nbytes=local_numel * tensor.dtype.itemsize,
        global_numel=tensor.numel(),
    )


def _expected_grouped_local(tensor: torch.Tensor, info: ParamInfo) -> torch.Tensor:
    assert info.bucket_layout is not None
    param_layout = info.bucket_layout.param_layouts[info.fqn]
    start = param_layout.local_global_offset - param_layout.param_offset
    return (
        tensor.contiguous()
        .view(-1)[start : start + info.local_numel]
        .view(info.local_shape)
    )


class _TinyRaggedModule(nn.Module):
    def __init__(self, device: torch.device | str) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.arange(16, dtype=torch.float32, device=device).view(4, 4)
        )
        self.bias = nn.Parameter(torch.arange(4, dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.t() + self.bias


class TestRaggedShardPlacement(TestCase):
    def test_equality_hash_and_repr_include_ragged_semantics(self):
        placement = RaggedShard(dims=(0,), local_units=(1, 2))

        self.assertEqual(placement, RaggedShard(dims=(0,), local_units=(1, 2)))
        self.assertNotEqual(placement, RaggedShard(dims=(0,), local_units=(2, 1)))
        self.assertEqual(hash(placement), hash(RaggedShard((0,), (1, 2))))
        self.assertEqual(repr(placement), "RaggedShard(dims=(0,), local_units=(1, 2))")

    def test_extracts_prefix_dim_shards_by_local_units(self):
        placement = RaggedShard(dims=(0,), local_units=(1, 2, 1, 0))
        param = torch.arange(24, dtype=torch.float32).view(8, 3)
        expected = [
            param[0:2],
            param[2:6],
            param[6:8],
            param[8:8],
        ]

        for rank, expected_shard in enumerate(expected):
            self.assertEqual(
                placement.compute_local_shape(param.shape, rank, world_size=4),
                expected_shard.shape,
            )
            self.assertEqual(
                placement.extract_local_shard(param, rank, world_size=4),
                expected_shard,
            )

    def test_extracts_flattened_prefix_dims(self):
        placement = RaggedShard(dims=(0, 1), local_units=(1, 2, 1, 0))
        param = torch.arange(24, dtype=torch.float32).view(2, 4, 3)
        flat = param.contiguous().view(-1)
        expected = [
            flat[0:6].view(2, 3),
            flat[6:18].view(4, 3),
            flat[18:24].view(2, 3),
            flat[24:24].view(0, 3),
        ]

        for rank, expected_shard in enumerate(expected):
            self.assertEqual(
                placement.compute_local_shape(param.shape, rank, world_size=4),
                expected_shard.shape,
            )
            self.assertEqual(
                placement.extract_local_shard(param, rank, world_size=4),
                expected_shard,
            )

    def test_rejects_invalid_config_and_shapes(self):
        with self.assertRaisesRegex(ValueError, "prefix dims"):
            RaggedShard(dims=(1,), local_units=(1, 1))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            RaggedShard(dims=(0,), local_units=(1, -1))
        with self.assertRaisesRegex(ValueError, "at least one positive"):
            RaggedShard(dims=(0,), local_units=(0, 0))

        placement = RaggedShard(dims=(0,), local_units=(1, 2))
        with self.assertRaisesRegex(ValueError, "divisible"):
            placement.compute_local_shape(torch.Size([5, 2]), rank=0, world_size=2)
        with self.assertRaisesRegex(ValueError, "world size"):
            placement.compute_local_shape(torch.Size([6, 2]), rank=0, world_size=3)

    def test_per_param_ragged_placements_uses_mesh_size(self):
        with single_rank_cpu_mesh() as mesh:
            weight = nn.Parameter(torch.empty(2, 2))
            placements = per_param_ragged_placements([("weight", weight)], mesh)

        self.assertEqual(placements["weight"], (RaggedShard((0,), (1,)),))

    def test_rejects_mixed_ragged_placements_in_one_bucket(self):
        from torchtitan.experiments.flex_shard.flex_shard.utils import (
            _validate_bucket_uniform_dtype_and_placement,
        )

        assignments = [["weight", "bias"]]
        placements = {
            "weight": (RaggedShard(dims=(0,), local_units=(1, 3)),),
            "bias": (RaggedShard(dims=(0,), local_units=(2, 2)),),
        }
        buckets = [
            BucketSpec(
                ["*"],
                placement_fn=make_ragged_placement_fn(
                    dims=(0,),
                    local_units=(1, 3),
                ),
                reshard_after_forward=False,
            )
        ]
        named_params = [
            ("weight", nn.Parameter(torch.empty(4, 4))),
            ("bias", nn.Parameter(torch.empty(4))),
        ]

        with self.assertRaisesRegex(ValueError, "mixed placements"):
            _validate_bucket_uniform_dtype_and_placement(
                assignments,
                placements,
                buckets,
                named_params,
            )


class TestRaggedShardDistributed(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 2

    def _mesh(self):
        return init_device_mesh("cpu", (self.world_size,), mesh_dim_names=("fsdp",))

    def test_bucket_storage_materializes_ragged_local_shards(self):
        mesh = self._mesh()
        placement = RaggedShard(dims=(0,), local_units=(1, 3))

        class TinyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(
                    torch.arange(8, dtype=torch.float32).view(4, 2)
                )
                self.bias = nn.Parameter(torch.arange(4, dtype=torch.float32))

        model = TinyModule()
        named_params = list(model.named_parameters())
        original_params = {fqn: param.detach().clone() for fqn, param in named_params}
        placements = {fqn: (placement,) for fqn, _ in named_params}
        bucket_spec = BucketSpec(
            ["*"],
            placement_fn=make_ragged_placement_fn(
                dims=(0,),
                local_units=(1, 3),
            ),
            reshard_after_forward=False,
        )

        bucket_storage = ShardedBucketStorage.from_bucket(
            model,
            named_params,
            placements,
            mesh,
            torch.device("cpu"),
            bucket_spec,
        )

        current_params = dict(model.named_parameters())
        for fqn, info in bucket_storage.param_infos.items():
            expected = placement.extract_local_shard(
                original_params[fqn],
                self.rank,
                self.world_size,
            )
            self.assertEqual(info.local_shape, expected.shape)
            self.assertEqual(current_params[fqn], expected)
            self.assertEqual(bucket_storage.get_local_view(fqn), expected)

    def test_unshard_all_gathers_ragged_bucket(self):
        mesh = self._mesh()
        placement = RaggedShard(dims=(0,), local_units=(1, 3))
        weight = torch.arange(8, dtype=torch.float32).view(4, 2)
        bias = torch.arange(4, dtype=torch.float32)
        infos = [
            _make_param_info("weight", weight, placement, self.rank, self.world_size),
            _make_param_info("bias", bias, placement, self.rank, self.world_size),
        ]
        local_shards = [
            placement.extract_local_shard(weight, self.rank, self.world_size),
            placement.extract_local_shard(bias, self.rank, self.world_size),
        ]

        prepared = placement.prepare_unshard_bucket(local_shards, infos, mesh, None)
        placement.run_prepared_unshard(prepared)
        result = placement.finish_prepared_unshard(prepared).full_params

        self.assertEqual(result[0], weight)
        self.assertEqual(result[1], bias)

    def test_reduce_scatter_returns_ragged_local_grads(self):
        mesh = self._mesh()
        placement = RaggedShard(dims=(0,), local_units=(1, 3))
        weight_grad = torch.arange(8, dtype=torch.float32).view(4, 2)
        bias_grad = torch.arange(4, dtype=torch.float32)
        infos = [
            _make_param_info(
                "weight",
                weight_grad,
                placement,
                self.rank,
                self.world_size,
            ),
            _make_param_info(
                "bias",
                bias_grad,
                placement,
                self.rank,
                self.world_size,
            ),
        ]

        prepared = placement.prepare_reduce_grad(
            [weight_grad, bias_grad],
            infos,
            mesh,
            None,
        )
        result = placement.reduce_prepared_grad(prepared).sharded_grads

        self.assertEqual(
            result[0],
            placement.extract_local_shard(
                weight_grad,
                self.rank,
                self.world_size,
            ),
        )
        self.assertEqual(
            result[1],
            placement.extract_local_shard(
                bias_grad,
                self.rank,
                self.world_size,
            ),
        )

    def test_zero_unit_rank_participates_in_bucket_collectives(self):
        mesh = self._mesh()
        placement = RaggedShard(dims=(0,), local_units=(1, 0))
        weight = torch.arange(8, dtype=torch.float32).view(4, 2)
        info = _make_param_info("weight", weight, placement, self.rank, self.world_size)
        local_shard = placement.extract_local_shard(weight, self.rank, self.world_size)

        prepared_unshard = placement.prepare_unshard_bucket(
            [local_shard],
            [info],
            mesh,
            None,
        )
        placement.run_prepared_unshard(prepared_unshard)
        full_param = placement.finish_prepared_unshard(
            prepared_unshard,
        ).full_params[0]

        prepared_reduce = placement.prepare_reduce_grad(
            [weight],
            [info],
            mesh,
            None,
        )
        local_grad = placement.reduce_prepared_grad(prepared_reduce).sharded_grads[0]

        self.assertEqual(full_param, weight)
        self.assertEqual(local_grad, local_shard)

    def test_grouped_ragged_bucket_materializes_bucket_global_local_views(self):
        mesh = self._mesh()
        placement = GroupedRaggedShard(dims=(0,), local_units=(1, 3))

        class TinyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(
                    torch.arange(8, dtype=torch.float32).view(4, 2)
                )
                self.bias = nn.Parameter(torch.arange(4, dtype=torch.float32))

        model = TinyModule()
        named_params = list(model.named_parameters())
        original_params = {fqn: param.detach().clone() for fqn, param in named_params}
        placements = {fqn: (placement,) for fqn, _ in named_params}
        bucket_spec = BucketSpec(
            ["*"],
            placement_fn=make_grouped_ragged_placement_fn(
                dims=(0,),
                local_units=(1, 3),
            ),
            reshard_after_forward=False,
        )

        bucket_storage = ShardedBucketStorage.from_bucket(
            model,
            named_params,
            placements,
            mesh,
            torch.device("cpu"),
            bucket_spec,
        )

        current_params = dict(model.named_parameters())
        infos = bucket_storage.param_infos
        for fqn, info in infos.items():
            expected = _expected_grouped_local(original_params[fqn], info)
            self.assertEqual(current_params[fqn], expected)
            self.assertEqual(bucket_storage.get_local_view(fqn), expected)
            if expected.numel() > 0:
                self.assertEqual(
                    current_params[fqn].untyped_storage().data_ptr(),
                    bucket_storage.byte_storage.untyped_storage().data_ptr(),
                )

    def test_grouped_ragged_rejects_padding_only_local_bucket_range(self):
        mesh = self._mesh()
        placement = GroupedRaggedShard(dims=(0,), local_units=(1, 1))

        class TinyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(
                    torch.arange(4, dtype=torch.float32).view(1, 4)
                )

        model = TinyModule()
        named_params = list(model.named_parameters())
        placements = {fqn: (placement,) for fqn, _ in named_params}
        bucket_spec = BucketSpec(
            ["*"],
            placement_fn=make_grouped_ragged_placement_fn(
                dims=(0,),
                local_units=(1, 1),
            ),
            reshard_after_forward=False,
        )

        if self.rank == 0:
            bucket_storage = ShardedBucketStorage.from_bucket(
                model,
                named_params,
                placements,
                mesh,
                torch.device("cpu"),
                bucket_spec,
            )
            self.assertEqual(bucket_storage.total_bytes, 4 * torch.float32.itemsize)
        else:
            with self.assertRaisesRegex(ValueError, "contains only padding"):
                ShardedBucketStorage.from_bucket(
                    model,
                    named_params,
                    placements,
                    mesh,
                    torch.device("cpu"),
                    bucket_spec,
                )

    def test_grouped_ragged_unshard_is_view_in_and_view_out(self):
        mesh = self._mesh()
        placement = GroupedRaggedShard(dims=(0,), local_units=(1, 3))

        class TinyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(
                    torch.arange(8, dtype=torch.float32).view(4, 2)
                )
                self.bias = nn.Parameter(torch.arange(4, dtype=torch.float32))

        model = TinyModule()
        named_params = list(model.named_parameters())
        original_params = {fqn: param.detach().clone() for fqn, param in named_params}
        placements = {fqn: (placement,) for fqn, _ in named_params}
        bucket_spec = BucketSpec(
            ["*"],
            placement_fn=make_grouped_ragged_placement_fn(
                dims=(0,),
                local_units=(1, 3),
            ),
            reshard_after_forward=False,
        )
        bucket_storage = ShardedBucketStorage.from_bucket(
            model,
            named_params,
            placements,
            mesh,
            torch.device("cpu"),
            bucket_spec,
        )
        infos = [bucket_storage.param_infos[fqn] for fqn, _ in named_params]
        local_shards = [bucket_storage.get_local_view(fqn) for fqn, _ in named_params]

        prepared = placement.prepare_unshard_bucket(local_shards, infos, mesh, None)
        send_buf = prepared.buffers[0]
        self.assertEqual(
            send_buf.untyped_storage().data_ptr(),
            bucket_storage.byte_storage.untyped_storage().data_ptr(),
        )
        self.assertEqual(send_buf.data_ptr(), bucket_storage.byte_storage.data_ptr())

        placement.run_prepared_unshard(prepared)
        result = placement.finish_prepared_unshard(prepared).full_params
        gathered_bucket = prepared.buffers[1]

        for full_param, (fqn, original_param) in zip(
            result,
            named_params,
            strict=True,
        ):
            self.assertEqual(full_param, original_params[fqn])
            self.assertEqual(
                full_param.untyped_storage().data_ptr(),
                gathered_bucket.untyped_storage().data_ptr(),
            )
            self.assertNotEqual(full_param.data_ptr(), original_param.data_ptr())

    def test_grouped_ragged_reduce_scatter_returns_local_grad_views(self):
        mesh = self._mesh()
        placement = GroupedRaggedShard(dims=(0,), local_units=(1, 3))

        class TinyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.empty(4, 2))
                self.bias = nn.Parameter(torch.empty(4))

        model = TinyModule()
        named_params = list(model.named_parameters())
        placements = {fqn: (placement,) for fqn, _ in named_params}
        bucket_spec = BucketSpec(
            ["*"],
            placement_fn=make_grouped_ragged_placement_fn(
                dims=(0,),
                local_units=(1, 3),
            ),
            reshard_after_forward=False,
        )
        bucket_storage = ShardedBucketStorage.from_bucket(
            model,
            named_params,
            placements,
            mesh,
            torch.device("cpu"),
            bucket_spec,
        )
        weight_grad = torch.arange(8, dtype=torch.float32).view(4, 2)
        bias_grad = torch.arange(4, dtype=torch.float32)
        grads = [weight_grad, bias_grad]
        infos = [bucket_storage.param_infos[fqn] for fqn, _ in named_params]

        prepared = placement.prepare_reduce_grad(grads, infos, mesh, None)
        reduce_result = placement.reduce_prepared_grad(prepared)
        sharded_grads = reduce_result.sharded_grads
        recv_buf = reduce_result.buffers[0]

        for grad, sharded_grad, info in zip(grads, sharded_grads, infos, strict=True):
            self.assertEqual(sharded_grad, _expected_grouped_local(grad, info))
            if sharded_grad.numel() > 0:
                self.assertEqual(
                    sharded_grad.untyped_storage().data_ptr(),
                    recv_buf.untyped_storage().data_ptr(),
                )


class TestRaggedShardRuntime(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_flex_shard_forward_backward_on_cuda_mesh(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        placement = RaggedShard(dims=(0,), local_units=(1, 3))
        reference = _TinyRaggedModule(device_type.type)
        model = _TinyRaggedModule(device_type.type)
        for param in [*reference.parameters(), *model.parameters()]:
            dist.broadcast(param.data, src=0)
        x = torch.arange(8, dtype=torch.float32, device=device_type).view(2, 4)
        dist.broadcast(x, src=0)

        ref_output = reference(x)
        ref_output.sum().backward()
        for param in reference.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        original_params = {
            fqn: param.detach().clone() for fqn, param in model.named_parameters()
        }

        flex_shard(
            model,
            mesh,
            buckets=[
                BucketSpec(
                    ["*"],
                    placement_fn=make_ragged_placement_fn(
                        dims=(0,),
                        local_units=(1, 3),
                    ),
                    reshard_after_forward=False,
                )
            ],
        )
        output = model(x)
        output.sum().backward()

        self.assertEqual(output, ref_output.detach())
        reference_params = dict(reference.named_parameters())
        state_dict = model.state_dict()
        for fqn, param in model.named_parameters():
            expected_param = placement.extract_local_shard(
                original_params[fqn],
                self.rank,
                self.world_size,
            )
            self.assertEqual(param.detach(), expected_param)
            self.assertEqual(state_dict[fqn], expected_param)
            param_grad = param.grad
            ref_grad = reference_params[fqn].grad
            self.assertIsNotNone(param_grad)
            self.assertIsNotNone(ref_grad)
            assert param_grad is not None
            assert ref_grad is not None
            expected_grad = placement.extract_local_shard(
                ref_grad.detach(),
                self.rank,
                self.world_size,
            )
            self.assertEqual(param_grad.detach(), expected_grad)

    @skip_if_lt_x_gpu(2)
    def test_grouped_ragged_flex_shard_forward_backward_on_cuda_mesh(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        reference = _TinyRaggedModule(device_type.type)
        model = _TinyRaggedModule(device_type.type)
        for param in [*reference.parameters(), *model.parameters()]:
            dist.broadcast(param.data, src=0)
        x = torch.arange(8, dtype=torch.float32, device=device_type).view(2, 4)
        dist.broadcast(x, src=0)

        ref_output = reference(x)
        ref_output.sum().backward()
        for param in reference.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        original_params = {
            fqn: param.detach().clone() for fqn, param in model.named_parameters()
        }

        flex_shard(
            model,
            mesh,
            buckets=[
                BucketSpec(
                    ["*"],
                    placement_fn=make_grouped_ragged_placement_fn(
                        dims=(0,),
                        local_units=(1, 3),
                    ),
                    reshard_after_forward=False,
                )
            ],
        )
        output = model(x)
        output.sum().backward()

        self.assertEqual(output, ref_output.detach())
        bucket_storage = model.sharded_bucket_storages[0]
        reference_params = dict(reference.named_parameters())
        for fqn, param in model.named_parameters():
            info = bucket_storage.param_infos[fqn]
            expected_param = _expected_grouped_local(original_params[fqn], info)
            self.assertEqual(param.detach(), expected_param)
            param_grad = param.grad
            ref_grad = reference_params[fqn].grad
            self.assertIsNotNone(param_grad)
            self.assertIsNotNone(ref_grad)
            assert param_grad is not None
            assert ref_grad is not None
            expected_grad = _expected_grouped_local(ref_grad.detach(), info)
            self.assertEqual(param_grad.detach(), expected_grad)


if __name__ == "__main__":
    run_tests()
