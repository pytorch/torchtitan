#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_fsdp import FSDPTestMultiThread
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.experiments.flex_shard import BucketSpec
from torchtitan.experiments.flex_shard.example import (
    make_ragged_placement_fn,
    per_param_ragged_placements,
    RaggedShard,
)
from torchtitan.experiments.flex_shard.flex_shard.bucket_storage import (
    ParamInfo,
    ShardedBucketStorage,
)
from torchtitan.experiments.flex_shard.tests.common import single_rank_cpu_mesh


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


if __name__ == "__main__":
    run_tests()
