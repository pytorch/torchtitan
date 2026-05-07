# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from torchtitan.experiments.flex_shard import BucketSpec, is_flex_shard_param
from torchtitan.experiments.flex_shard.bucket_storage import (
    _create_param_infos,
    _materialize_bucket_storages,
)
from torchtitan.experiments.flex_shard.placements import Shard


class _FakeRankMesh:
    def __init__(self, rank: int = 0, world_size: int = 2) -> None:
        self.rank = rank
        self.world_size = world_size

    def get_local_rank(self) -> int:
        return self.rank

    def size(self) -> int:
        return self.world_size


class TestFlexShardBucketStorage(unittest.TestCase):
    def test_create_param_infos_uses_sequential_byte_offsets(self):
        mesh = _FakeRankMesh(rank=0, world_size=2)
        named_params = [
            ("weight", nn.Parameter(torch.empty(6, 4))),
            ("bias", nn.Parameter(torch.empty(6))),
        ]
        placements = {fqn: (Shard(0),) for fqn, _ in named_params}

        infos, total_bytes = _create_param_infos(named_params, mesh, placements)

        weight = infos["weight"]
        bias = infos["bias"]
        self.assertEqual(weight.local_shape, torch.Size([3, 4]))
        self.assertEqual(weight.byte_offset, 0)
        self.assertEqual(bias.local_shape, torch.Size([3]))
        self.assertEqual(bias.byte_offset, 3 * 4 * torch.float32.itemsize)
        self.assertEqual(total_bytes, (3 * 4 + 3) * torch.float32.itemsize)

    def test_materialized_params_are_views_into_bucket_storage(self):
        mesh = _FakeRankMesh(rank=1, world_size=2)
        model = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 2))
        named_params = list(model.named_parameters())
        placements = {fqn: (Shard(0),) for fqn, _ in named_params}
        buckets = [
            BucketSpec(["0.*"], reshard_after_forward=False),
            BucketSpec(["1.*"], reshard_after_forward=False),
        ]

        storages, fqn_to_bucket_spec = _materialize_bucket_storages(
            model,
            named_params,
            [["0.weight", "0.bias"], ["1.weight", "1.bias"]],
            buckets,
            placements,
            mesh,
            torch.device("cpu"),
        )

        self.assertEqual(len(storages), 2)
        self.assertIs(fqn_to_bucket_spec["0.weight"], buckets[0])
        self.assertIs(fqn_to_bucket_spec["1.weight"], buckets[1])

        current_params = dict(model.named_parameters())
        for storage in storages:
            storage_ptr = storage.byte_storage.untyped_storage().data_ptr()
            for fqn, info in storage.param_infos.items():
                param = current_params[fqn]
                local_view = storage.get_local_view(fqn)

                self.assertEqual(param.shape, info.local_shape)
                self.assertEqual(
                    param.untyped_storage().data_ptr(),
                    storage_ptr,
                )
                self.assertEqual(
                    local_view.untyped_storage().data_ptr(),
                    storage_ptr,
                )
                torch.testing.assert_close(param, local_view)
                self.assertTrue(is_flex_shard_param(param))


if __name__ == "__main__":
    unittest.main()
