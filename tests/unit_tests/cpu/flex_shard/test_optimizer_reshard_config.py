# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torch.distributed.tensor import Shard

from torchtitan.distributed.flex_shard import BlockShard, ComputeLayout


class TestComputeLayout(unittest.TestCase):
    def test_ordered_block_shards_require_one_block_size(self):
        with self.assertRaisesRegex(ValueError, "equal BlockShard block sizes"):
            ComputeLayout(
                shardings_by_mesh_axis={
                    "dp_shard": BlockShard(0, 2),
                    "tp": BlockShard(0, 4),
                },
                shard_order_by_tensor_dim={0: ("tp", "dp_shard")},
            )

    def test_shard_order_cannot_mix_shard_types(self):
        with self.assertRaisesRegex(ValueError, "cannot mix Shard and BlockShard"):
            ComputeLayout(
                shardings_by_mesh_axis={
                    "dp_shard": BlockShard(0, 2),
                    "tp": Shard(0),
                },
                shard_order_by_tensor_dim={0: ("tp", "dp_shard")},
            )
