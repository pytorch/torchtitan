# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import spmd_types as spmd
from torch.distributed.tensor import Replicate, Shard

from torchtitan.distributed.parallel_dims import MeshAxisName, SpmdLayout
from torchtitan.protocols.sharding import resolve_placements


class _Mesh:
    mesh_dim_names = ("dp", "tp")

    def size(self, axis: int) -> int:
        return (2, 2)[axis]


class ResolvePlacementsTest(unittest.TestCase):
    def test_logical_dp_axis(self) -> None:
        layout = SpmdLayout(
            {
                MeshAxisName.DP: spmd.R,
                MeshAxisName.TP: spmd.S(0),
            }
        )

        placements = resolve_placements(layout, _Mesh())  # type: ignore[arg-type]

        self.assertEqual(placements, (Replicate(), Shard(0)))


if __name__ == "__main__":
    unittest.main()
