# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtitan.components.optim import DistMuon
from torchtitan.distributed.flex_shard import BlockShard
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan_recipes.tests.models.kimi_k2_7 import kimi_k2_5_debugmodel


class KimiK2DistMuonConfigTest(unittest.TestCase):
    def test_parallelism_variant_rebuilds_compute_layouts_and_buckets(self) -> None:
        config = kimi_k2_5_debugmodel()
        config.parallelism.tensor_parallel_degree = 2
        config.parallelism.expert_parallel_degree = 2
        config.__post_init__()

        dist_muon = next(
            optimizer
            for optimizer in config.optim.optimizer.optimizers
            if isinstance(optimizer, DistMuon.Config)
        )
        query_fqn = next(
            fqn
            for fqn in dist_muon.compute_sharding_by_fqn
            if fqn.endswith("attention.wq.weight")
        )
        query_layout = dist_muon.compute_sharding_by_fqn[query_fqn]
        self.assertIsInstance(
            query_layout.shardings_by_mesh_axis[MeshAxisName.TP.value],
            BlockShard,
        )
        self.assertEqual(
            query_layout.shard_order_by_tensor_dim[0],
            (MeshAxisName.TP.value, MeshAxisName.DP_SHARD.value),
        )
        self.assertTrue(
            any(bucket.name.endswith(".dp-only") for bucket in dist_muon.bucket_configs)
        )

        expert_fqn = next(
            fqn
            for fqn in dist_muon.compute_sharding_by_fqn
            if ".moe.routed_experts." in fqn
        )
        expert_layout = dist_muon.compute_sharding_by_fqn[expert_fqn]
        self.assertEqual(
            set(expert_layout.shardings_by_mesh_axis),
            {MeshAxisName.EDP_SHARD.value, MeshAxisName.EP.value},
        )
