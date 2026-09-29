# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtitan.components.optimizer import DistMuon
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.flex_shard import BlockShard
from torchtitan.models.kimi_k2_7 import model_registry
from torchtitan.models.kimi_k2_7.config_registry import _dist_muon_optimizer


class TestKimiK25DistMuonConfig(unittest.TestCase):
    def test_feed_forward_matrices_use_block_shard(self):
        model_config = model_registry("debugmodel", enable_sp=True, seq_len=128)
        optimizer = _dist_muon_optimizer(
            model_config,
            muon_lr=1e-3,
            adamw_lr=1e-3,
            parallelism=ParallelismConfig(),
        )
        muon_config = next(
            config
            for config in optimizer.optimizers
            if isinstance(config, DistMuon.Config)
        )
        compute_layouts = muon_config.compute_sharding_by_fqn

        dense_feed_forward = model_config.layers[0].feed_forward
        assert dense_feed_forward is not None
        moe = model_config.layers[1].moe
        assert moe is not None and moe.shared_experts is not None
        for prefix, feed_forward in (
            ("layers.0.feed_forward", dense_feed_forward),
            ("layers.1.moe.shared_experts", moe.shared_experts),
        ):
            for projection, linear in (
                ("w13", feed_forward.w13),
                ("w2", feed_forward.w2),
            ):
                layout = compute_layouts[f"{prefix}.{projection}.weight"]
                block_shard = layout.shardings_by_mesh_axis["dp_shard"]
                self.assertIsInstance(block_shard, BlockShard)
                assert isinstance(block_shard, BlockShard)
                self.assertEqual(
                    block_shard.block_sizes,
                    (linear.out_features * linear.in_features,),
                )
                self.assertEqual(
                    block_shard.num_blocks(
                        linear.num_linears * linear.out_features * linear.in_features
                    ),
                    linear.num_linears,
                )


if __name__ == "__main__":
    unittest.main()
