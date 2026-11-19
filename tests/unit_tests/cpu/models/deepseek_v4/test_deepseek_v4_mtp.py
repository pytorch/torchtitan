# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.deepseek_v4 import build_model_config
from torchtitan_recipes.tests.models.deepseek_v4 import (
    deepseek_v4_debugmodel,
    deepseek_v4_mtp_debugmodel,
)


class TestDeepSeekV4MTPConfig(unittest.TestCase):
    def test_dataloader_mtp_depth_matches_model(self):
        for config in (deepseek_v4_debugmodel(), deepseek_v4_mtp_debugmodel()):
            self.assertEqual(
                config.dataloader.num_mtp_layers, config.model.n_mtp_layers
            )

    def test_mtp_debugmodel_builds_mtp_layers(self):
        config = deepseek_v4_mtp_debugmodel()
        model_config = config.model
        self.assertEqual(model_config.n_mtp_layers, 1)
        self.assertIsNotNone(model_config.mtp_layers)
        self.assertEqual(len(model_config.mtp_layers), 1)

    def test_mtp_metadata_remains_replicated_at_block_boundary(self):
        config = build_model_config(
            "debugmodel",
            n_mtp_layers=1,
            parallelism=ParallelismConfig(
                enable_sequence_parallel=True, expert_parallel_degree=2
            ),
        )

        assert config.mtp_layers is not None
        mtp_config = config.mtp_layers[0].sharding_config
        assert mtp_config is not None
        self.assertIsNotNone(mtp_config.input_shardings)


if __name__ == "__main__":
    unittest.main()
