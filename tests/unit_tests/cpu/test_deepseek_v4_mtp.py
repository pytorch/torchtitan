# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import spmd_types as spmd

from torchtitan.models.deepseek_v4.config_registry import deepseek_v4_mtp_debugmodel


class TestDeepSeekV4MTPConfig(unittest.TestCase):
    def test_mtp_debugmodel_builds_mtp_layers(self):
        config = deepseek_v4_mtp_debugmodel()
        model_config = config.model_spec.model
        self.assertEqual(model_config.n_mtp_layers, 1)
        self.assertIsNotNone(model_config.mtp_layers)
        self.assertEqual(len(model_config.mtp_layers), 1)

    def test_mtp_spmd_layouts_match_runtime_tensor_ranks(self):
        for enable_sp in (False, True):
            config = deepseek_v4_mtp_debugmodel(seq_len=32)
            config.parallelism.tensor_parallel_degree = 2
            config.parallelism.enable_sequence_parallel = enable_sp
            config.parallelism.spmd_backend = "spmd_types"
            model_config = config.model_spec.model
            model_config.update_from_config(config=config)

            mtp_config = model_config.mtp_layers[0]
            inputs = mtp_config.sharding_config.in_src_shardings
            activation_spec = spmd.PartitionSpec(
                ("dp", "cp", "tp") if enable_sp else ("dp", "cp"),
                None,
            )
            hc_activation_spec = spmd.PartitionSpec(
                ("dp", "cp", "tp") if enable_sp else ("dp", "cp"),
                None,
                None,
            )
            self.assertEqual(
                inputs["mtp_input_embed"].partition_spec,
                activation_spec,
            )
            self.assertEqual(
                inputs["prev_hc_hidden"].partition_spec,
                hc_activation_spec,
            )
            self.assertEqual(
                inputs["mtp_input_ids_T"].partition_spec,
                spmd.PartitionSpec(("dp", "cp")),
            )
            self.assertEqual(
                inputs["mtp_input_valid_mask"].partition_spec,
                spmd.PartitionSpec(("dp", "cp")),
            )
            self.assertEqual(
                mtp_config.hc_head.sharding_config.in_src_shardings[
                    "x"
                ].partition_spec,
                hc_activation_spec,
            )


if __name__ == "__main__":
    unittest.main()
