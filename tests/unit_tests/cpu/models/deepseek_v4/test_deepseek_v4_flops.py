# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.deepseek_v4 import build_model_config


class TestDeepSeekV4Flops(unittest.TestCase):
    def test_flash_mtp_4k_model_flops(self):
        model_config = build_model_config(
            "deepseek_v4_flash", n_mtp_layers=1, parallelism=ParallelismConfig()
        )

        with torch.device("meta"):
            model = model_config.build()

        self.assertEqual(model_config.get_parameter_counts(model)[0], 290_942_278_866)
        estimate = model_config.build_flops_estimator(model, seq_len=4096)
        self.assertEqual(estimate({"input": torch.empty(4096)}), 92_762_352_876 * 4096)


if __name__ == "__main__":
    unittest.main()
