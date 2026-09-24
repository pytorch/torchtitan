# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.models.deepseek_v4 import model_registry


class TestDeepSeekV4Flops(unittest.TestCase):
    def test_flash_mtp_4k_model_flops(self):
        model_config = model_registry(
            "deepseek_v4_flash",
            n_mtp_layers=1,
        )

        with torch.device("meta"):
            model = model_config.build()

        estimator = model_config.build_flops_estimator(model, seq_len=4096)
        self.assertIsNotNone(estimator)
        assert estimator is not None
        estimated_flops = estimator({"input": torch.empty(4096, dtype=torch.long)})
        self.assertIsInstance(estimated_flops, int)
        self.assertEqual(estimated_flops, 92_762_352_876 * 4096)


if __name__ == "__main__":
    unittest.main()
