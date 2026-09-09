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
        ).model

        with torch.device("meta"):
            model = model_config.build()

        self.assertEqual(
            model_config.get_nparams_and_flops(model, seq_len=4096),
            (290_942_278_866, 92_762_352_876),
        )


if __name__ == "__main__":
    unittest.main()
