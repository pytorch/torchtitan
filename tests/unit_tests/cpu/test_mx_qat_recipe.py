# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import types
import unittest
from unittest.mock import patch

import torch
from torchao.prototype.qat import MXFakeQuantizeConfig
from torchao.quantization.quantize_.common import KernelPreference
from torchtitan.config import ConfigManager
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel_mx_qat
from torchtitan.models.kimi_k3.model import KimiK3Model


class MXQATRecipeTest(unittest.TestCase):
    def test_standard_model_entrypoint_and_training_override(self):
        config = ConfigManager().parse_args(
            [
                "--module",
                "kimi_k3",
                "--config",
                "kimi_k3_debugmodel_mx_qat",
                "--training.steps",
                "1",
            ]
        )
        self.assertIsInstance(config.model, KimiK3Model.Config)
        self.assertEqual(config.training.steps, 1)
        self.assertFalse(config.checkpointer.initial_load_in_hf_quantized)
        self.assertTrue(type(config.model.layers[1].moe.routed_up)._owner._mx_qat)

    def test_custom_recipe_uses_existing_torchao_configs(self):
        def qat():
            return kimi_k3_debugmodel_mx_qat(
                seq_len=16,
                checkpoint_path="/tmp/packed-kimi",
                weight_fake_quant_config=MXFakeQuantizeConfig(
                    dtype=torch.float4_e2m1fn_x2,
                    kernel_preference=KernelPreference.AUTO,
                ),
                activation_fake_quant_config=MXFakeQuantizeConfig(
                    dtype=torch.float8_e4m3fn,
                    kernel_preference=KernelPreference.AUTO,
                ),
            )

        module = types.ModuleType("my_kimi_runs")
        module.qat = qat
        with patch.dict(sys.modules, {module.__name__: module}):
            config = ConfigManager().parse_args(
                ["--module", module.__name__, "--config", "qat"]
            )
        self.assertEqual(config.checkpointer.initial_load_path, "/tmp/packed-kimi")
        self.assertTrue(config.checkpointer.initial_load_in_hf)
        self.assertTrue(config.checkpointer.initial_load_in_hf_quantized)
        experts = list(config.model.traverse(GroupedExperts.Config))
        self.assertTrue(experts)
        for _, expert, _, _ in experts:
            self.assertEqual(
                expert.weight_fake_quant_config.kernel_preference, KernelPreference.AUTO
            )
            self.assertEqual(
                expert.activation_fake_quant_config.kernel_preference,
                KernelPreference.AUTO,
            )
        self.assertEqual(
            config.model.layers[
                1
            ].moe.routed_up.weight_fake_quant_config.kernel_preference,
            KernelPreference.AUTO,
        )


if __name__ == "__main__":
    unittest.main()
