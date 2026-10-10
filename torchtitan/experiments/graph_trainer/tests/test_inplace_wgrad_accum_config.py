# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtitan_recipes.tests.models.deepseek_v3 import deepseek_v3_debugmodel

from torchtitan.experiments.graph_trainer.configs import to_graph_trainer_config
from torchtitan.experiments.graph_trainer.deepseek_v3 import build_model_config
from torchtitan.experiments.graph_trainer.deepseek_v3.model import (
    GraphTrainerDeepSeekV3Model,
)
from torchtitan.models.common.linear import Linear


def _linear_configs(model):
    return [config for _, config, _, _ in model.traverse(Linear.Config)]


class TestInplaceWgradAccumConfig(unittest.TestCase):
    def test_conversion_disables_and_validation_rejects(self):
        # Every Linear, including the HiMidLoLinear router gates, is on by default in eager.
        base = deepseek_v3_debugmodel()
        linears = _linear_configs(base.model)
        self.assertTrue(linears)
        self.assertTrue(all(linear.inplace_wgrad_accum for linear in linears))

        config = to_graph_trainer_config(base, GraphTrainerDeepSeekV3Model.Config)
        linears = _linear_configs(config.model)
        self.assertTrue(linears)
        self.assertFalse(any(linear.inplace_wgrad_accum for linear in linears))

        # A transform applied after conversion may build one with the eager default.
        linears[0].inplace_wgrad_accum = True
        with self.assertRaisesRegex(ValueError, "inplace_wgrad_accum=False"):
            config.__post_init__()

    def test_flavor_builder_disables(self):
        linears = _linear_configs(build_model_config("debugmodel"))
        self.assertTrue(linears)
        self.assertFalse(any(linear.inplace_wgrad_accum for linear in linears))


if __name__ == "__main__":
    unittest.main()
