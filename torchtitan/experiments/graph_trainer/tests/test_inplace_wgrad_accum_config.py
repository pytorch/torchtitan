# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtitan_recipes.tests.models.deepseek_v3 import deepseek_v3_debugmodel

from torchtitan.experiments.graph_trainer.common_utils import (
    inplace_wgrad_accum_configs,
)
from torchtitan.experiments.graph_trainer.configs import to_graph_trainer_config
from torchtitan.experiments.graph_trainer.deepseek_v3 import build_model_config
from torchtitan.experiments.graph_trainer.deepseek_v3.model import (
    GraphTrainerDeepSeekV3Model,
)


class TestInplaceWgradAccumConfig(unittest.TestCase):
    def test_conversion_disables_and_validation_rejects(self):
        # The DeepSeek-V3 MoE router gates are HiMidLoLinear, on by default in eager.
        base = deepseek_v3_debugmodel()
        gates = inplace_wgrad_accum_configs(base.model)
        self.assertTrue(gates)
        self.assertTrue(all(gate.inplace_wgrad_accum for _, gate in gates))

        config = to_graph_trainer_config(base, GraphTrainerDeepSeekV3Model.Config)
        gates = inplace_wgrad_accum_configs(config.model)
        self.assertTrue(gates)
        self.assertFalse(any(gate.inplace_wgrad_accum for _, gate in gates))

        # A transform applied after conversion may build one with the eager default.
        gates[0][1].inplace_wgrad_accum = True
        with self.assertRaisesRegex(ValueError, "inplace_wgrad_accum=False"):
            config.__post_init__()

    def test_flavor_builder_disables(self):
        gates = inplace_wgrad_accum_configs(build_model_config("debugmodel"))
        self.assertTrue(gates)
        self.assertFalse(any(gate.inplace_wgrad_accum for _, gate in gates))


if __name__ == "__main__":
    unittest.main()
