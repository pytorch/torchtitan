# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.models.common.activation import Sigmoid

from torchtitan.models.common.linear import RouterGateLinear
from torchtitan.models.deepseek_v3 import deepseekv3_configs
from torchtitan.models.deepseek_v3.moe import DeepSeekV3Router
from torchtitan.models.deepseek_v3.sharding import set_deepseek_v3_sharding_config


class TestDeepSeekV3Router(unittest.TestCase):
    def test_select_experts_limits_choices_to_selected_groups(self):
        router = DeepSeekV3Router.Config(
            num_experts=4,
            gate=RouterGateLinear.Config(in_features=4, out_features=4),
            score_func=Sigmoid.Config(),
            num_expert_groups=2,
            num_limited_groups=1,
            top_k=1,
        ).build()

        scores_TE = torch.tensor([[0.51, 0.49, 0.90, 0.00]])

        torch.testing.assert_close(
            router._select_experts(scores_TE),
            torch.tensor([[0]]),
        )

    def test_model_config_uses_deepseek_v3_router(self):
        build_config, _ = deepseekv3_configs["236B"]
        config = build_config(
            attn_backend="flex",
            moe_comm_backend="standard",
            seq_len=2048,
        )

        router_config = config.layers[1].moe.router
        self.assertIsInstance(router_config, DeepSeekV3Router.Config)
        self.assertEqual(router_config.num_expert_groups, 8)
        self.assertEqual(router_config.num_limited_groups, 3)

    def test_attention_owns_shared_input_gather(self):
        build_config, _ = deepseekv3_configs["debugmodel"]
        config = build_config(
            attn_backend="flex",
            moe_comm_backend="standard",
            seq_len=128,
        )

        set_deepseek_v3_sharding_config(config, enable_sp=True, enable_ep=True)
        attention_config = config.layers[0].attention
        assert attention_config.sharding_config is not None
        self.assertIsNotNone(attention_config.sharding_config.in_src_shardings)
        self.assertIsNone(attention_config.sharding_config.in_dst_shardings)


if __name__ == "__main__":
    unittest.main()
