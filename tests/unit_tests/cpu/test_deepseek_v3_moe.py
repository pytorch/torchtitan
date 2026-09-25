# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.config.transform import AsyncTensorParallelTransform
from torchtitan.models.common.activation import Sigmoid

from torchtitan.models.common.async_linear import AsyncRowParallelLinear
from torchtitan.models.common.linear import (
    ColumnParallelLinear,
    Linear,
    RouterGateLinear,
    RowParallelLinear,
)
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

        shared_experts = config.layers[1].moe.shared_experts
        self.assertIsNotNone(shared_experts)
        assert shared_experts is not None
        self.assertIs(type(shared_experts.w13), ColumnParallelLinear.Config)
        self.assertIs(type(shared_experts.w2), Linear.Config)

    def test_attention_owns_input_gather_and_wo_owns_output_reduction(self):
        build_config, _ = deepseekv3_configs["debugmodel"]
        config = build_config(
            attn_backend="flex",
            moe_comm_backend="standard",
            seq_len=128,
        )

        set_deepseek_v3_sharding_config(config, enable_sp=True, enable_ep=True)
        attention_config = config.layers[0].attention
        self.assertIs(type(attention_config.wo), RowParallelLinear.Config)
        assert attention_config.sharding_config is not None
        self.assertIsNotNone(attention_config.sharding_config.in_src_shardings)
        self.assertIsNone(attention_config.sharding_config.in_dst_shardings)
        assert attention_config.wo.sharding_config is not None
        self.assertIsNotNone(attention_config.wo.sharding_config.out_src_shardings)
        self.assertIsNone(attention_config.wo.sharding_config.out_dst_shardings)

        AsyncTensorParallelTransform(enable_sequence_parallel=True).transform(
            attention_config
        )
        self.assertIs(type(attention_config.wo), AsyncRowParallelLinear.Config)


if __name__ == "__main__":
    unittest.main()
