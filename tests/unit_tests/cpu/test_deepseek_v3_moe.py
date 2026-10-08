# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import types
import unittest
from unittest.mock import patch

import spmd_types as spmd
import torch
import torch_remat as remat

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import AsyncTensorParallelTransform
from torchtitan.models.common.activation import Sigmoid

from torchtitan.models.common.async_linear import AsyncRowParallelLinear
from torchtitan.models.common.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    SharedExpertRowParallelLinear,
)
from torchtitan.models.deepseek_v3 import build_model_config, MODEL_FLAVORS
from torchtitan.models.deepseek_v3.moe import DeepSeekV3Router
from torchtitan.models.deepseek_v3.mtp import MTPTransformerBlock
from torchtitan.models.deepseek_v3.sharding import set_deepseek_v3_sharding_config


class TestDeepSeekV3Router(unittest.TestCase):
    def test_mtp_valid_mask_is_explicitly_sharded(self):
        mtp = MTPTransformerBlock.__new__(MTPTransformerBlock)
        valid_mask_T = torch.ones(4, dtype=torch.bool)
        tp_group = object()

        with (
            patch(
                "torchtitan.models.deepseek_v3.mtp.spmd_dense_sp_enabled",
                return_value=True,
            ),
            patch(
                "torchtitan.models.deepseek_v3.mtp.spmd_mesh_group",
                return_value=tp_group,
            ),
            patch(
                "torchtitan.models.deepseek_v3.mtp.spmd.redistribute",
                side_effect=lambda tensor, *_args, **_kwargs: tensor,
            ) as redistribute,
        ):
            actual_valid_mask_T = mtp._maybe_shard_mtp_valid_mask_across_tp(
                valid_mask_T
            )

        self.assertIs(actual_valid_mask_T, valid_mask_T)
        redistribute.assert_called_once_with(
            valid_mask_T,
            tp_group,
            src=spmd.R,
            dst=spmd.S(0),
            backward_options={"op_dtype": valid_mask_T.dtype},
        )

    def test_mtp_mask_remains_replicated_at_block_boundary(self):
        config = build_model_config(
            "debugmodel",
            seq_len=128,
            num_mtp_layers=1,
        )
        config.set_sharding_(ParallelismConfig())

        mtp_config = config.mtp_layers[0].sharding_config
        assert mtp_config is not None
        assert mtp_config.in_src_shardings is not None
        self.assertIn("mtp_input_valid_mask", mtp_config.in_src_shardings)
        self.assertIsNone(mtp_config.in_dst_shardings)

    def test_select_experts_limits_choices_to_selected_groups(self):
        router = DeepSeekV3Router.Config(
            num_experts=4,
            gate=HiMidLoLinear.Config(in_features=4, out_features=4),
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

    def test_compiled_select_experts_matches_eager(self):
        router = DeepSeekV3Router.Config(
            num_experts=256,
            gate=HiMidLoLinear.Config(
                backward_mode="hi_mid_lo", in_features=4, out_features=256
            ),
            score_func=Sigmoid.Config(),
            num_expert_groups=8,
            num_limited_groups=4,
            top_k=8,
        ).build()
        generator = torch.Generator().manual_seed(0)
        scores_TE = torch.rand(512, 256, generator=generator)
        expert_bias_E = torch.randn(256, generator=generator) * 1e-2

        # The compiled branch replaces topk with argmax rounds; the backend="eager"
        # trace runs it without codegen. topk returns ids in an unspecified order.
        compiled_select = torch.compile(
            DeepSeekV3Router._select_experts, fullgraph=True, backend="eager"
        )
        torch.testing.assert_close(
            compiled_select(router, scores_TE, expert_bias_E).sort(dim=-1).values,
            router._select_experts(scores_TE, expert_bias_E).sort(dim=-1).values,
        )

    def test_rejects_limited_groups_with_fewer_than_top_k_experts(self):
        config = DeepSeekV3Router.Config(
            num_experts=16,
            gate=HiMidLoLinear.Config(
                backward_mode="hi_mid_lo", in_features=4, out_features=16
            ),
            score_func=Sigmoid.Config(),
            num_expert_groups=8,
            num_limited_groups=1,
            top_k=4,
        )
        with self.assertRaisesRegex(ValueError, "fewer than top_k"):
            config.build()

    def test_rejects_more_limited_groups_than_groups(self):
        config = DeepSeekV3Router.Config(
            num_experts=16,
            gate=HiMidLoLinear.Config(
                backward_mode="hi_mid_lo", in_features=4, out_features=16
            ),
            score_func=Sigmoid.Config(),
            num_expert_groups=4,
            num_limited_groups=5,
            top_k=4,
        )
        with self.assertRaisesRegex(ValueError, "must be <= num_expert_groups"):
            config.build()

    def test_compiled_route_counts_experts_once_under_region_ac(self):
        router = DeepSeekV3Router.Config(
            num_experts=64,
            gate=HiMidLoLinear.Config(
                backward_mode="hi_mid_lo", in_features=16, out_features=64
            ),
            score_func=Sigmoid.Config(),
            num_expert_groups=8,
            num_limited_groups=4,
            top_k=8,
        ).build()
        router.init_states(buffer_device=torch.device("cpu"))
        # Compile only this instance's _route, as the router region does. Under compile
        # torch_remat.is_recomputing() is always False, so a count inside the region
        # would run again in the RegionAC recompute pass.
        router._route = types.MethodType(
            torch.compile(
                DeepSeekV3Router._route.__wrapped__, fullgraph=True, backend="eager"
            ),
            router,
        )
        x_TD = torch.randn(32, 16, requires_grad=True)
        checkpointed_forward = remat.checkpoint(
            region_name="layers.0", preserve_rng_state=False
        )(lambda x_TD: router(x_TD)[0].sum())

        checkpointed_forward(x_TD).backward()

        self.assertEqual(router.tokens_per_expert_E.sum().item(), 32 * 8)

    def test_model_config_uses_deepseek_v3_router(self):
        config = build_model_config(
            "236B",
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
        self.assertIs(type(shared_experts.w2), SharedExpertRowParallelLinear.Config)

    def test_671b_recipe_compiles_the_router_region(self):
        from torchtitan_recipes.models.deepseek_v3 import deepseek_v3_671b

        self.assertIn("router", deepseek_v3_671b().model.local_compile_regions)
        # The model default stays without it: Kimi K2.x inherits that default.
        self.assertNotIn("router", build_model_config("671B").local_compile_regions)

    @unittest.skipUnless(importlib.util.find_spec("dist_moe"), "dist_moe not installed")
    def test_671b_dist_moe_recipe_compiles_the_router_region(self):
        from torchtitan_recipes.models.deepseek_v3 import deepseek_v3_671b_dist_moe_bf16

        config = deepseek_v3_671b_dist_moe_bf16()
        self.assertIn("router", config.model.local_compile_regions)

    def test_attention_owns_input_gather_and_wo_owns_output_reduction(self):
        build_config, _ = MODEL_FLAVORS["debugmodel"]
        config = build_config(
            attn_backend="flex",
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
