# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace

import spmd_types as spmd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import _per_axis_types
from torchtitan.models.common.activation import Sigmoid, SiTUGLU, Softmax, SqrtSoftplus
from torchtitan.models.common.config_utils import (
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
)
from torchtitan.models.common.decoder_sharding import token_id_placement
from torchtitan.models.common.linear import RouterGateLinear
from torchtitan.models.common.moe import (
    GroupedExperts,
    MicrobatchWiseLoadBalanceLoss,
    TokenChoiceTopKRouter,
)
from torchtitan.models.common.moe_sharding import (
    _moe_sharding_config,
    _router_sharding_config,
    set_moe_block_padding_mask_sharding,
    set_moe_sharding_config,
)
from torchtitan.protocols.sharding import ShardingConfig


class _PassthroughRoutedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_tokens_per_expert_E = None

    def forward(
        self,
        x_TD,
        topk_scores_TK,
        topk_expert_ids_TK,
        num_local_tokens_per_expert_E,
    ):
        self.num_tokens_per_expert_E = num_local_tokens_per_expert_E
        return x_TD


class _CapturingAuxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.routing_map_TE = None

    def forward(self, scores_TE, routing_map_TE, *, carrier, padding_mask_T=None):
        del scores_TE, padding_mask_T
        self.routing_map_TE = routing_map_TE
        return carrier


class TestMoE(unittest.TestCase):
    def test_make_router_config_requires_score_func(self):
        with self.assertRaisesRegex(TypeError, "score_func"):
            make_router_config(
                dim=4,
                num_experts=4,
                gate_param_init={"weight": nn.init.zeros_},
            )

    def test_token_choice_router_requires_score_func(self):
        with self.assertRaisesRegex(TypeError, "score_func"):
            TokenChoiceTopKRouter.Config(
                num_experts=4,
                gate=RouterGateLinear.Config(in_features=4, out_features=4),
            )

    def test_grouped_experts_use_configured_activation(self):
        activation_fn = SiTUGLU.Config(beta=4.0, linear_beta=25.0)
        experts = GroupedExperts.Config(
            dim=4,
            hidden_dim=8,
            num_experts=2,
            activation_fn=activation_fn,
        ).build()
        gate_RF = torch.randn(3, 8)
        up_RF = torch.randn(3, 8)

        expected_RF = activation_fn.build()(gate_RF, up_RF)
        actual_RF = experts.activation_fn(gate_RF, up_RF)
        torch.testing.assert_close(actual_RF, expected_RF)

    def test_token_choice_router_uses_normalization_epsilon(self):
        x_TD = torch.zeros(1, 4)
        expert_bias_E = torch.tensor([4.0, 3.0, 2.0, 1.0])
        route_norm_epsilon = 1.0
        route_scale = 4.0

        for route_norm in (False, True):
            with self.subTest(route_norm=route_norm):
                config = make_router_config(
                    dim=4,
                    num_experts=4,
                    score_func=Sigmoid.Config(),
                    gate_param_init={"weight": nn.init.zeros_},
                    top_k=2,
                    route_norm=route_norm,
                    route_norm_epsilon=route_norm_epsilon,
                    route_scale=route_scale,
                )
                router = config.build()
                with torch.no_grad():
                    router.gate.weight.zero_()

                (
                    actual_topk_scores_TK,
                    actual_topk_expert_ids_TK,
                    _,
                ) = router(x_TD, expert_bias_E=expert_bias_E)
                actual_scores_TE = torch.zeros_like(x_TD).scatter(
                    dim=-1,
                    index=actual_topk_expert_ids_TK,
                    src=actual_topk_scores_TK,
                )
                expected_scores_TE = torch.tensor(
                    [[1.0, 1.0, 0.0, 0.0]] if route_norm else [[2.0, 2.0, 0.0, 0.0]]
                )

                torch.testing.assert_close(
                    actual_scores_TE,
                    expected_scores_TE,
                    rtol=0,
                    atol=0,
                )

    def test_token_choice_router_uses_configured_score_functions(self):
        x_TD = torch.tensor([[-2.0, 0.0, 1.0, 3.0]], dtype=torch.bfloat16)
        x_fp32_TD = x_TD.float()
        cases = (
            ("sigmoid", Sigmoid.Config(), torch.sigmoid(x_fp32_TD)),
            ("softmax", Softmax.Config(), F.softmax(x_fp32_TD, dim=-1)),
            (
                "sqrtsoftplus",
                SqrtSoftplus.Config(),
                F.softplus(x_fp32_TD).sqrt(),
            ),
        )

        for name, score_func, expected_scores_TE in cases:
            with self.subTest(score_func=name):
                config = make_router_config(
                    dim=4,
                    num_experts=4,
                    gate_param_init={"weight": nn.init.zeros_},
                    score_func=score_func,
                    top_k=4,
                )
                router = config.build()
                with torch.no_grad():
                    router.gate.weight.copy_(torch.eye(4))

                (
                    actual_topk_scores_TK,
                    actual_topk_expert_ids_TK,
                    _,
                ) = router(x_TD)
                actual_scores_TE = torch.zeros_like(expected_scores_TE).scatter(
                    dim=-1,
                    index=actual_topk_expert_ids_TK,
                    src=actual_topk_scores_TK,
                )

                self.assertIs(actual_topk_scores_TK.dtype, torch.float32)
                torch.testing.assert_close(
                    actual_scores_TE,
                    expected_scores_TE,
                    rtol=0,
                    atol=0,
                )

    def _build_moe(self):
        num_experts = 2
        dim = 4
        top_k = 1
        moe = make_moe_config(
            num_experts=num_experts,
            router=make_router_config(
                dim=dim,
                num_experts=num_experts,
                gate_param_init={"weight": nn.init.zeros_},
                score_func=Sigmoid.Config(),
                top_k=top_k,
            ),
            routed_experts=make_routed_experts_config(
                dim=dim,
                hidden_dim=8,
                num_experts=num_experts,
                top_k=top_k,
                param_init={},
                comm_backend="standard",
            ),
        ).build()
        with torch.no_grad():
            moe.router.gate.weight.zero_()
        moe.routed_experts = _PassthroughRoutedExperts()
        return moe

    def test_eval_forward_does_not_accumulate_tokens_per_expert(self):
        dim = 4
        top_k = 1
        moe = self._build_moe()

        x_TD = torch.randn(6, dim)
        moe.train()
        moe(x_TD)
        self.assertEqual(moe.router.tokens_per_expert_E.sum().item(), 2 * 3 * top_k)
        training_counts = moe.router.tokens_per_expert_E.clone()

        moe.eval()
        with torch.no_grad():
            moe(x_TD)

        torch.testing.assert_close(
            moe.router.tokens_per_expert_E,
            training_counts,
        )

    def test_padding_is_excluded_from_counts_but_still_dispatched(self):
        moe = self._build_moe()
        x_TD = torch.randn(6, 4)
        padding_mask_T = torch.tensor([False, False, False, True, True, True])

        moe.train()
        moe(x_TD, padding_mask_T=padding_mask_T)

        self.assertEqual(moe.router.tokens_per_expert_E.sum().item(), 3)
        self.assertEqual(moe.routed_experts.num_tokens_per_expert_E.sum().item(), 6)

    def test_router_masks_padding_only_for_aux_loss(self):
        router = make_router_config(
            dim=4,
            num_experts=2,
            score_func=Sigmoid.Config(),
            gate_param_init={"weight": nn.init.zeros_},
            top_k=1,
        ).build()
        router.init_states()
        aux_loss = _CapturingAuxLoss()
        router.aux_loss = aux_loss
        router.train()

        padding_mask_T = torch.tensor([False, False, True, True])
        _, _, routing_map_TE = router(
            torch.randn(4, 4),
            padding_mask_T=padding_mask_T,
        )

        self.assertTrue(routing_map_TE[padding_mask_T].any())
        self.assertIsNotNone(aux_loss.routing_map_TE)
        self.assertFalse(aux_loss.routing_map_TE[padding_mask_T].any())
        torch.testing.assert_close(
            aux_loss.routing_map_TE[~padding_mask_T],
            routing_map_TE[~padding_mask_T],
        )
        torch.testing.assert_close(
            router.tokens_per_expert_E,
            aux_loss.routing_map_TE.sum(dim=0).to(torch.float32),
        )

    def test_router_validates_padding_mask(self):
        router = make_router_config(
            dim=4,
            num_experts=2,
            score_func=Sigmoid.Config(),
            gate_param_init={"weight": nn.init.zeros_},
            top_k=1,
        ).build()
        x_TD = torch.randn(4, 4)

        with self.assertRaisesRegex(ValueError, "dtype bool"):
            router(x_TD, padding_mask_T=torch.zeros(4))
        with self.assertRaisesRegex(ValueError, "routing-map token axis"):
            router(x_TD, padding_mask_T=torch.zeros(3, dtype=torch.bool))

    def test_padding_mask_sharding_follows_input_token_layout(self):
        for enable_ep, enable_sp in ((False, False), (True, False), (True, True)):
            with self.subTest(enable_ep=enable_ep, enable_sp=enable_sp):
                moe_config = _moe_sharding_config(
                    enable_ep=enable_ep,
                    enable_sp=enable_sp,
                )
                assert moe_config.in_src_shardings is not None
                assert moe_config.in_dst_shardings is not None
                self.assertEqual(
                    _per_axis_types(moe_config.in_src_shardings["padding_mask_T"]),
                    _per_axis_types(token_id_placement(enable_sp=enable_sp)),
                )
                self.assertEqual(
                    _per_axis_types(moe_config.in_dst_shardings["padding_mask_T"]),
                    _per_axis_types(
                        token_id_placement(enable_sp=enable_sp and enable_ep)
                    ),
                )
                self.assertEqual(
                    moe_config.in_src_shardings["x_TD"].partition_spec[0],
                    moe_config.in_src_shardings["padding_mask_T"].partition_spec[0],
                )
                self.assertEqual(
                    moe_config.in_dst_shardings["x_TD"].partition_spec[0],
                    moe_config.in_dst_shardings["padding_mask_T"].partition_spec[0],
                )

                router_config = _router_sharding_config(
                    enable_ep=enable_ep,
                    enable_sp=enable_sp,
                )
                assert router_config.in_src_shardings is not None
                assert router_config.in_dst_shardings is not None
                self.assertEqual(
                    router_config.in_src_shardings,
                    moe_config.in_dst_shardings,
                )
                self.assertEqual(
                    router_config.in_dst_shardings["x_TD"].partition_spec[0],
                    router_config.in_dst_shardings["padding_mask_T"].partition_spec[0],
                )
                self.assertEqual(
                    _per_axis_types(router_config.in_dst_shardings["padding_mask_T"]),
                    _per_axis_types(token_id_placement(enable_sp=enable_ep)),
                )

    def test_moe_block_sequence_shards_padding_mask(self):
        x_src = token_id_placement()
        x_dst = token_id_placement(enable_sp=True)
        block_cfg = SimpleNamespace(
            sharding_config=ShardingConfig(
                in_src_shardings={"x": x_src},
                in_dst_shardings={"x": x_dst},
            )
        )

        set_moe_block_padding_mask_sharding(block_cfg, enable_sp=True)

        config = block_cfg.sharding_config
        assert config.in_src_shardings is not None
        assert config.in_dst_shardings is not None
        self.assertIs(config.in_src_shardings["x"], x_src)
        self.assertIs(config.in_dst_shardings["x"], x_dst)
        self.assertEqual(
            _per_axis_types(config.in_src_shardings["padding_mask"]),
            _per_axis_types(token_id_placement()),
        )
        self.assertEqual(
            _per_axis_types(config.in_dst_shardings["padding_mask"]),
            _per_axis_types(token_id_placement(enable_sp=True)),
        )

    def test_moe_without_ep_keeps_routed_weights_replicated(self):
        def make_config():
            return SimpleNamespace(
                sharding_config=None,
                router=SimpleNamespace(
                    sharding_config=None,
                    aux_loss=MicrobatchWiseLoadBalanceLoss.Config(coeff=1e-3),
                    gate=SimpleNamespace(sharding_config=None),
                ),
                shared_experts=SimpleNamespace(
                    sharding_config=None,
                    w13=SimpleNamespace(sharding_config=None),
                    w2=SimpleNamespace(sharding_config=None),
                ),
                routed_experts=SimpleNamespace(
                    sharding_config=None,
                    inner_experts=SimpleNamespace(sharding_config=None),
                ),
            )

        def tp_type(layout):
            return _per_axis_types(layout)[MeshAxisName.TP]

        moe_config = make_config()
        set_moe_sharding_config(
            moe_config,
            enable_ep=False,
            enable_sp=False,
        )

        root = moe_config.sharding_config
        assert root is not None
        assert root.in_dst_shardings is not None
        self.assertEqual(tp_type(root.in_dst_shardings["x_TD"]), spmd.R)
        self.assertEqual(tp_type(root.out_src_shardings), spmd.R)

        shared = moe_config.shared_experts
        assert shared.sharding_config.in_dst_shardings is not None
        self.assertEqual(tp_type(shared.sharding_config.in_dst_shardings["x"]), spmd.R)
        self.assertEqual(
            tp_type(shared.w13.sharding_config.state_shardings["weight"]), spmd.S(1)
        )
        self.assertEqual(
            tp_type(shared.w13.sharding_config.out_src_shardings), spmd.S(2)
        )
        self.assertEqual(
            tp_type(shared.w2.sharding_config.state_shardings["weight"]), spmd.S(1)
        )
        self.assertEqual(tp_type(shared.w2.sharding_config.out_src_shardings), spmd.P)
        self.assertEqual(tp_type(shared.w2.sharding_config.out_dst_shardings), spmd.R)

        routed = moe_config.routed_experts
        assert routed.sharding_config.in_dst_shardings is not None
        for name in ("x_TD", "topk_scores_TK", "topk_expert_ids_TK"):
            self.assertEqual(
                tp_type(routed.sharding_config.in_dst_shardings[name]), spmd.R
            )
        self.assertEqual(
            tp_type(
                routed.sharding_config.in_dst_shardings["num_local_tokens_per_expert_E"]
            ),
            spmd.R,
        )
        self.assertEqual(tp_type(routed.sharding_config.out_src_shardings), spmd.R)
        self.assertEqual(tp_type(routed.sharding_config.out_dst_shardings), spmd.R)
        self.assertIsNone(routed.inner_experts.sharding_config)


if __name__ == "__main__":
    unittest.main()
