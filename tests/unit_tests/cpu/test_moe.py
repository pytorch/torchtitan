# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import call, patch

import spmd_types as spmd
import torch
import torch.nn as nn
import torch.nn.functional as F
from spmd_types import SpmdType

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import _per_axis_types
from torchtitan.models.common.activation import Sigmoid, SiTUGLU, Softmax, SqrtSoftplus
from torchtitan.models.common.config_utils import (
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
    make_shared_expert_ffn_config,
)
from torchtitan.models.common.decoder_sharding import (
    dense_param_placement,
    token_id_placement,
)
from torchtitan.models.common.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.linear import (
    ColumnParallelLinear,
    SharedExpertRowParallelLinear,
)
from torchtitan.models.common.moe import (
    MicrobatchWiseLoadBalanceLoss,
    MoE,
    RoundRobinTokenChoiceTopKRouter,
    TokenChoiceTopKRouter,
)
from torchtitan.models.common.moe_sharding import (
    _moe_sharding_config,
    _routed_experts_sharding_configs,
    _router_sharding_config,
    _shared_experts_sharding_configs,
    set_moe_sharding_config,
)
from torchtitan.models.common.nn_modules import RMSNorm


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

    def forward(
        self,
        scores_TE,
        routing_map_TE,
        *,
        carrier,
        padding_mask_T=None,
        denominator,
    ):
        del scores_TE, padding_mask_T, denominator
        self.routing_map_TE = routing_map_TE
        return carrier


class _IdentityDispatcher(nn.Module):
    def dispatch(
        self,
        x_TD,
        topk_scores_TK,
        topk_expert_ids_TK,
        num_local_tokens_per_expert_E,
    ):
        del topk_scores_TK, topk_expert_ids_TK
        return x_TD, num_local_tokens_per_expert_E, None

    def combine(self, routed_output_RD, metadata, x_TD):
        del metadata, x_TD
        return routed_output_RD


class _AddOneW13(nn.Module):
    def forward(self, x_RD, offsets_E):
        del offsets_E
        return torch.stack((x_RD + 1, x_RD), dim=-2)


class _SelectGate(nn.Module):
    def forward(self, gate_up_R2D, *, offsets):
        del offsets
        return gate_up_R2D[:, 0]


class _IdentityW2(nn.Module):
    def forward(self, hidden_RD, offsets_E):
        del offsets_E
        return hidden_RD


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
                gate=HiMidLoLinear.Config(in_features=4, out_features=4),
            )

    def test_round_robin_router_balances_assignments(self):
        router = RoundRobinTokenChoiceTopKRouter.Config(
            num_experts=4,
            gate=HiMidLoLinear.Config(in_features=4, out_features=4),
            score_func=Sigmoid.Config(),
            top_k=2,
        ).build()

        _, topk_expert_ids_TK, _ = router(torch.randn(4, 4))

        torch.testing.assert_close(
            topk_expert_ids_TK,
            torch.tensor([[0, 1], [2, 3], [0, 1], [2, 3]]),
        )

    def test_routed_experts_use_configured_activation(self):
        """Routed experts build and execute their configured binary activation."""
        activation_fn = SiTUGLU.Config(beta=4.0, linear_beta=25.0)
        config = make_routed_experts_config(
            dim=4,
            hidden_dim=8,
            num_experts=2,
            top_k=1,
            param_init={},
        )
        config.activation_fn = activation_fn
        experts = config.build()
        gate_up_R2F = torch.randn(3, 2, 8)

        expected_RF = activation_fn.build()(gate_up_R2F)
        actual_RF = experts.activation_fn(gate_up_R2F)
        torch.testing.assert_close(actual_RF, expected_RF)

    def test_routed_experts_own_postprocess_before_combine(self):
        config = replace(
            make_routed_experts_config(
                dim=4,
                hidden_dim=4,
                num_experts=2,
                top_k=1,
                param_init={},
            ),
            output_postprocess=RMSNorm.Config(normalized_shape=4),
        )
        routed_experts = config.build()
        routed_experts.w13 = _AddOneW13()
        routed_experts.activation_fn = _SelectGate()
        routed_experts.w2 = _IdentityW2()
        routed_experts.token_dispatcher = _IdentityDispatcher()
        assert isinstance(routed_experts.output_postprocess, RMSNorm)
        with torch.no_grad():
            routed_experts.output_postprocess.weight.fill_(3.0)
        x_TD = torch.arange(8, dtype=torch.float32).reshape(2, 4).requires_grad_()

        output_TD = routed_experts(
            x_TD,
            torch.ones(2, 1),
            torch.zeros(2, 1, dtype=torch.int64),
            torch.tensor([2, 0]),
        )
        expected_TD = F.rms_norm(
            x_TD + 1,
            (4,),
            routed_experts.output_postprocess.weight,
            routed_experts.output_postprocess.eps,
        )
        torch.testing.assert_close(output_TD, expected_TD)
        self.assertIn("output_postprocess.weight", routed_experts.state_dict())

        output_TD.sum().backward()
        self.assertIsNotNone(x_TD.grad)
        self.assertIsNotNone(routed_experts.output_postprocess.weight.grad)

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
            aux_loss_denominator=torch.tensor(2),
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

    def test_padding_mask_enters_moe_replicated_and_router_sharded(self):
        for enable_sp in (False, True):
            with self.subTest(enable_sp=enable_sp):
                moe_config = _moe_sharding_config(enable_sp=enable_sp)
                assert moe_config.in_src_shardings is not None
                self.assertIsNone(moe_config.in_dst_shardings)
                self.assertEqual(
                    _per_axis_types(moe_config.in_src_shardings["padding_mask_T"]),
                    _per_axis_types(token_id_placement()),
                )

                router_config = _router_sharding_config()
                router_inputs = router_config.in_src_shardings
                assert router_inputs is not None
                self.assertIsNone(router_config.in_dst_shardings)
                self.assertEqual(
                    _per_axis_types(router_inputs["x_TD"])[MeshAxisName.TP],
                    spmd.S(0),
                )
                self.assertEqual(
                    _per_axis_types(router_inputs["padding_mask_T"]),
                    _per_axis_types(token_id_placement(enable_sp=True)),
                )

    def test_shared_expert_uses_runtime_sp_aware_output_projection(self):
        config = make_shared_expert_ffn_config(
            dim=4,
            hidden_dim=8,
            w13_param_init={},
            w2_param_init={},
        )

        self.assertIs(type(config.w13), ColumnParallelLinear.Config)
        self.assertIs(type(config.w2), SharedExpertRowParallelLinear.Config)
        self.assertEqual(config.w13.num_linears, 2)

    def test_common_shared_expert_input_gather_is_owned_by_w13(self):
        moe_config = _moe_sharding_config(enable_sp=True)
        shared_config, w13_config, _ = _shared_experts_sharding_configs(enable_sp=True)

        assert moe_config.in_src_shardings is not None
        assert shared_config.in_src_shardings is not None
        assert w13_config.in_src_shardings is not None
        self.assertIsNone(moe_config.in_dst_shardings)
        self.assertIsNone(shared_config.in_dst_shardings)
        self.assertIsNone(w13_config.in_dst_shardings)
        self.assertEqual(
            _per_axis_types(moe_config.in_src_shardings["x_TD"])[MeshAxisName.TP],
            spmd.S(0),
        )
        self.assertEqual(
            _per_axis_types(shared_config.in_src_shardings["x"])[MeshAxisName.TP],
            spmd.S(0),
        )
        self.assertEqual(
            _per_axis_types(w13_config.in_src_shardings["input"])[MeshAxisName.TP],
            spmd.S(0),
        )

    def test_explicit_moe_tp_transitions_with_ep_without_sp(self):
        moe = MoE.__new__(MoE)
        x_TD = torch.randn(4, 8)
        padding_mask_T = torch.zeros(4, dtype=torch.bool)
        tp_group = object()

        with (
            patch(
                "torchtitan.models.common.moe.spmd_sparse_mesh",
                return_value=object(),
            ),
            patch(
                "torchtitan.models.common.moe.spmd_dense_sp_enabled",
                return_value=False,
            ),
            patch(
                "torchtitan.models.common.moe.spmd_mesh_group",
                return_value=tp_group,
            ),
            patch(
                "torchtitan.models.common.moe.spmd.redistribute",
                side_effect=lambda tensor, *_args, **_kwargs: tensor,
            ) as redistribute,
            patch(
                "torchtitan.models.common.moe.remat.region",
                side_effect=lambda function, *_args, **_kwargs: function,
            ) as region,
            patch(
                "torchtitan.models.common.moe.remat.recompute_needs_tensor"
            ) as recompute_needs_tensor,
        ):
            moe._maybe_shard_routed_branch_inputs_across_tp(x_TD, padding_mask_T)
            moe._maybe_zero_fill_routed_output_to_tp_partial(x_TD)
            moe._maybe_all_reduce_moe_output_across_tp(x_TD)
            self.assertEqual(
                redistribute.call_args_list,
                [
                    call(
                        x_TD,
                        tp_group,
                        src=spmd.I,
                        dst=spmd.S(0),
                        backward_options={"op_dtype": x_TD.dtype},
                    ),
                    call(
                        padding_mask_T,
                        tp_group,
                        src=spmd.R,
                        dst=spmd.S(0),
                        backward_options={"op_dtype": padding_mask_T.dtype},
                    ),
                    call(
                        x_TD,
                        tp_group,
                        src=spmd.S(0),
                        dst=spmd.P,
                        backward_options={"op_dtype": x_TD.dtype},
                    ),
                    call(
                        x_TD,
                        tp_group,
                        src=spmd.P,
                        dst=spmd.I,
                        backward_options={"op_dtype": x_TD.dtype},
                    ),
                ],
            )
            region.assert_called_once_with(
                spmd.redistribute,
                "tp_output_reduction",
                recompute=True,
            )
            # The zero-fill pins the routed output it reads; the block's residual
            # add pins the reduced output.
            recompute_needs_tensor.assert_called_once_with(x_TD)

    def test_routed_branch_rejects_tp_without_ep(self):
        moe = MoE.__new__(MoE)
        x_TD = torch.randn(4, 8)

        with (
            patch(
                "torchtitan.models.common.moe.spmd_sparse_mesh",
                return_value=None,
            ),
            patch(
                "torchtitan.models.common.moe.spmd_mesh_group",
                return_value=object(),
            ),
            self.assertRaisesRegex(
                AssertionError,
                "requires expert parallelism",
            ),
        ):
            moe._maybe_shard_routed_branch_inputs_across_tp(x_TD, None)

    def test_expert_branch_layouts_before_moe_boundary(self):
        for enable_sp, expected in (
            (False, spmd.P),
            (True, spmd.S(0)),
        ):
            with self.subTest(enable_sp=enable_sp):
                shared, _w13, w2 = _shared_experts_sharding_configs(enable_sp=enable_sp)
                routed, _w13, _w2 = _routed_experts_sharding_configs(
                    enable_ep=True,
                )

                shared_output = shared.out_src_shardings
                w2_output = w2.out_src_shardings
                routed_output = routed.out_src_shardings
                assert isinstance(shared_output, SpmdType)
                assert isinstance(w2_output, SpmdType)
                assert isinstance(routed_output, SpmdType)
                self.assertEqual(
                    _per_axis_types(shared_output).get(MeshAxisName.TP),
                    expected,
                )
                self.assertEqual(
                    _per_axis_types(w2_output).get(MeshAxisName.TP), expected
                )
                self.assertIsNone(w2.out_dst_shardings)
                self.assertEqual(
                    _per_axis_types(routed_output).get(MeshAxisName.TP),
                    spmd.S(0),
                )
                self.assertIsNone(routed.out_dst_shardings)

    def test_explicit_padding_mask_sharding_with_sp(self):
        moe = MoE.__new__(MoE)
        x_TD = torch.randn(4, 8)
        padding_mask_T = torch.zeros(4, dtype=torch.bool)
        tp_group = object()

        with (
            patch(
                "torchtitan.models.common.moe.spmd_sparse_mesh",
                return_value=object(),
            ),
            patch(
                "torchtitan.models.common.moe.spmd_dense_sp_enabled",
                return_value=True,
            ),
            patch(
                "torchtitan.models.common.moe.spmd_mesh_group",
                return_value=tp_group,
            ),
            patch(
                "torchtitan.models.common.moe.spmd.redistribute",
                side_effect=lambda tensor, *_args, **_kwargs: tensor,
            ) as redistribute,
        ):
            (
                actual_x_TD,
                actual_padding_mask_T,
            ) = moe._maybe_shard_routed_branch_inputs_across_tp(x_TD, padding_mask_T)

        self.assertIs(actual_x_TD, x_TD)
        self.assertIs(actual_padding_mask_T, padding_mask_T)
        redistribute.assert_called_once_with(
            padding_mask_T,
            tp_group,
            src=spmd.R,
            dst=spmd.S(0),
            backward_options={"op_dtype": padding_mask_T.dtype},
        )

    def test_moe_without_ep_leaves_routed_weights_unsharded(self):
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
                    w13=SimpleNamespace(sharding_config=None),
                    w2=SimpleNamespace(sharding_config=None),
                ),
            )

        moe_config = make_config()
        set_moe_sharding_config(
            moe_config,
            enable_ep=False,
            enable_sp=False,
        )

        routed = moe_config.routed_experts
        for grouped_linear in (routed.w13, routed.w2):
            self.assertEqual(
                grouped_linear.sharding_config.state_shardings["weight"],
                dense_param_placement(tp=spmd.R),
            )


if __name__ == "__main__":
    unittest.main()
