# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.models.common import Linear, RouterGateLinear
from torchtitan.models.common.config_utils import make_routed_experts_config
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.kimi_k3.moe import KimiLatentMoE
from torchtitan.models.kimi_k3.quantile_balance import (
    QuantileBalancedTopKRouter,
    QuantileBalancer,
    register_moe_quantile_balancing_hook,
)


class _PassthroughRoutedExperts(nn.Module):
    def forward(
        self,
        x_TD,
        topk_scores_TK,
        topk_expert_ids_TK,
        num_local_tokens_per_expert_E,
    ):
        return x_TD


class _QuantileModelFixture(nn.Module):
    def __init__(self):
        super().__init__()
        block = nn.Module()
        block.moe_enabled = True
        block.moe = _build_latent_moe(top_k=1, num_bins=10)
        self.layers = nn.ModuleDict({"0": block})


class _ParallelDimsFixture:
    ep_enabled = False
    tp = 1

    def get_optional_mesh(self, name):
        return None


def _build_latent_moe(
    *,
    dim: int = 4,
    num_experts: int = 4,
    top_k: int = 2,
    num_bins: int = 20,
) -> KimiLatentMoE:
    config = KimiLatentMoE.Config(
        num_experts=num_experts,
        router=QuantileBalancedTopKRouter.Config(
            num_experts=num_experts,
            top_k=top_k,
            gate=RouterGateLinear.Config(
                in_features=dim,
                out_features=num_experts,
                bias=False,
                param_init={"weight": nn.init.zeros_},
            ),
            score_func="sigmoid",
            route_norm=True,
            num_bins=num_bins,
        ),
        routed_experts=make_routed_experts_config(
            dim=dim,
            hidden_dim=8,
            num_experts=num_experts,
            top_k=top_k,
            param_init={},
            comm_backend="standard",
        ),
        routed_down=Linear.Config(
            in_features=dim,
            out_features=dim,
            bias=False,
            param_init={"weight": nn.init.zeros_},
        ),
        routed_norm=RMSNorm.Config(
            normalized_shape=dim,
            param_init={"weight": nn.init.ones_},
        ),
        routed_up=Linear.Config(
            in_features=dim,
            out_features=dim,
            bias=False,
            param_init={"weight": nn.init.zeros_},
        ),
        shared_experts=None,
        load_balance_coeff=None,
    )
    moe = config.build()
    moe.routed_experts = _PassthroughRoutedExperts()
    with torch.no_grad():
        moe.router.gate.weight.copy_(torch.eye(dim))
        moe.routed_down.weight.copy_(torch.eye(dim))
        moe.routed_norm.weight.fill_(1.0)
        moe.routed_up.weight.copy_(torch.eye(dim))
    return moe


class TestKimiK3QuantileBalancing(unittest.TestCase):
    def test_latent_moe_owns_bias_and_accumulates_histogram(self):
        moe = _build_latent_moe()

        self.assertIn("expert_bias_E", dict(moe.named_buffers()))
        self.assertIn("expert_bias_E", moe.state_dict())
        x_TD = torch.randn(6, 4)
        moe(x_TD)
        torch.testing.assert_close(
            moe.router.quantile_balancer.required_bias_histogram_EB.sum(dim=-1),
            torch.full((4,), 6, dtype=torch.int64),
        )

    def test_router_uses_top_k_plus_one_cutoff_during_training(self):
        router = QuantileBalancedTopKRouter.Config(
            num_experts=4,
            top_k=2,
            gate=RouterGateLinear.Config(
                in_features=4,
                out_features=4,
                bias=False,
                param_init={"weight": nn.init.zeros_},
            ),
            score_func="sigmoid",
            route_norm=True,
            num_bins=20,
        ).build()
        with torch.no_grad():
            router.gate.weight.copy_(torch.eye(4))

        x_TD = torch.tensor(
            [
                [2.0, 1.0, 0.0, -1.0],
                [-0.5, 0.5, 1.5, 2.5],
            ]
        )
        expert_bias_E = torch.tensor([0.0, 0.2, -0.1, 0.1])
        router.train()
        weights_TK, expert_ids_TK, scores_TE = router(x_TD, expert_bias_E)

        expected_scores_TE = torch.sigmoid(x_TD)
        expected_values_TJ, expected_ids_TJ = torch.topk(
            expected_scores_TE + expert_bias_E,
            k=3,
            dim=-1,
            sorted=True,
        )
        torch.testing.assert_close(scores_TE, expected_scores_TE)
        torch.testing.assert_close(expert_ids_TK, expected_ids_TJ[:, :2])
        expected_weights_TK = expected_scores_TE.gather(-1, expected_ids_TJ[:, :2])
        expected_weights_TK /= expected_weights_TK.sum(dim=-1, keepdim=True)
        torch.testing.assert_close(weights_TK, expected_weights_TK)

        lower_bound = expert_bias_E.min() - 1.0
        bin_width = (expert_bias_E.max() - expert_bias_E.min() + 2.0) / 20
        expected_bin_indices_TE = torch.floor(
            (expected_values_TJ[:, 2:] - expected_scores_TE - lower_bound) / bin_width
        ).to(torch.int64)
        expected_histogram_EB = torch.zeros(4, 20, dtype=torch.int32)
        expected_histogram_EB.scatter_add_(
            1,
            expected_bin_indices_TE.clamp_(0, 19).transpose(0, 1),
            torch.ones(4, 2, dtype=torch.int32),
        )
        torch.testing.assert_close(
            router.quantile_balancer.required_bias_histogram_EB,
            expected_histogram_EB,
        )

        training_histogram_EB = (
            router.quantile_balancer.required_bias_histogram_EB.clone()
        )
        router.eval()
        _, eval_expert_ids_TK, _ = router(x_TD, expert_bias_E)
        torch.testing.assert_close(
            eval_expert_ids_TK.sort(dim=-1).values,
            expected_ids_TJ[:, :2].sort(dim=-1).values,
        )
        torch.testing.assert_close(
            router.quantile_balancer.required_bias_histogram_EB,
            training_histogram_EB,
        )

    def test_histogram_observes_every_token_for_every_expert(self):
        balancer = QuantileBalancer.Config(
            num_experts=4,
            top_k=1,
            num_bins=20,
        ).build()
        scores_TE = torch.tensor(
            [
                [0.9, 0.7, 0.2, 0.1],
                [0.8, 0.6, 0.4, 0.3],
                [0.7, 0.5, 0.4, 0.2],
            ]
        )
        cutoff_T1 = torch.topk(scores_TE, k=2, dim=-1, sorted=True).values[:, 1:]
        balancer.observe(scores_TE, cutoff_T1, torch.zeros(4))

        torch.testing.assert_close(
            balancer.required_bias_histogram_EB.sum(dim=-1),
            torch.full((4,), 3, dtype=torch.int64),
        )

        counts_EB = balancer.required_bias_histogram_EB.clone()
        balancer.eval()
        balancer.observe(scores_TE, cutoff_T1, torch.zeros(4))
        torch.testing.assert_close(balancer.required_bias_histogram_EB, counts_EB)

    def test_histogram_quantile_interpolation_and_mean_centering(self):
        balancer = QuantileBalancer.Config(
            num_experts=4,
            top_k=1,
            num_bins=10,
        ).build()
        balancer.required_bias_histogram_EB[0, 2] = 2
        balancer.required_bias_histogram_EB[0, 3] = 6
        balancer.required_bias_histogram_EB[1, 4] = 2
        balancer.required_bias_histogram_EB[1, 5] = 6
        balancer.required_bias_histogram_EB[2, 6] = 2
        balancer.required_bias_histogram_EB[2, 7] = 6
        balancer.required_bias_histogram_EB[3, 8] = 2
        balancer.required_bias_histogram_EB[3, 9] = 6

        expert_bias_E = balancer.estimate_expert_bias(torch.zeros(4))

        torch.testing.assert_close(
            expert_bias_E,
            torch.tensor([-0.6, -0.2, 0.2, 0.6]),
        )
        self.assertAlmostEqual(expert_bias_E.mean().item(), 0.0)

    def test_optimizer_hook_updates_bias_and_resets_histogram(self):
        model = _QuantileModelFixture()
        model.layers["0"].moe.tokens_per_expert_E.fill_(1)
        histogram_EB = model.layers[
            "0"
        ].moe.router.quantile_balancer.required_bias_histogram_EB
        histogram_EB[0, 2] = 2
        histogram_EB[0, 3] = 6
        histogram_EB[1, 4] = 2
        histogram_EB[1, 5] = 6
        histogram_EB[2, 6] = 2
        histogram_EB[2, 7] = 6
        histogram_EB[3, 8] = 2
        histogram_EB[3, 9] = 6
        config = OptimizersContainer.Config(
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 0.0, "weight_decay": 0.0},
                ),
            ],
        )
        container = config.build(model_parts=[model])
        register_moe_quantile_balancing_hook(
            container,
            [model],
            _ParallelDimsFixture(),
        )

        container.step()

        torch.testing.assert_close(
            model.layers["0"].moe.expert_bias_E,
            torch.tensor([-0.6, -0.2, 0.2, 0.6]),
        )
        self.assertEqual(histogram_EB.count_nonzero().item(), 0)
        self.assertEqual(
            model.layers["0"].moe.tokens_per_expert_E.count_nonzero().item(),
            0,
        )


if __name__ == "__main__":
    unittest.main()
