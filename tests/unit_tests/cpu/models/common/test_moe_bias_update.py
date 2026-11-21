# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the auxiliary-loss-free expert bias update rules."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from torchtitan.models.common.activation import Sigmoid
from torchtitan.models.common.config_utils import (
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
)
from torchtitan.models.common.moe import register_moe_load_balancing_hook
from torchtitan.models.common.moe.moe import compute_expert_bias_delta


def _counts(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def _centered(delta: torch.Tensor) -> torch.Tensor:
    return delta - delta.mean()


def test_sign_rule_matches_the_previous_update() -> None:
    counts = _counts(10, 2, 6, 6)
    expected = _centered(1e-3 * torch.sign(counts.mean() - counts))
    delta = compute_expert_bias_delta(counts, rule="sign", coeff=1e-3)
    torch.testing.assert_close(delta, expected, rtol=0, atol=0)


def test_linear_rule_scales_with_the_relative_load_error() -> None:
    counts = _counts(10, 2, 6, 6)  # mean 6: errors -2/3, +2/3, 0, 0
    delta = compute_expert_bias_delta(counts, rule="linear", coeff=6e-3)
    torch.testing.assert_close(delta, torch.tensor([-4e-3, 4e-3, 0.0, 0.0]))


def test_linear_step_grows_with_imbalance_unlike_sign() -> None:
    mild, severe = _counts(7, 5, 6, 6), _counts(18, 2, 2, 2)
    for rule in ("sign", "linear"):
        mild_step = compute_expert_bias_delta(mild, rule=rule, coeff=1e-3).abs().max()
        severe_step = (
            compute_expert_bias_delta(severe, rule=rule, coeff=1e-3).abs().max()
        )
        if rule == "sign":
            assert severe_step <= mild_step * 1.5  # bounded by coeff either way
        else:
            assert severe_step > 10 * mild_step


@pytest.mark.parametrize("rule", ["sign", "linear"])
def test_updates_are_zero_mean_and_vanish_when_balanced(rule: str) -> None:
    delta = compute_expert_bias_delta(_counts(9, 1, 4, 2), rule=rule, coeff=1e-3)
    assert abs(delta.sum().item()) < 1e-9
    for balanced in (_counts(5, 5, 5, 5), torch.zeros(4)):
        delta = compute_expert_bias_delta(balanced, rule=rule, coeff=1e-3)
        assert torch.count_nonzero(delta) == 0


def _build_moe(**overrides) -> nn.Module:
    num_experts, dim, top_k = 4, 8, 1
    config = make_moe_config(
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
            enable_ep=False,
        ),
        load_balance_coeff=overrides.pop("load_balance_coeff", 1e-3),
        enable_sp=True,
    )
    for name, value in overrides.items():
        setattr(config, name, value)
    config.__post_init__()
    return config.build()


class _StepHooks:
    def __init__(self) -> None:
        self.hooks = []

    def register_step_pre_hook(self, hook) -> None:
        self.hooks.append(hook)

    def step(self) -> None:
        for hook in self.hooks:
            hook()


def _model_part(moe: nn.Module) -> nn.Module:
    block = nn.Module()
    block.moe = moe
    model_part = nn.Module()
    model_part.layers = nn.ModuleList([block])
    return model_part


@pytest.mark.parametrize("rule", ["sign", "linear"])
def test_hook_applies_the_configured_rule(rule: str) -> None:
    moe = _build_moe(bias_update_rule=rule, load_balance_coeff=2e-3)
    optimizers = _StepHooks()
    parallelism = SimpleNamespace(
        ep_enabled=False, tp=1, get_optional_mesh=lambda name: None
    )
    register_moe_load_balancing_hook(optimizers, [_model_part(moe)], parallelism)

    counts = _counts(5, 1, 1, 1)
    moe.router.tokens_per_expert_E.copy_(counts)
    optimizers.step()
    expected = compute_expert_bias_delta(counts, rule=rule, coeff=2e-3)
    torch.testing.assert_close(moe.expert_bias_E, expected)
    assert torch.count_nonzero(moe.router.tokens_per_expert_E) == 0


def test_default_rule_is_sign_and_adds_no_state() -> None:
    moe = _build_moe()
    assert moe.bias_update_rule == "sign"
    linear = _build_moe(bias_update_rule="linear")
    assert set(linear.state_dict()) == set(moe.state_dict())


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"bias_update_rule": "momentum"}, "bias_update_rule"),
        (
            {"bias_update_rule": "linear", "load_balance_coeff": None},
            "load_balance_coeff",
        ),
    ],
)
def test_config_rejects_invalid_settings(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _build_moe(**overrides)
