# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F

import torchtitan.models.deepseek_v3.flavors as deepseek_v3_flavors
import torchtitan.models.deepseek_v4.flavors as deepseek_v4_flavors
import torchtitan.models.kimi_k2_7.flavors as kimi_k2_7_flavors
import torchtitan.models.llama3.flavors as llama3_flavors
import torchtitan.models.qwen3_5.flavors as qwen3_5_flavors
from torchtitan.models.common.activation import SiTUGLU
from torchtitan.models.common.config_utils import (
    fused_gate_up_param_init,
    make_ffn_config,
    make_shared_expert_ffn_config,
)
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.param_init import depth_scaled_std


def _fill(value: float) -> Callable[[torch.Tensor], None]:
    def init(tensor: torch.Tensor) -> None:
        torch.nn.init.constant_(tensor, value)

    return init


def test_feed_forward_uses_one_physical_gate_up_linear():
    config = FeedForward.Config(
        w13=Linear.Config(
            in_features=4,
            out_features=8,
            num_linears=2,
            param_init=fused_gate_up_param_init(
                {"weight": _fill(1.0)},
                {"weight": _fill(5.0)},
            ),
        ),
        w2=Linear.Config(
            in_features=8,
            out_features=4,
            bias=True,
            param_init={"weight": _fill(3.0), "bias": _fill(4.0)},
        ),
    )
    feed_forward = config.build()
    feed_forward.init_states()

    assert set(feed_forward._modules) == {"w13", "w2"}
    assert set(feed_forward.state_dict()) == {
        "w13.weight",
        "w2.weight",
        "w2.bias",
    }
    assert feed_forward.state_dict()["w13.weight"].data_ptr() == (
        feed_forward.w13.weight.data_ptr()
    )

    w13_2HD = feed_forward.w13.weight
    torch.testing.assert_close(w13_2HD[0], torch.ones_like(w13_2HD[0]))
    torch.testing.assert_close(w13_2HD[1], 5 * torch.ones_like(w13_2HD[1]))


def test_make_ffn_config_uses_input_init_for_gate_and_up():
    config = make_ffn_config(
        dim=4,
        hidden_dim=8,
        w13_param_init={"weight": _fill(1.0)},
        w2_param_init={"weight": _fill(2.0)},
    )
    feed_forward = config.build()
    feed_forward.init_states()

    w13_2HD = feed_forward.w13.weight
    torch.testing.assert_close(w13_2HD[0], torch.ones_like(w13_2HD[0]))
    torch.testing.assert_close(w13_2HD[1], torch.ones_like(w13_2HD[1]))
    torch.testing.assert_close(
        feed_forward.w2.weight, 2 * torch.ones_like(feed_forward.w2.weight)
    )


def test_make_shared_expert_ffn_config_uses_input_init_for_gate_and_up():
    config = make_shared_expert_ffn_config(
        dim=4,
        hidden_dim=8,
        w13_param_init={"weight": _fill(1.0)},
        w2_param_init={"weight": _fill(2.0)},
    )
    feed_forward = config.build()
    feed_forward.init_states()

    w13_2HD = feed_forward.w13.weight
    torch.testing.assert_close(w13_2HD[0], torch.ones_like(w13_2HD[0]))
    torch.testing.assert_close(w13_2HD[1], torch.ones_like(w13_2HD[1]))
    torch.testing.assert_close(
        feed_forward.w2.weight, 2 * torch.ones_like(feed_forward.w2.weight)
    )


def test_llama3_depth_scales_only_ffn_output(monkeypatch):
    linear_init = {"weight": _fill(1.0), "bias": _fill(0.0)}
    depth_init = {"weight": _fill(2.0), "bias": _fill(0.0)}
    monkeypatch.setattr(llama3_flavors, "_LINEAR_INIT", linear_init)
    monkeypatch.setattr(llama3_flavors, "_depth_init", lambda _layer_id: depth_init)

    build_config, max_context_length = llama3_flavors.MODEL_FLAVORS["debugmodel"]
    model_config = build_config(attn_backend="flex", seq_len=max_context_length)
    feed_forward = model_config.layers[0].feed_forward.build()
    feed_forward.init_states()

    w13_2HD = feed_forward.w13.weight
    torch.testing.assert_close(w13_2HD[0], torch.ones_like(w13_2HD[0]))
    torch.testing.assert_close(w13_2HD[1], torch.ones_like(w13_2HD[1]))
    torch.testing.assert_close(
        feed_forward.w2.weight, 2 * torch.ones_like(feed_forward.w2.weight)
    )


@pytest.mark.parametrize(
    "flavors",
    [deepseek_v3_flavors, deepseek_v4_flavors, kimi_k2_7_flavors, qwen3_5_flavors],
    ids=["deepseek_v3", "deepseek_v4", "kimi_k2_7", "qwen3_5"],
)
@pytest.mark.parametrize("layer_id", [0, 7])
def test_routed_experts_depth_scale_only_output(flavors, layer_id):
    param_init = flavors._depth_experts_init(layer_id)
    with torch.random.fork_rng(devices=[]):
        for name, std in (
            ("w1_EFD", 0.02),
            ("w3_EFD", 0.02),
            ("w2_EDF", depth_scaled_std(0.02, layer_id)),
        ):
            actual = torch.empty(2, 8, 4)
            expected = torch.empty_like(actual)
            torch.manual_seed(0)
            param_init[name](actual)
            torch.manual_seed(0)
            torch.nn.init.trunc_normal_(expected, std=std)
            torch.testing.assert_close(actual, expected)


def test_feed_forward_requires_two_w13_projections():
    config = FeedForward.Config(
        w13=Linear.Config(in_features=4, out_features=8),
        w2=Linear.Config(in_features=8, out_features=4),
    )

    with pytest.raises(ValueError, match="w13 requires num_linears=2"):
        config.build()


def test_feed_forward_loads_native_checkpoint_and_matches_reference():
    config = FeedForward.Config(
        w13=Linear.Config(in_features=4, out_features=8, num_linears=2),
        w2=Linear.Config(in_features=8, out_features=4),
    )
    feed_forward = config.build()
    w1_HD = torch.randn(8, 4)
    w3_HD = torch.randn(8, 4)
    state_dict = {
        "w13.weight": torch.stack((w1_HD, w3_HD)),
        "w2.weight": torch.randn(4, 8),
    }
    feed_forward.load_state_dict(state_dict)

    x_TD = torch.randn(3, 4, requires_grad=True)
    reference_x_TD = x_TD.detach().clone().requires_grad_()
    w1_HD = w1_HD.detach().clone().requires_grad_()
    w2_DH = state_dict["w2.weight"].detach().clone().requires_grad_()
    w3_HD = w3_HD.detach().clone().requires_grad_()
    expected_TD = F.linear(
        F.silu(F.linear(reference_x_TD, w1_HD)) * F.linear(reference_x_TD, w3_HD),
        w2_DH,
    )
    actual_TD = feed_forward(x_TD)
    torch.testing.assert_close(actual_TD, expected_TD)

    grad_TD = torch.randn_like(actual_TD)
    actual_TD.backward(grad_TD)
    expected_TD.backward(grad_TD)
    torch.testing.assert_close(x_TD.grad, reference_x_TD.grad)
    w13_grad_2HD = feed_forward.w13.weight.grad
    torch.testing.assert_close(w13_grad_2HD[0], w1_HD.grad)
    torch.testing.assert_close(w13_grad_2HD[1], w3_HD.grad)
    torch.testing.assert_close(feed_forward.w2.weight.grad, w2_DH.grad)


def test_feed_forward_uses_configured_activation():
    activation_fn = SiTUGLU.Config(beta=4.0, linear_beta=25.0)
    config = FeedForward.Config(
        w13=Linear.Config(in_features=4, out_features=8, num_linears=2),
        w2=Linear.Config(in_features=8, out_features=4),
        activation_fn=activation_fn,
    )
    feed_forward = config.build()
    feed_forward.load_state_dict(
        {
            "w13.weight": torch.randn(2, 8, 4),
            "w2.weight": torch.randn(4, 8),
        }
    )

    x_TD = torch.randn(3, 4)
    gate_up_T2F = feed_forward.w13(x_TD)
    expected_TD = feed_forward.w2(activation_fn.build()(gate_up_T2F))
    torch.testing.assert_close(feed_forward(x_TD), expected_TD)
