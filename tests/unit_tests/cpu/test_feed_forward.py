# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear


def _fill(value: float) -> Callable[[torch.Tensor], None]:
    def init(tensor: torch.Tensor) -> None:
        torch.nn.init.constant_(tensor, value)

    return init


def test_feed_forward_uses_one_physical_gate_up_linear():
    config = FeedForward.Config(
        w1=Linear.Config(
            in_features=4,
            out_features=8,
            bias=True,
            param_init={"weight": _fill(1.0), "bias": _fill(2.0)},
        ),
        w2=Linear.Config(
            in_features=8,
            out_features=4,
            bias=True,
            param_init={"weight": _fill(3.0), "bias": _fill(4.0)},
        ),
        w3=Linear.Config(
            in_features=4,
            out_features=8,
            bias=True,
            param_init={"weight": _fill(5.0), "bias": _fill(6.0)},
        ),
    )
    feed_forward = config.build()
    feed_forward.init_states()

    assert set(feed_forward._modules) == {"w13", "w2"}
    assert feed_forward.w13._logical_output_slices == (("w1", 8), ("w3", 8))
    assert set(feed_forward.state_dict()) == {
        "w1.weight",
        "w1.bias",
        "w2.weight",
        "w2.bias",
        "w3.weight",
        "w3.bias",
    }

    w13_H2D = feed_forward.w13.weight.unflatten(0, (8, 2))
    b13_H2 = feed_forward.w13.bias.unflatten(0, (8, 2))
    torch.testing.assert_close(w13_H2D[:, 0], torch.ones_like(w13_H2D[:, 0]))
    torch.testing.assert_close(w13_H2D[:, 1], 5 * torch.ones_like(w13_H2D[:, 1]))
    torch.testing.assert_close(b13_H2[:, 0], 2 * torch.ones_like(b13_H2[:, 0]))
    torch.testing.assert_close(b13_H2[:, 1], 6 * torch.ones_like(b13_H2[:, 1]))


def test_feed_forward_loads_logical_checkpoint_and_matches_reference():
    config = FeedForward.Config(
        w1=Linear.Config(in_features=4, out_features=8),
        w2=Linear.Config(in_features=8, out_features=4),
        w3=Linear.Config(in_features=4, out_features=8),
    )
    feed_forward = config.build()
    state_dict = {
        "w1.weight": torch.randn(8, 4),
        "w2.weight": torch.randn(4, 8),
        "w3.weight": torch.randn(8, 4),
    }
    feed_forward.load_state_dict(state_dict)

    x_TD = torch.randn(3, 4, requires_grad=True)
    reference_x_TD = x_TD.detach().clone().requires_grad_()
    w1_HD = state_dict["w1.weight"].detach().clone().requires_grad_()
    w2_DH = state_dict["w2.weight"].detach().clone().requires_grad_()
    w3_HD = state_dict["w3.weight"].detach().clone().requires_grad_()
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
    w13_grad_H2D = feed_forward.w13.weight.grad.unflatten(0, (8, 2))
    torch.testing.assert_close(w13_grad_H2D[:, 0], w1_HD.grad)
    torch.testing.assert_close(w13_grad_H2D[:, 1], w3_HD.grad)
    torch.testing.assert_close(feed_forward.w2.weight.grad, w2_DH.grad)


def test_feed_forward_rejects_mismatched_gate_up_implementations():
    class AlternateLinear(Linear):
        @dataclass(kw_only=True, slots=True)
        class Config(Linear.Config):
            pass

    config = FeedForward.Config(
        w1=Linear.Config(in_features=4, out_features=8),
        w2=Linear.Config(in_features=8, out_features=4),
        w3=AlternateLinear.Config(in_features=4, out_features=8),
    )

    with pytest.raises(ValueError, match="different implementations"):
        config.build()


def test_feed_forward_rejects_mismatched_gate_up_output_sizes():
    config = FeedForward.Config(
        w1=Linear.Config(in_features=4, out_features=8),
        w2=Linear.Config(in_features=8, out_features=4),
        w3=Linear.Config(in_features=4, out_features=16),
    )

    with pytest.raises(ValueError, match="matching out_features"):
        config.build()
