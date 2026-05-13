# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchtitan.models.common.config_utils import make_experts_config
from torchtitan.models.common.moe import FlexGroupedExperts, _pack_flex_ep_w13
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_debugmodel_flex_ep,
)


def test_make_experts_config_flex_ep():
    experts_config = make_experts_config(
        dim=16,
        hidden_dim=32,
        num_experts=8,
        top_k=2,
        param_init={
            "w1": torch.nn.init.zeros_,
            "w2": torch.nn.init.zeros_,
            "w3": torch.nn.init.zeros_,
        },
        score_before_experts=False,
        comm_backend="flex_ep",
    )

    assert isinstance(experts_config, FlexGroupedExperts.Config)
    assert experts_config.num_experts == 8
    assert experts_config.top_k == 2
    assert not experts_config.score_before_experts


def test_make_experts_config_flex_ep_rejects_score_before_experts():
    with pytest.raises(ValueError, match="score_before_experts=False"):
        make_experts_config(
            dim=16,
            hidden_dim=32,
            num_experts=8,
            top_k=2,
            param_init={
                "w1": torch.nn.init.zeros_,
                "w2": torch.nn.init.zeros_,
                "w3": torch.nn.init.zeros_,
            },
            score_before_experts=True,
            comm_backend="flex_ep",
        )


def test_flex_grouped_experts_builds_without_token_dispatcher():
    config = FlexGroupedExperts.Config(
        dim=16,
        hidden_dim=32,
        num_experts=8,
        top_k=2,
        param_init={
            "w1": torch.nn.init.zeros_,
            "w2": torch.nn.init.zeros_,
            "w3": torch.nn.init.zeros_,
        },
    )
    with torch.device("meta"):
        experts = config.build()

    assert isinstance(experts, FlexGroupedExperts)
    assert experts.num_experts == 8
    assert not hasattr(experts, "token_dispatcher")
    assert experts.ep_mesh is None


def test_deepseek_v3_flex_ep_config_uses_flex_grouped_experts():
    config = deepseek_v3_debugmodel_flex_ep()
    moe_config = next(
        layer.moe for layer in config.model_spec.model.layers if layer.moe is not None
    )

    assert isinstance(moe_config.experts, FlexGroupedExperts.Config)


def test_flex_w13_pack_helper():
    w1 = torch.randn(4, 8, 16)
    w3 = torch.randn(4, 8, 16)

    w13 = _pack_flex_ep_w13(w1, w3)

    assert w13.shape == (4, 16, 16)
    torch.testing.assert_close(w13[:, :8], w1)
    torch.testing.assert_close(w13[:, 8:], w3)
