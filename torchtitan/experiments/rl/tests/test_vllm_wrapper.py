# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import spmd_types as spmd
import torch

from torchtitan.experiments.rl.models.vllm_wrapper import VLLMModelWrapper
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import GroupedLinear, Linear
from torchtitan.protocols.sharding import ShardingConfig


def test_state_dict_layouts_include_split_feed_forward_weights():
    """Verify fused dense FFN layouts use the exposed w1/w3 state-dict keys."""
    colwise = dense_param_placement(tp=spmd.S(0))
    rowwise = dense_param_placement(tp=spmd.S(1))
    config = FeedForward.Config(
        w13=Linear.Config(
            in_features=16,
            out_features=64,
            sharding_config=ShardingConfig(state_shardings={"weight": colwise}),
        ),
        w2=Linear.Config(
            in_features=32,
            out_features=16,
            sharding_config=ShardingConfig(state_shardings={"weight": rowwise}),
        ),
    )
    model = torch.nn.Module()
    model.feed_forward = config.build()
    wrapper = VLLMModelWrapper.__new__(VLLMModelWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = model

    layouts = wrapper.get_state_dict_layouts()

    assert layouts["feed_forward.w1.weight"] is colwise
    assert layouts["feed_forward.w3.weight"] is colwise
    assert layouts["feed_forward.w2.weight"] is rowwise


def test_state_dict_layouts_include_grouped_linear_weights():
    """Verify first-class grouped linears expose their native state layouts."""
    w13_sharding = dense_param_placement(tp=spmd.S(2))
    rowwise = dense_param_placement(tp=spmd.S(2))
    w13_config = GroupedLinear.Config(
        group_size=4,
        in_features=16,
        out_features=(2, 32),
        sharding_config=ShardingConfig(state_shardings={"weight": w13_sharding}),
    )
    w2_config = GroupedLinear.Config(
        group_size=4,
        in_features=32,
        out_features=16,
        sharding_config=ShardingConfig(state_shardings={"weight": rowwise}),
    )
    model = torch.nn.Module()
    model.experts = torch.nn.Module()
    model.experts.w13 = w13_config.build()
    model.experts.w2 = w2_config.build()
    wrapper = VLLMModelWrapper.__new__(VLLMModelWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = model

    layouts = wrapper.get_state_dict_layouts()

    assert layouts["experts.w13.weight"] is w13_sharding
    assert layouts["experts.w2.weight"] is rowwise
