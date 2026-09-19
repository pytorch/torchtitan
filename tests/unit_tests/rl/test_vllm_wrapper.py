# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import spmd_types as spmd
import torch

from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.overrides.fused_swiglu import fused_grouped_experts
from torchtitan.protocols.sharding import ShardingConfig

from torchtitan.rl.model.vllm_wrapper import VLLMModelWrapper


def test_state_dict_layouts_include_native_feed_forward_weight():
    """Verify the fused dense FFN layout uses its native w13 state-dict key."""
    colwise = dense_param_placement(tp=spmd.S(1))
    rowwise = dense_param_placement(tp=spmd.S(1))
    config = FeedForward.Config(
        w13=Linear.Config(
            in_features=16,
            out_features=32,
            num_linears=2,
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

    assert layouts["feed_forward.w13.weight"] is colwise
    assert "feed_forward.w1.weight" not in layouts
    assert "feed_forward.w3.weight" not in layouts
    assert layouts["feed_forward.w2.weight"] is rowwise


def test_state_dict_layouts_include_split_expert_weights():
    """Verify fused grouped-expert layouts use the exported split state-dict keys."""
    colwise = dense_param_placement(tp=spmd.S(1))
    rowwise = dense_param_placement(tp=spmd.S(2))
    config = GroupedExperts.Config(
        dim=16,
        hidden_dim=32,
        num_experts=4,
        sharding_config=ShardingConfig(
            state_shardings={
                "w1_EFD": colwise,
                "w2_EDF": rowwise,
                "w3_EFD": colwise,
            }
        ),
    )
    model = torch.nn.Module()
    model.experts = fused_grouped_experts(config).build()
    wrapper = VLLMModelWrapper.__new__(VLLMModelWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = model

    layouts = wrapper.get_state_dict_layouts()

    assert layouts["experts.w1_EFD"] is colwise
    assert layouts["experts.w3_EFD"] is colwise
    assert layouts["experts.w2_EDF"] is rowwise
