# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import spmd_types as spmd
import torch

from torchtitan.experiments.rl.models.vllm_wrapper import VLLMModelWrapper
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.overrides.fused_swiglu import fused_grouped_experts
from torchtitan.protocols.sharding import ShardingConfig


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
