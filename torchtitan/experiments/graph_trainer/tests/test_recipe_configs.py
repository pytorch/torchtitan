# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.experiments.graph_trainer.tests import integration_tests
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.moe.sharding import expert_param_placement_sparse


@pytest.mark.parametrize(
    "recipe",
    [
        integration_tests.deepseek_v3_fused_mla_swiglu_fsdp_tp_ep,
        integration_tests.deepseek_v3_hsdp_edp_shard_ep,
        integration_tests.deepseek_v3_fsdp_tp_cp_ep,
        integration_tests.deepseek_v3_fsdp_tp_ep,
        integration_tests.deepseek_v3_fsdp_tp_ep_regional_inductor,
        integration_tests.deepseek_v3_regional_ep_overlap_transformer_batch,
        integration_tests.deepseek_v3_regional_ep_overlap_moe_batch,
        integration_tests.deepseek_v3_regional_ep_overlap_moe_seq,
        integration_tests.deepseek_v3_full_ep_overlap_transformer_batch,
        integration_tests.deepseek_v3_full_ep_overlap_moe_batch,
        integration_tests.deepseek_v3_full_ep_overlap_moe_seq,
        integration_tests.deepseek_v3_graph_pp_dual_pipe_v,
        integration_tests.deepseek_v3_graph_pp_zbv_zero_bubble,
        integration_tests.deepseek_v3_graph_pp_interleaved_1f1b,
        integration_tests.deepseek_v3_autoparallel_edp_shard_ep,
    ],
    ids=lambda recipe: recipe.__name__,
)
def test_deepseek_v3_ep_recipes_build_expert_sharded_weights(recipe):
    config = recipe()
    assert config.parallelism.expert_parallel_degree > 1
    moe_configs = [
        layer.feed_forward
        for layer in config.model.layers
        if isinstance(layer.feed_forward, MoE.Config)
    ]
    assert moe_configs
    for moe in moe_configs:
        for linear in (moe.routed_experts.w13, moe.routed_experts.w2):
            assert (
                linear.sharding_config.state_shardings["weight"]
                == expert_param_placement_sparse()
            )
