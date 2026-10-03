# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The RL integration test for Dist-MoE only guards anything if its recipe is wired right."""

import pytest

pytest.importorskip(
    "dist_moe",
    reason="Dist-MoE integration tests require the optional dist_moe package",
)
from torchtitan.models.common.dist_moe import DistMoeInferenceRuntime  # noqa: E402


def test_rl_integration_recipe_runs_dist_moe_on_both_roles():
    # The RL integration test guards the Dist-MoE path only if its recipe really
    # puts Dist-MoE on the trainer and on the generator, with graphs and uneven DP.
    from torchtitan.models.common.dist_moe import DistMoeRoutedExperts, DistMoeRuntime
    from torchtitan.models.common.moe import RoutedExperts
    from torchtitan_recipes.tests.rl import rl_grpo_moe_debug_dist_moe_tp2_ep4

    config = rl_grpo_moe_debug_dist_moe_tp2_ep4()

    assert list(config.model.traverse(DistMoeRoutedExperts.Config))
    assert not [
        entry
        for entry in config.model.traverse(RoutedExperts.Config)
        if type(entry[1]) is RoutedExperts.Config
    ]
    assert isinstance(config.trainer.dist_moe, DistMoeRuntime.Config)
    assert isinstance(config.generator.dist_moe_runtime, DistMoeInferenceRuntime.Config)
    assert config.generator.cuda_graph.mode == "FULL_DECODE_ONLY"
    # Two vLLM DP replicas, so ranks of the EP group see different token counts.
    assert config.generator.parallelism.data_parallel_degree == 2
    assert config.generator.parallelism.expert_parallel_degree == 4
    assert config.trainer.parallelism.expert_parallel_degree == 4


def test_rl_dist_moe_suite_lists_the_recipe():
    from tests.integration_tests import get_importable_config_module
    from tests.rl.integration_tests.rl import build_rl_dist_moe_test_list

    (definition,) = build_rl_dist_moe_test_list()

    assert definition.test_name == "rl_grpo_moe_debug_dist_moe_tp2_ep4"
    assert definition.ngpu == 8
    (config_fn,) = definition.configs
    assert get_importable_config_module(config_fn) == "torchtitan_recipes.tests.rl"
