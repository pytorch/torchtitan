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


def test_rl_integration_recipe_runs_dist_moe_on_both_roles():
    # The RL integration test guards the Dist-MoE path only if its recipe really
    # puts Dist-MoE on the trainer and on the generator, with graphs and uneven DP.
    from torchtitan.models.common.dist_moe import DistMoeRoutedExperts, DistMoeRuntime
    from torchtitan.models.common.moe import RoutedExperts
    from torchtitan_recipes.tests.rl.alphabet_sort import (
        rl_grpo_moe_debug_dist_moe_tp2_ep4,
    )

    config = rl_grpo_moe_debug_dist_moe_tp2_ep4()

    assert list(config.model.traverse(DistMoeRoutedExperts.Config))
    assert not [
        entry
        for entry in config.model.traverse(RoutedExperts.Config)
        if type(entry[1]) is RoutedExperts.Config
    ]
    assert isinstance(config.trainer.dist_moe_runtime, DistMoeRuntime.Config)
    assert isinstance(config.generator.dist_moe_runtime, DistMoeRuntime.Config)
    assert config.generator.dist_moe_runtime.inference
    assert config.generator.cuda_graph.mode == "FULL"
    # Two vLLM DP replicas, so ranks of the EP group see different token counts.
    assert config.generator.parallelism.data_parallel_degree == 2
    assert config.generator.parallelism.expert_parallel_degree == 4
    assert config.trainer.parallelism.expert_parallel_degree == 4


def test_b200_rl_suite_runs_kda_and_dist_moe():
    from tests.integration_tests import get_importable_config_module
    from tests.rl.integration_tests.rl import (
        build_b200_rl_test_list,
        build_rl_kda_test_list,
    )

    definitions = {d.test_name: d for d in build_b200_rl_test_list()}

    for kda in build_rl_kda_test_list():
        assert kda.test_name in definitions
    dist_moe_test = definitions["rl_grpo_moe_debug_dist_moe_tp2_ep4"]
    assert dist_moe_test.ngpu == 8
    (config_fn,) = dist_moe_test.configs
    assert (
        get_importable_config_module(config_fn)
        == "torchtitan_recipes.tests.rl.alphabet_sort"
    )
