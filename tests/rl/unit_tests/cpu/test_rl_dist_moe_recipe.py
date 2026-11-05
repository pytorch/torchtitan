# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The RL integration test for Dist-MoE only guards anything if its recipe is wired right."""

import pytest

dist_moe = pytest.importorskip(
    "dist_moe",
    reason="Dist-MoE integration tests require the optional dist_moe package",
)


@pytest.mark.parametrize(
    (
        "recipe_name",
        "tensor_parallel_degree",
        "expert_parallel_degree",
        "expert_precision",
    ),
    [
        ("rl_grpo_moe_debug_dist_moe_tp1_ep2", 1, 2, "bf16"),
        ("rl_grpo_moe_debug_dist_moe_mxfp8_tp1_ep2", 1, 2, "mxfp8"),
        (
            "rl_grpo_moe_debug_dist_moe_nvfp4_fwd_bf16_bwd_tp1_ep2",
            1,
            2,
            "nvfp4_fwd_bf16_bwd",
        ),
        ("rl_grpo_moe_debug_dist_moe_tp2_ep4", 2, 4, "bf16"),
    ],
)
def test_rl_integration_recipe_runs_dist_moe_on_both_roles(
    recipe_name, tensor_parallel_degree, expert_parallel_degree, expert_precision
):
    # The RL integration test guards the Dist-MoE path only if its recipe really
    # puts Dist-MoE on the trainer and on the generator, with graphs and uneven DP.
    from torchtitan.models.common.dist_moe import (
        DistMoeRoutedExperts,
        DistMoeRuntime,
        MXFP8DistMoeRoutedExperts,
        NVFP4FwdBF16BwdDistMoeRoutedExperts,
    )
    from torchtitan.models.common.moe import RoutedExperts
    from torchtitan_recipes.tests.rl import alphabet_sort

    config = getattr(alphabet_sort, recipe_name)()

    for model_config in (config.trainer.model, config.generator.model):
        assert list(model_config.traverse(DistMoeRoutedExperts.Config))
        assert not [
            entry
            for entry in model_config.traverse(RoutedExperts.Config)
            if type(entry[1]) is RoutedExperts.Config
        ]
    assert isinstance(config.trainer.dist_moe_runtime, DistMoeRuntime.Config)
    assert isinstance(config.generator.dist_moe_runtime, DistMoeRuntime.Config)
    assert config.generator.dist_moe_runtime.inference
    assert config.generator.engine.cuda_graph.mode == "FULL"
    # Two vLLM DP replicas, so ranks of the EP group see different token counts.
    assert config.generator.parallelism.data_parallel_degree == 2
    assert config.generator.parallelism.tensor_parallel_degree == tensor_parallel_degree
    assert config.generator.parallelism.expert_parallel_degree == expert_parallel_degree
    assert config.trainer.parallelism.tensor_parallel_degree == tensor_parallel_degree
    assert config.trainer.parallelism.expert_parallel_degree == expert_parallel_degree
    if tensor_parallel_degree == 1:
        assert config.async_loop.validation.steps == 0
    expected_experts_type = {
        "bf16": DistMoeRoutedExperts.Config,
        "mxfp8": MXFP8DistMoeRoutedExperts.Config,
        "nvfp4_fwd_bf16_bwd": NVFP4FwdBF16BwdDistMoeRoutedExperts.Config,
    }[expert_precision]
    for model_config in (config.trainer.model, config.generator.model):
        assert list(model_config.traverse(expected_experts_type))
    if expert_precision == "mxfp8":
        trainer_experts = [
            experts
            for _, experts, _, _ in config.trainer.model.traverse(
                MXFP8DistMoeRoutedExperts.Config
            )
        ]
        generator_experts = [
            experts
            for _, experts, _, _ in config.generator.model.traverse(
                MXFP8DistMoeRoutedExperts.Config
            )
        ]
        assert not any(experts.mega_fused_inference for experts in trainer_experts)
        assert all(experts.mega_fused_inference for experts in generator_experts)
        assert all(
            experts.block_scaled_config.pipeline == "mega"
            for experts in generator_experts
        )
    elif expert_precision == "nvfp4_fwd_bf16_bwd":
        for model_config in (config.trainer.model, config.generator.model):
            experts = [
                experts
                for _, experts, _, _ in model_config.traverse(
                    NVFP4FwdBF16BwdDistMoeRoutedExperts.Config
                )
            ]
            assert all(
                expert.block_scaled_config.format is dist_moe.BlockScaledFormat.NVFP4
                for expert in experts
            )
            assert all(
                expert.block_scaled_config.pipeline == "staged" for expert in experts
            )


def test_b200_rl_suite_runs_h100_tests_and_dist_moe():
    from tests.integration_tests import get_importable_config_module
    from tests.integration_tests.rl import (
        build_b200_rl_test_list,
        build_h100_rl_test_list,
    )

    definitions = {d.test_name: d for d in build_b200_rl_test_list()}

    for h100_test in build_h100_rl_test_list():
        assert h100_test.test_name in definitions
    dist_moe_test = definitions["rl_grpo_moe_debug_dist_moe_tp2_ep4"]
    assert dist_moe_test.ngpu == 8
    (config_fn,) = dist_moe_test.configs
    assert (
        get_importable_config_module(config_fn)
        == "torchtitan_recipes.tests.rl.alphabet_sort"
    )
