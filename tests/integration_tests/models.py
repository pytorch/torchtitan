# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from tests.integration_tests import OverrideDefinitions


def build_model_tests_list() -> list[OverrideDefinitions]:
    """
    Build a dictionary of model parallelism test configurations.
    This test suite is aimed at testing the model parallelism of torchtitan, and will broadly cover
    all the supported model parallelism patterns on all the supported models.

    Returns:
        A dictionary where each key is a model name and value is a list of OverrideDefinitions
    """
    model_tests = [
        # Integration Test Cases for DeepSeek V3
        OverrideDefinitions(
            [
                [
                    "--module deepseek_v3 --config deepseek_v3_debugmodel",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                    "--compile.enable",
                ],
            ],
            "DeepSeek V3 FSDP+EP+compile",
            "deepseek_v3_fsdp+ep+compile",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module deepseek_v3 --config deepseek_v3_debugmodel",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule DualPipeV",
                    # AC is not supported for DualPipeV yet
                    "--activation_checkpoint.mode 'none'",
                ],
            ],
            "PP dual pipe v schedule test",
            "pp_dualpipev",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module deepseek_v3 --config deepseek_v3_debugmodel",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule Interleaved1F1B",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 4",
                    "--parallelism.expert_tensor_parallel_degree 1",
                ],
            ],
            "DeepSeek V3 PP+FSDP+TP+EP",
            "deepseek_v3_pp+fsdp+tp+ep",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module deepseek_v3 --config deepseek_v3_debugmodel",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule Interleaved1F1B",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 2",
                    "--parallelism.expert_tensor_parallel_degree 2",
                ],
            ],
            "DeepSeek V3 PP+FSDP+TP+EP+ETP",
            "deepseek_v3_pp+fsdp+tp+ep+etp",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module deepseek_v3 --config deepseek_v3_debugmodel_flex_attn",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule Interleaved1F1B",
                    "--parallelism.expert_parallel_degree 4",
                    "--activation_checkpoint.mode 'selective'",
                    "--activation_checkpoint.selective_ac_option 'op'",
                ],
            ],
            "DeepSeek V3 Flex+PP+FSDP+EP+SACOP",
            "deepseek_v3_flex+pp+fsdp+ep+sacop",
            ngpu=8,
        ),
        # Integration Test Cases for Qwen3 dense and MoE model
        OverrideDefinitions(
            [
                [
                    "--module qwen3 --config qwen3_debugmodel",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "Qwen3 FSDP+TP",
            "qwen3_fsdp+tp",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module qwen3 --config qwen3_moe_debug",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 2",
                    "--parallelism.expert_tensor_parallel_degree 2",
                ],
            ],
            "Qwen3 FSDP+TP+EP+ETP",
            "qwen3_fsdp+tp+ep+etp",
            ngpu=4,
        ),
        # Integration Test Cases for Llama 4
        OverrideDefinitions(
            [
                [
                    "--module llama4 --config llama4_debugmodel",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule Interleaved1F1B",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 4",
                    "--parallelism.expert_tensor_parallel_degree 1",
                    "--compile.enable",
                ],
            ],
            "Llama 4 PP+FSDP+TP+EP+compile",
            "llama4_pp+fsdp+tp+ep+compile",
            ngpu=8,
        ),
        # Integration Test Cases for gpt-oss
        OverrideDefinitions(
            [
                [
                    "--module gpt_oss --config gpt_oss_debugmodel",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 4",
                    "--parallelism.expert_tensor_parallel_degree 1",
                    "--compile.enable",
                ],
            ],
            "Gpt-oss FSDP+TP+EP+compile",
            "gpt_oss_fsdp+tp+ep+compile",
            ngpu=8,
        ),
    ]

    return model_tests
