# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from tests.integration_tests import OverrideDefinitions


def build_experimental_model_tests_list() -> list[OverrideDefinitions]:
    """
    Build a dictionary of model parallelism test configurations.
    This test suite is aimed at testing the model under experiments folder of torchtitan,
    and will be used to monitor the side effect of core changes to experimental models.

    Returns:
        A dictionary where each key is a model name and value is a list of OverrideDefinitions
    """
    model_tests = [
        # Integration Test Cases for Llama4
        OverrideDefinitions(
            [
                [
                    "--model.name llama4",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule Interleaved1F1B",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 2",
                    "--parallelism.expert_tensor_parallel_degree 2",
                ],
            ],
            "PP+FSDP+TP+EP+ETP",
            "llama4_pp+fsdp+tp+ep+etp",
            ngpu=8,
        ),
        # Integration Test Cases for Qwen3 Dense
        OverrideDefinitions(
            [
                [
                    "--model.name qwen3",
                    "--model.flavor 0.6B",
                    "--parallelism.data_parallel_shard_degree 4",
                ],
            ],
            "Qwen3 FSDP",
            "qwen3_fsdp",
            ngpu=4,
        ),
        # Integration Test Cases for Qwen3 MoE
        # TODO: replace this test case with n-D parallelism
        OverrideDefinitions(
            [
                [
                    "--model.name qwen3",
                    "--model.flavor debugmodel_moe",
                    "--parallelism.data_parallel_shard_degree 4",
                ],
            ],
            "Qwen3 MoE FSDP",
            "qwen3_moe_fsdp",
            ngpu=4,
        ),
        # Integration Test Cases for VLM
        # TODO: replace this test case with n-D parallelism
        OverrideDefinitions(
            [
                [
                    "--experimental.custom_args_module torchtitan.experiments.vlm.assets.job_config",
                    "--model.name vlm",
                    "--training.dataset cc12m-test",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--data.max_patches_per_image 1024",
                    "--data.max_images_per_batch 64",
                ],
            ],
            "VLM FSDP",
            "vlm_fsdp",
            ngpu=4,
        ),
    ]

    return model_tests
