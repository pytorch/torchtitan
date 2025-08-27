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
        # Integration Test Cases for DeepSeek-V3
        OverrideDefinitions(
            [
                [
                    "--model.name deepseek_v3",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                ],
            ],
            "FSDP+DP2EP",
            "fsdp_dp2ep",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--model.name deepseek_v3",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.data_parallel_replicate_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "HSDP+TP",
            "hsdp+tp",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--model.name deepseek_v3",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule Interleaved1F1B",
                ],
            ],
            "FSDP+TP+PP",
            "fsdp+tp+pp",
            ngpu=8,
        ),
    ]

    return model_tests
