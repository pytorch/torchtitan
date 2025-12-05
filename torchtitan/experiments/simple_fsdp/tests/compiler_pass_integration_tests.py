# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.run_tests import run_tests


def build_simple_fsdp_test_list() -> list[OverrideDefinitions]:
    """
    key is the config file name and value is a list of OverrideDefinitions
    that is used to generate variations of integration tests based on the
    same root config file.
    """
    integration_tests_flavors = [
        OverrideDefinitions(
            [
                [
                    "--model.name simple_fsdp.llama3",
                    "--model.flavor 8B",
                    "--compile.enable",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                    "--compile.backend aot_eager",
                    "--compile.graph_passes auto_bucketing",
                ],
            ],
            "1D+autobucketing",
            "1d_autobucketing",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--model.name simple_fsdp.llama3",
                    "--model.flavor 8B",
                    "--compile.enable",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                    "--compile.backend aot_eager",
                    "--compile.graph_passes transformer_block_bucketing",
                ],
            ],
            "1D+transformer_block_bucketing",
            "1d_transformer_block_bucketing",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--model.name simple_fsdp.llama3",
                    "--model.flavor 8B",
                    "--parallelism.tensor_parallel_degree 2",
                    "--compile.enable",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                    "--compile.backend aot_eager",
                    "--compile.graph_passes auto_bucketing",
                ],
            ],
            "2D+autobucketing",
            "2d_autobucketing",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--model.name simple_fsdp.llama3",
                    "--model.flavor 8B",
                    "--parallelism.tensor_parallel_degree 2",
                    "--compile.enable",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                    "--compile.backend aot_eager",
                    "--compile.graph_passes transformer_block_bucketing",
                ],
            ],
            "2D+transformer_block_bucketing",
            "2d_transformer_block_bucketing",
            ngpu=8,
        ),
        # TODO(ruisizhang123): add back after passes + PP is supported
        # OverrideDefinitions(
        #     [
        #         [
        #             "--model.name simple_fsdp.llama3",
        #             "--model.flavor 8B",
        #             "--parallelism.tensor_parallel_degree 2",
        #             "--parallelism.pipeline_parallel_degree 2",
        #             "--compile.enable",
        #             "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
        #             "--compile.backend aot_eager",
        #             "--compile.graph_passes auto_bucketing",
        #         ],
        #     ],
        #     "3D+autobucketing",
        #     "3d_autobucketing",
        #     ngpu=8,
        # ),
        # OverrideDefinitions(
        #     [
        #         [
        #             "--model.name simple_fsdp.llama3",
        #             "--model.flavor 8B",
        #             "--parallelism.tensor_parallel_degree 2",
        #             "--parallelism.pipeline_parallel_degree 2",
        #             "--compile.enable",
        #             "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
        #             "--compile.backend aot_eager",
        #             "--compile.graph_passes transformer_block_bucketing",
        #         ],
        #     ],
        #     "3D+transformer_block_bucketing",
        #     "3d_transformer_block_bucketing",
        #     ngpu=8,
        # ),
        # OverrideDefinitions(
        #     [
        #         [
        #             "--model.name simple_fsdp.llama3",
        #             "--model.flavor 8B",
        #             "--parallelism.tensor_parallel_degree 2",
        #             "--parallelism.context_parallel_degree 2",
        #             "--compile.enable",
        #             "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
        #             "--compile.backend aot_eager",
        #             "--compile.graph_passes auto_bucketing",
        #         ],
        #     ],
        #     "FSDP+TP+CP+autobucketing",
        #     "fsdp+tp+cp_autobucketing",
        #     ngpu=8,
        # ),
        OverrideDefinitions(
            [
                [
                    "--model.name simple_fsdp.llama3",
                    "--model.flavor 8B",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.context_parallel_degree 2",
                    "--compile.enable",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                    "--compile.backend aot_eager",
                    "--compile.graph_passes transformer_block_bucketing",
                ],
            ],
            "FSDP+TP+CP+transformer_block_bucketing",
            "fsdp+tp+cp_transformer_block_bucketing",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--model.name simple_fsdp.deepseek_v3",
                    "--model.flavor 16B",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                    "--compile.backend aot_eager",
                    "--compile.graph_passes auto_bucketing",
                ],
            ],
            "FSDP+EP+autobucketing",
            "fsdp+ep_autobucketing",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--model.name simple_fsdp.deepseek_v3",
                    "--model.flavor 16B",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                    "--compile.backend aot_eager",
                    "--compile.graph_passes transformer_block_bucketing",
                ],
            ],
            "FSDP+EP+transformer_block_bucketing",
            "fsdp+ep_transformer_block_bucketing",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--model.name simple_fsdp.deepseek_v3",
                    "--model.flavor 16B",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 4",
                    "--parallelism.expert_tensor_parallel_degree 1",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                    "--compile.backend aot_eager",
                    "--compile.graph_passes auto_bucketing",
                ],
            ],
            "FSDP+TP+EP+autobucketing",
            "fsdp+tp+ep_autobucketing",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--model.name simple_fsdp.deepseek_v3",
                    "--model.flavor 16B",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 4",
                    "--parallelism.expert_tensor_parallel_degree 1",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                    "--compile.backend aot_eager",
                    "--compile.graph_passes transformer_block_bucketing",
                ],
            ],
            "FSDP+TP+EP+transformer_block_bucketing",
            "fsdp+tp+ep_transformer_block_bucketing",
            ngpu=4,
        ),
    ]
    return integration_tests_flavors


_TEST_SUITES_FUNCTION = {
    "simple_fsdp": build_simple_fsdp_test_list,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument(
        "--comm_mode",
        default="default",
        choices=["default", "fake_backend", "local_tensor"],
        help="Communication mode to validate tests",
    )
    parser.add_argument(
        "--config_path",
        default="./tests/integration_tests/base_config.toml",
        help="Base config path for integration tests. This is the config that will be used as a base for all tests.",
    )
    parser.add_argument(
        "--test_name",
        default="all",
        help="test to run, acceptable values: `test_name` in `build_test_list` (default: all)",
    )
    parser.add_argument("--ngpu", default=8, type=int)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.listdir(args.output_dir):
        raise RuntimeError("Please provide an empty output directory.")

    test_list = _TEST_SUITES_FUNCTION["simple_fsdp"]()
    run_tests(args, test_list)


if __name__ == "__main__":
    main()
