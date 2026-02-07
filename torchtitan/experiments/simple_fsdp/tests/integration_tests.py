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
                    "--compile.enable",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                ],
            ],
            "1D",
            "1d",
        ),
        OverrideDefinitions(
            [
                [
                    "--model.name simple_fsdp.llama3",
                    "--compile.enable",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                    "--compile.backend aot_eager",
                    "--compile.graph_passes auto_bucketing",
                ],
            ],
            "1D+autobucketing",
            "1d_autobucketing",
        ),
        OverrideDefinitions(
            [
                [
                    "--model.name simple_fsdp.llama3",
                    "--compile.enable",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                    "--compile.backend aot_eager",
                    "--compile.graph_passes transformer_block_bucketing",
                ],
            ],
            "1D+transformer_block_bucketing",
            "1d_transformer_block_bucketing",
        ),
        OverrideDefinitions(
            [
                [
                    "--model.name simple_fsdp.llama3",
                    "--compile.enable",
                    "--activation_checkpoint.mode selective",
                    "--activation_checkpoint.selective_ac_option op",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                ],
            ],
            "1D with selective op AC",
            "1d_sac_op",
        ),
        OverrideDefinitions(
            [
                [
                    "--model.name simple_fsdp.llama3",
                    "--compile.enable",
                    "--activation_checkpoint.mode full",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                ],
            ],
            "1D with full AC",
            "1d_full_ac",
        ),
        OverrideDefinitions(
            [
                [
                    "--model.name simple_fsdp.llama3",
                    "--compile.enable",
                    "--parallelism.tensor_parallel_degree 2",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                ],
            ],
            "2D",
            "2d",
        ),
        # TODO: re-enable this test once the async TP issue is fixed
        OverrideDefinitions(
            [
                [
                    "--model.name simple_fsdp",
                    "--compile.enable",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.enable_async_tensor_parallel",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                ],
            ],
            "2D async TP",
            "2d_asynctp",
            disabled=True,
        ),
        OverrideDefinitions(
            [
                [
                    "--model.name simple_fsdp.llama3",
                    "--compile.enable",
                    "--checkpoint.enable",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                ],
                [
                    "--model.name simple_fsdp.llama3",
                    "--compile.enable",
                    "--checkpoint.enable",
                    "--training.steps 20",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                ],
            ],
            "Checkpoint Integration Test - Save Load Full Checkpoint",
            "full_checkpoint",
        ),
        OverrideDefinitions(
            [
                [
                    "--model.name simple_fsdp.llama3",
                    "--compile.enable",
                    "--checkpoint.enable",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                ],
                [
                    "--model.name simple_fsdp.llama3",
                    "--compile.enable",
                    "--training.steps 20",
                    "--checkpoint.enable",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                ],
            ],
            "PP+DP+TP 3D test with save/load resume ckpt",
            "pp_dp_tp",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--model.name simple_fsdp.llama3",
                    "--compile.enable",
                    "--parallelism.data_parallel_shard_degree 1",
                    "--parallelism.data_parallel_replicate_degree 4",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                ]
            ],
            "DDP",
            "ddp",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--model.name simple_fsdp.llama3",
                    "--compile.enable",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.data_parallel_replicate_degree 2",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                ]
            ],
            "HSDP",
            "hsdp",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--model.name simple_fsdp.llama3",
                    "--compile.enable",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.data_parallel_replicate_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                ]
            ],
            "HSDP+TP",
            "hsdp+tp",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--model.name simple_fsdp.llama3",
                    "--compile.enable",
                    "--parallelism.data_parallel_replicate_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                ]
            ],
            "DDP+TP",
            "ddp+tp",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--model.name simple_fsdp.llama3",
                    "--compile.enable",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.data_parallel_replicate_degree 2",
                    "--parallelism.context_parallel_degree 2",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                ]
            ],
            "HSDP+CP (with dp_shard)",
            "hsdp+cp_with_dp_shard",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--model.name simple_fsdp.llama3",
                    "--compile.enable",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.context_parallel_degree 2",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                ]
            ],
            "FSDP+TP+CP",
            "fsdp+tp+cp",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--model.name simple_fsdp.llama3",
                    "--compile.enable",
                    "--checkpoint.enable",
                    "--training.steps 10",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                ],
                # Save at [dp:4] and load at [dp:2, tp:2]. Note that the dataloader should be
                # excluded during loading to avoid errors caused by mismatched dp_degree.
                [
                    "--model.name simple_fsdp.llama3",
                    "--compile.enable",
                    "--checkpoint.enable",
                    "--checkpoint.exclude_from_loading lr_scheduler,dataloader,optimizer",
                    "--parallelism.tensor_parallel_degree 2",
                    "--training.steps 20",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                ],
                # load at [tp:4].
                [
                    "--model.name simple_fsdp.llama3",
                    "--compile.enable",
                    "--checkpoint.enable",
                    "--checkpoint.exclude_from_loading lr_scheduler,dataloader,optimizer",
                    "--parallelism.tensor_parallel_degree 4",
                    "--training.steps 30",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                ],
            ],
            "Optional checkpoint",
            "optional_checkpoint",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--model.name simple_fsdp.deepseek_v3",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                ],
            ],
            "FSDP+EP",
            "fsdp+ep",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--model.name simple_fsdp.deepseek_v3",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 4",
                    "--parallelism.expert_tensor_parallel_degree 1",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                ],
            ],
            "FSDP+TP+EP",
            "fsdp+tp+ep",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--model.name simple_fsdp.deepseek_v3",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 2",
                    "--parallelism.expert_tensor_parallel_degree 2",
                    "--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config",
                ],
            ],
            "FSDP+TP+EP+ETP",
            "fsdp+tp+ep+etp",
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
