# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import subprocess
from collections import defaultdict

from .integration_tests import OverrideDefinitions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def build_test_list():
    """
    key is the config file name and value is a list of OverrideDefinitions
    that is used to generate variations of integration tests based on the
    same root config file.
    """
    integration_tests_flavors = []
    integration_tests_flavors.extend(
        [
            TestCaseConfigs(
                [
                    [
                        "--training.compile",
                        "--parallelism.tensor_parallel_degree 2",
                        "--parallelism.enable_async_tensor_parallel",
                    ],
                ],
                "2D async TP compile",
                "2d_asynctp_compile",
                supported_models=["llama3"],
            ),
            TestCaseConfigs(
                [
                    "--model.converters float8",
                    "--float8.enable_fsdp_float8_all_gather",
                    "--float8.precompute_float8_dynamic_scale_for_fsdp",
                ],
                "Float8 test",
                "float8",
                supported_models=["llama3"],
            ),
            TestCaseConfigs(
                [
                    "--compile.enable",
                    "--parallelism.data_parallel_shard_degree=2",
                    "--parallelism.tensor_parallel_degree=2",
                    "--parallelism.pipeline_parallel_degree=2",
                    "--parallelism.enable_async_tensor_parallel",
                    "--model.converters float8",
                    "--float8.enable_fsdp_float8_all_gather",
                    "--float8.precompute_float8_dynamic_scale_for_fsdp",
                ]
            ],
            "FSDP+async TP+PP+torch.compile+Float8",
            "fsdp+tp+cp+compile+float8",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--compile.enable",
                    "--parallelism.data_parallel_shard_degree=2",
                    "--parallelism.data_parallel_replicate_degree=2",
                    "--parallelism.context_parallel_degree=2",
                    "--model.converters float8",
                    "--float8.enable_fsdp_float8_all_gather",
                    "--float8.precompute_float8_dynamic_scale_for_fsdp",
                ]
            ],
            "HSDP+CP+torch.compile+Float8",
            "hsdp+cp+compile+float8",
            ngpu=8,
        ),
    ]
    return integration_tests_flavors


def run_h100_tests(args):
    # If user specifies a specific test name, the test_suite argument is ignored
    if args.test_name != "all":
        args.test_suite = "all"

    # build integration tests list
    test_list = build_h100_test_list()

    for test_flavor in test_list:
        model_names = test_flavor.supported_models
        for model_name in model_names:
            # Filter by test_name if specified
            if args.test_name != "all" and test_flavor.test_name != args.test_name:
                continue

            # Check if config file exists
            assert args.config_path.endswith(
                ".toml"
            ), "Base config path must end with .toml"
            assert os.path.exists(
                args.config_path
            ), f"Base config path {args.config_path} does not exist"

            # Check if we have enough GPUs
            if args.ngpu < test_flavor.ngpu:
                logger.info(
                    f"Skipping test {test_flavor.test_name} that requires {test_flavor.ngpu} gpus,"
                    f" because --ngpu arg is {args.ngpu}"
                )
            else:
                run_single_test(
                    test_flavor, model_name, args.config_path, args.output_dir
                )


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
        help="Specific test name to run (e.g., 'tp_only', 'full_checkpoint'). Use 'all' to run all tests (default: all)",
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", "llama3", "deepseek_v3"],
        help="Specify the model to run tests on (default: llama3)",
    )
    parser.add_argument("--ngpu", default=8, type=int)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.listdir(args.output_dir):
        raise RuntimeError("Please provide an empty output directory.")
    run_h100_tests(args)


if __name__ == "__main__":
    main()
