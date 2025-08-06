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
from .features import OverrideDefinitions, run_single_test


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def build_model_tests() -> list[OverrideDefinitions]:
    """
    Build a dictionary of model parallelism test configurations.
    This test suite is aimed at testing the model parallelism of torchtitan, and will broadly cover
    all the supported model parallelism patterns on all the supported models.

    Returns:
        A dictionary where each key is a model name and value is a list of OverrideDefinitions
    """
    model_tests = [
        # Test cases for DeepSeek-V3 models
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


def _run_cmd(cmd):
    return subprocess.run([cmd], text=True, shell=True)


def run_single_test(
    test_flavor: OverrideDefinitions, model_name: str, full_path: str, output_dir: str
):
    # run_test supports sequence of tests.
    test_name = test_flavor.test_name
    model_name_arg = f"--model.name {model_name}"
    dump_folder_arg = f"--job.dump_folder {output_dir}/{test_name}"

    all_ranks = ",".join(map(str, range(test_flavor.ngpu)))

    for idx, override_arg in enumerate(test_flavor.override_args):
        cmd = f"CONFIG_FILE={full_path} NGPU={test_flavor.ngpu} LOG_RANK={all_ranks} ./run_train.sh"
        # dump compile trace for debugging purpose
        cmd = f'TORCH_TRACE="{output_dir}/{test_name}/compile_trace" ' + cmd
        cmd += " " + model_name_arg
        cmd += " " + dump_folder_arg
        if override_arg:
            cmd += " " + " ".join(override_arg)
        logger.info(
            f"=====Integration test, flavor : {test_flavor.test_descr}, command : {cmd}====="
        )

        result = _run_cmd(cmd)
        logger.info(result.stdout)
        if result.returncode != 0:
            raise Exception(
                f"Integration test failed, flavor : {test_flavor.test_descr}, command : {cmd}"
            )


def run_tests(args):
    """Run integration tests to test the supported model parallelsim of TorchTitan"""
    test_case_dict = build_model_parallelism_tests()

    for model_name, test_list in test_case_dict.items():
        for test_flavor in test_list:
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
    parser.add_argument("output_dir", help="Directory to store test outputs")
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
        "--ngpu", default=8, type=int, help="Maximum number of GPUs to use"
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.listdir(args.output_dir):
        raise RuntimeError("Please provide an empty output directory.")
    run_tests(args)


if __name__ == "__main__":
    main()
