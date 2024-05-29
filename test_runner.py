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
from dataclasses import dataclass
from typing import Sequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


@dataclass
class OverrideDefinitions:
    """
    This class is used to define the override definitions for the integration tests.
    """

    override_args: Sequence[Sequence[str]] = tuple(tuple(" "))
    test_descr: str = "default"
    test_id: str = "default"
    requires_seed_checkpoint: bool = False
    ngpu: int = 4


def build_test_list():
    """
    key is the config file name and value is a list of OverrideDefinitions
    that is used to generate variations of integration tests based on the
    same root config file.
    """
    integration_tests_flavors = defaultdict(list)
    integration_tests_flavors["debug_model.toml"] = [
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable_checkpoint",
                    "--experimental.pipeline_parallel_degree 2",
                    "--experimental.pipeline_parallel_split_points layers.1",
                    "--experimental.pipeline_parallel_schedule 1f1b",
                    "--training.data_parallel_degree 1",
                ],
            ],
            "PP 1D test 1f1b",
            "pp_1f1b",
            requires_seed_checkpoint=True,
            ngpu=2,
        ),
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable_checkpoint",
                    "--experimental.pipeline_parallel_degree 2",
                    "--experimental.pipeline_parallel_split_points layers.1",
                    "--experimental.pipeline_parallel_schedule gpipe",
                    "--training.data_parallel_degree 1",
                ],
            ],
            "PP 1D test gpipe",
            "pp_gpipe",
            requires_seed_checkpoint=True,
            ngpu=2,
        ),
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable_checkpoint",
                    "--experimental.pipeline_parallel_degree 2",
                    "--experimental.pipeline_parallel_split_points layers.1",
                    "--experimental.pipeline_parallel_schedule 1f1b",
                    "--training.data_parallel_degree 2",
                ],
            ],
            "PP+DP 1f1b 2D test",
            "pp_dp_1f1b",
            requires_seed_checkpoint=True,
        ),
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable_checkpoint",
                    "--experimental.pipeline_parallel_degree 2",
                    "--experimental.pipeline_parallel_split_points layers.1",
                    "--experimental.pipeline_parallel_schedule gpipe",
                    "--training.data_parallel_degree 2",
                ],
            ],
            "PP+DP gpipe 2D test",
            "pp_dp_gpipe",
            requires_seed_checkpoint=True,
        ),
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable_checkpoint",
                    "--experimental.pipeline_parallel_degree 2",
                    "--experimental.pipeline_parallel_split_points layers.1",
                    "--training.tensor_parallel_degree 2",
                    "--model.norm_type rmsnorm",  # fused_rmsnorm not yet compatible with TP
                ],
            ],
            "PP+TP 2D test",
            "pp_tp",
            requires_seed_checkpoint=True,
        ),
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable_checkpoint",
                    "--experimental.pipeline_parallel_degree 2",
                    "--experimental.pipeline_parallel_split_points layers.1",
                    "--model.norm_type rmsnorm",  # fused_rmsnorm not yet compatible with tracer
                ],
            ],
            "PP tracer frontend test",
            "pp_tracer",
            requires_seed_checkpoint=True,
        ),
        OverrideDefinitions(
            [
                [],
            ],
            "default",
            "default",
        ),
        OverrideDefinitions(
            [
                [
                    "--training.compile --model.norm_type=rmsnorm",
                ],
            ],
            "1D compile",
            "1d_compile",
        ),
        OverrideDefinitions(
            [
                [
                    "--training.compile --training.tensor_parallel_degree 2 --model.norm_type=rmsnorm",
                ],
            ],
            "2D compile",
            "2d_compile",
        ),
        OverrideDefinitions(
            [
                [
                    "--training.tensor_parallel_degree 2 --model.norm_type=rmsnorm",
                ],
            ],
            "Eager mode 2DParallel",
            "eager_2d",
        ),
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable_checkpoint",
                ],
                [
                    "--checkpoint.enable_checkpoint",
                    "--training.steps 20",
                ],
            ],
            "Checkpoint Integration Test - Save Load Full Checkpoint",
            "full_checkpoint",
        ),
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable_checkpoint",
                    "--checkpoint.model_weights_only",
                ],
            ],
            "Checkpoint Integration Test - Save Model Weights Only fp32",
            "model_weights_only_fp32",
        ),
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable_checkpoint",
                    "--checkpoint.model_weights_only",
                    "--checkpoint.export_dtype bfloat16",
                ],
            ],
            "Checkpoint Integration Test - Save Model Weights Only bf16",
            "model_weights_only_bf16",
        ),
    ]
    return integration_tests_flavors


def _run_cmd(cmd):
    return subprocess.run(
        [cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=True,
    )


def run_test(test_flavor: OverrideDefinitions, full_path: str, output_dir: str):
    # run_test supports sequence of tests.
    for override_arg in test_flavor.override_args:
        test_id = test_flavor.test_id
        dump_folder_arg = f"--job.dump_folder {output_dir}/{test_id}"

        cmd = f"CONFIG_FILE={full_path} NGPU={test_flavor.ngpu} LOG_RANK=0,1,2,3 ./run_llama_train.sh"
        cmd += " " + dump_folder_arg

        if override_arg:
            cmd += " " + " ".join(override_arg)
        logger.info(
            f"=====Integration test, flavor : {test_flavor.test_descr}, command : {cmd}====="
        )

        if test_flavor.requires_seed_checkpoint:
            logger.info("Creating seed checkpoint")
            result = _run_cmd(
                f"CONFIG_FILE={full_path} ./create_seed_checkpoint.sh {dump_folder_arg}"
            )
            logger.info(result.stdout)

        result = _run_cmd(cmd)
        logger.info(result.stdout)
        if result.returncode != 0:
            raise Exception(
                f"Integration test failed, flavor : {test_flavor.test_descr}, command : {cmd}"
            )


def run_tests(args):
    integration_tests_flavors = build_test_list()
    for config_file in os.listdir(args.config_dir):
        if config_file.endswith(".toml"):
            full_path = os.path.join(args.config_dir, config_file)
            with open(full_path, "rb") as f:
                config = tomllib.load(f)
                is_integration_test = config["job"].get(
                    "use_for_integration_test", False
                )
                if is_integration_test:
                    for test_flavor in integration_tests_flavors[config_file]:
                        if args.test == "all" or test_flavor.test_id == args.test:
                            run_test(test_flavor, full_path, args.output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--config_dir", default="./train_configs")
    parser.add_argument("--test", default="all", help="test to run, acceptable values: `test_id` in `build_test_list` (default: all)")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.listdir(args.output_dir):
        raise RuntimeError("Please provide an empty output directory.")
    run_tests(args)


if __name__ == "__main__":
    main()
