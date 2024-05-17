# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

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


def build_test_list(args):
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
                    f"--job.dump_folder {args.output_dir}/default/",
                ],
            ],
            "Default",
        ),
        OverrideDefinitions(
            [
                [
                    "--training.compile --model.norm_type=rmsnorm",
                    f"--job.dump_folder {args.output_dir}/1d/",
                ],
            ],
            "1D compile",
        ),
        OverrideDefinitions(
            [
                [
                    "--training.compile --training.tensor_parallel_degree 2 --model.norm_type=rmsnorm
                    f"--job.dump_folder {args.output_dir}/1d_compile/",
                ],
            ],
            "2D compile",
        ),
        OverrideDefinitions(
            [
                [
                    "--training.tensor_parallel_degree 2 --model.norm_type=rmsnorm",
                    f"--job.dump_folder {args.output_dir}/eager_2d/",
                ],
            ],
            "Eager mode 2DParallel",
        ),
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable_checkpoint",
                    f"--job.dump_folder {args.output_dir}/full_checkpoint/",
                ],
                [
                    "--checkpoint.enable_checkpoint",
                    f"--job.dump_folder {args.output_dir}/full_checkpoint/",
                    "--training.steps 20",
                ],
            ],
            "Checkpoint Integration Test - Save Load Full Checkpoint",
        ),
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable_checkpoint",
                    f"--job.dump_folder {args.output_dir}/model_weights_only_fp32/",
                    "--checkpoint.model_weights_only",
                ],
            ],
            "Checkpoint Integration Test - Save Model Weights Only fp32",
        ),
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable_checkpoint",
                    f"--job.dump_folder {args.output_dir}/model_weights_only_bf16/",
                    "--checkpoint.model_weights_only",
                    "--checkpoint.export_dtype bfloat16",
                ],
            ],
            "Checkpoint Integration Test - Save Model Weights Only bf16",
        ),
    ]
    return integration_tests_flavors


def run_test(test_flavor: OverrideDefinitions, full_path: str):
    # run_test supports sequence of tests.
    for override_arg in test_flavor.override_args:
        cmd = f"CONFIG_FILE={full_path} NGPU=4 LOG_RANK=0,1,2,3 ./run_llama_train.sh"
        if override_arg:
            cmd += " " + " ".join(override_arg)
        print(
            f"=====Integration test, flavor : {test_flavor.test_descr}, command : {cmd}====="
        )
        result = subprocess.run(
            [cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            shell=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            raise Exception(
                f"Integration test failed, flavor : {test_flavor.test_descr}, command : {cmd}"
            )


def run_tests(args):
    integration_tests_flavors = build_test_list(args)
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
                        run_test(test_flavor, full_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--config_dir", default="./train_configs")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.listdir(args.output_dir):
        raise RuntimeError("Please provide an empty output directory.")
    run_tests(args)


if __name__ == "__main__":
    main()
