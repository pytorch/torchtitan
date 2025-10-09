# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import os
import subprocess
from dataclasses import dataclass
from typing import Sequence


@dataclass
class OverrideDefinitions:
    """
    This class is used to define the override definitions for the integration tests.
    """

    override_args: Sequence[Sequence[str]] = tuple(tuple(" "))
    test_descr: str = "default"
    test_name: str = "default"
    ngpu: int = 8

    def __repr__(self):
        return self.test_descr


def _run_cmd(cmd):
    return subprocess.run([cmd], text=True, shell=True)


def run_single_test(test_flavor: OverrideDefinitions, full_path: str, output_dir: str):
    # run_test supports sequence of tests.
    test_name = test_flavor.test_name
    dump_folder_arg = f"--job.dump_folder {output_dir}/{test_name}"

    all_ranks = ",".join(map(str, range(test_flavor.ngpu)))

    for idx, override_arg in enumerate(test_flavor.override_args):
        cmd = " ./run_train.sh --model.name torchcomms "
        cmd = (
            f"CONFIG_FILE={full_path} NGPU={test_flavor.ngpu} LOG_RANK={all_ranks}"
            + cmd
        )
        # dump compile trace for debugging purpose
        cmd = f'TORCH_TRACE="{output_dir}/{test_name}/compile_trace" ' + cmd

        cmd += " " + dump_folder_arg
        if override_arg:
            cmd += " " + " ".join(override_arg)

        result = _run_cmd(cmd)
        if result.returncode != 0:
            raise Exception(
                f"Integration test failed, flavor : {test_flavor.test_descr}, command : {cmd}"
            )


def run_tests(args, test_list: list[OverrideDefinitions]):
    """Run all integration tests to test the core features of TorchTitan"""

    # Check if config file exists
    assert args.config_path.endswith(".toml"), "Base config path must end with .toml"
    assert os.path.exists(
        args.config_path
    ), f"Base config path {args.config_path} does not exist"

    for test_flavor in test_list:
        # Filter by test_name if specified
        if args.test_name != "all" and test_flavor.test_name != args.test_name:
            continue

        # Check if we have enough GPUs
        print("run single test")
        run_single_test(test_flavor, args.config_path, args.output_dir)


def build_integration_tests_list() -> list[OverrideDefinitions]:
    """
    key is the config file name and value is a list of OverrideDefinitions
    that is used to generate variations of integration tests based on the
    same root config file.
    """
    integration_tests_flavors = [
        OverrideDefinitions(
            [
                [],
            ],
            "FSDP",
            "FSDP",
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "FSDP+TP",
            "FSDP+TP",
        ),
    ]
    return integration_tests_flavors


_TEST_SUITES_FUNCTION = {
    "torchcomms_integration": build_integration_tests_list,
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

    test_list = _TEST_SUITES_FUNCTION["torchcomms_integration"]()
    run_tests(args, test_list)


if __name__ == "__main__":
    main()
