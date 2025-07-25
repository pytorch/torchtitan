# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import concurrent.futures
import logging
import os
import subprocess

from .integration_tests import TestCaseConfigs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_test_list():
    """
    key is the config file name and value is a list of TestCaseConfigs
    that is used to generate variations of integration tests based on the
    same root config file.
    """
    integration_tests_flavors = []
    integration_tests_flavors.extend([
        TestCaseConfigs(
            [
                ["--training.steps 10", "--checkpoint.enable"],
            ],
            "Default TorchFT integration test",
            "default_torchft",
            ngpu=8,
        )
    ])
    return integration_tests_flavors


def _run_cmd(cmd):
    return subprocess.run([cmd], text=True, shell=True)


def run_single_test(test_flavor: TestCaseConfigs, model_name: str, full_path: str, output_dir: str):
    # run_test supports sequence of tests.
    test_name = test_flavor.test_name
    dump_folder_arg = f"--job.dump_folder {output_dir}/{test_name}"
    model_name_arg = f"--model.name {model_name}"

    # Use all 8 GPUs in a single replica
    # TODO: Use two replica groups
    # Right now when passing CUDA_VISIBLE_DEVICES=0,1,2,3 and 4,5,6,7 for 2 RGs I get
    # Cuda failure 217 'peer access is not supported between these two devices'
    all_ranks = [",".join(map(str, range(0, 8)))]

    for test_idx, override_arg in enumerate(test_flavor.override_args):
        cmds = []

        for replica_id, ranks in enumerate(all_ranks):
            cmd = (
                f'TORCH_TRACE="{output_dir}/{test_name}/compile_trace" '
                + f"CUDA_VISIBLE_DEVICES={ranks} "
                + f"CONFIG_FILE={full_path} NGPU={test_flavor.ngpu} ./run_train.sh "
                + "--fault_tolerance.enable "
                + f"--fault_tolerance.replica_id={replica_id} --fault_tolerance.group_size={test_flavor.ngpu}"
            )

            cmd += " " + dump_folder_arg
            cmd += " " + model_name_arg
            if override_arg:
                cmd += " " + " ".join(override_arg)

            logger.info(
                "=====TorchFT Integration test, flavor : "
                f"{test_flavor.test_descr}, command : {cmd}====="
            )
            cmds.append((replica_id, cmd))

        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(_run_cmd, cmd) for _, cmd in cmds]
            results = [future.result() for future in futures]

        for i, result in enumerate(results):
            logger.info(result.stdout)

            if result.returncode == 0:
                continue

            raise Exception(
                f"Integration test {test_idx} failed, flavor : {test_flavor.test_descr}, command : {cmds[i]}"
            )


def run_tests(args):
    integration_tests_flavors = build_test_list()

    if args.ngpu < 8:
        logger.info("Skipping TorchFT integration tests as we need 8 GPUs.")
        return
    
    for test_flavor in integration_tests_flavors:
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
    parser.add_argument("--ngpu", default=8, type=int)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    run_tests(args)


if __name__ == "__main__":
    main()
