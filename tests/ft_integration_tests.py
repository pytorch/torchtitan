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
    test_name: str = "default"
    ngpu: int = 4
    model_flavor: str = "debugmodel"

    def __repr__(self):
        return self.test_descr


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
                    "--training.steps 100",
                ],
            ],
            "Default TorchFT integration test",
            "default_torchft",
        )
    ]
    return integration_tests_flavors


def _run_cmd(cmd):
    return subprocess.run([cmd], text=True, shell=True)


def run_test(test_flavor: OverrideDefinitions, full_path: str, output_dir: str):
    # run_test supports sequence of tests.
    test_name = test_flavor.test_name
    dump_folder_arg = f"--job.dump_folder {output_dir}/{test_name}"
    model_flavor_arg = f"--model.flavor {test_flavor.model_flavor}"

    all_ranks = [",".join(map(str, range(0, 4))), ",".join(map(str, range(4, 8)))]

    for idx, override_arg in enumerate(test_flavor.override_args):
        cmds = []
        for replica_id, ranks in enumerate(all_ranks):
            cmd = (
                f'TORCH_TRACE="{output_dir}/{test_name}/compile_trace" '
                + f"NGPU=4 CUDA_VISIBLE_DEVICES={ranks} "
                + f"CONFIG_FILE={full_path} NGPU=4 ./run_train.sh "
                + "--fault_tolerance.enable "
                + f"--fault_tolerance.replica_id={replica_id} --fault_tolerance.group_size=2"
            )

            cmd += " " + dump_folder_arg
            cmd += " " + model_flavor_arg
            if override_arg:
                cmd += " " + " ".join(override_arg)

            logger.info(
                "=====TorchFT Integration test, flavor : "
                f"{test_flavor.test_descr}, command : {cmd}====="
            )
            cmds.append(cmd)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(_run_cmd, cmd) for cmd in cmds]
            results = [future.result() for future in futures]

        for result in results:
            logger.info(result.stdout)

            if result.returncode == 0:
                continue

            raise Exception(
                f"Integration test failed, flavor : {test_flavor.test_descr}, command : {cmd}"
            )


def run_tests(args):
    integration_tests_flavors = build_test_list()

    if args.ngpu < 8:
        logger.inf(f"Skipping TorchFT integration tests as we need 8 GPUs.")
        return

    for config_file in os.listdir(args.config_dir):
        if not config_file.endswith(".toml"):
            continue

        full_path = os.path.join(args.config_dir, config_file)
        with open(full_path, "rb") as f:
            config = tomllib.load(f)
            is_integration_test = config["job"].get("use_for_integration_test", False)
            if not is_integration_test:
                continue

            for test_flavor in integration_tests_flavors[config_file]:
                if not (args.test == "all" or test_flavor.test_name == args.test):
                    continue

                run_test(test_flavor, full_path, args.output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument(
        "--config_dir", default="./torchtitan/models/llama/train_configs"
    )
    parser.add_argument(
        "--test",
        default="all",
        help="test to run, acceptable values: `test_name` in `build_test_list` (default: all)",
    )
    parser.add_argument("--ngpu", default=8, type=int)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    run_tests(args)


if __name__ == "__main__":
    main()
