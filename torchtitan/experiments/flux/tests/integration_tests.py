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

from tests.integration_tests import OverrideDefinitions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def build_test_list():
    """
    key is the config file name and value is a list of OverrideDefinitions
    that is used to generate variations of integration tests based on the
    same root config file.
    """
    integration_tests_flavors = defaultdict(list)
    integration_tests_flavors["debug_model.toml"] = [
        # basic tests
        OverrideDefinitions(
            [
                [
                    "--profiling.enable_profiling",
                    "--metrics.enable_tensorboard",
                ],
            ],
            "default",
            "default",
        ),
        # Checkpointing tests.
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
                    "--checkpoint.last_save_model_only",
                ],
            ],
            "Checkpoint Integration Test - Save Model Only fp32",
            "last_save_model_only_fp32",
        ),
        # Parallelism tests.
        OverrideDefinitions(
            [
                [
                    "--parallelism.data_parallel_shard_degree=4",
                    "--parallelism.data_parallel_replicate_degree=1",
                ]
            ],
            "FSDP",
            "fsdp",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.data_parallel_shard_degree=2",
                    "--parallelism.data_parallel_replicate_degree=2",
                ]
            ],
            "HSDP",
            "hsdp",
            ngpu=4,
        ),
    ]
    return integration_tests_flavors


def _run_cmd(cmd):
    return subprocess.run([cmd], text=True, shell=True)


def run_test(test_flavor: OverrideDefinitions, full_path: str, output_dir: str):
    # run_test supports sequence of tests.
    test_name = test_flavor.test_name
    dump_folder_arg = f"--job.dump_folder {output_dir}/{test_name}"

    # Random init encoder for offline testing
    random_init_encoder_arg = "--training.test_mode"
    clip_encoder_version_arg = "--encoder.clip_encoder torchtitan/experiments/flux/tests/assets/clip-vit-large-patch14/"
    t5_encoder_version_arg = (
        "--encoder.t5_encoder torchtitan/experiments/flux/tests/assets/t5-v1_1-xxl/"
    )
    tokenzier_path_arg = "--model.tokenizer_path tests/assets/tokenizer"

    all_ranks = ",".join(map(str, range(test_flavor.ngpu)))

    for idx, override_arg in enumerate(test_flavor.override_args):
        cmd = f"CONFIG_FILE={full_path} NGPU={test_flavor.ngpu} LOG_RANK={all_ranks} ./torchtitan/experiments/flux/run_train.sh"
        # dump compile trace for debugging purpose
        cmd = f'TORCH_TRACE="{output_dir}/{test_name}/compile_trace" ' + cmd
        cmd += " " + dump_folder_arg
        cmd += " " + random_init_encoder_arg
        cmd += " " + clip_encoder_version_arg
        cmd += " " + t5_encoder_version_arg
        cmd += " " + tokenzier_path_arg
        if override_arg:
            cmd += " " + " ".join(override_arg)
        logger.info(
            f"=====Flux Integration test, flavor : {test_flavor.test_descr}, command : {cmd}====="
        )

        result = _run_cmd(cmd)
        logger.info(result.stdout)
        if result.returncode != 0:
            raise Exception(
                f"Flux Integration test failed, flavor : {test_flavor.test_descr}, command : {cmd}"
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
                        if args.test == "all" or test_flavor.test_name == args.test:
                            if args.ngpu < test_flavor.ngpu:
                                logger.info(
                                    f"Skipping test {test_flavor.test_name} that requires {test_flavor.ngpu} gpus,"
                                    f" because --ngpu arg is {args.ngpu}"
                                )
                            else:
                                run_test(test_flavor, full_path, args.output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument(
        "--config_dir", default="./torchtitan/experiments/flux/train_configs"
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
    if os.listdir(args.output_dir):
        raise RuntimeError("Please provide an empty output directory.")
    run_tests(args)


if __name__ == "__main__":
    main()
