# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import subprocess

from tests.integration_tests.integration_tests import TestCaseConfigs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_test_list():
    """
    key is the config file name and value is a list of TestCaseConfigs
    that is used to generate variations of integration tests based on the
    same root config file.
    """
    integration_tests_flavors = []
    integration_tests_flavors.extend(
        [
            # basic tests
            TestCaseConfigs(
                [
                    [
                        "--profiling.enable_profiling",
                        "--metrics.enable_tensorboard",
                    ],
                ],
                "default",
                "default",
                supported_models=["flux"],
            ),
            # Checkpointing tests.
            TestCaseConfigs(
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
                supported_models=["flux"],
            ),
            TestCaseConfigs(
                [
                    [
                        "--checkpoint.enable_checkpoint",
                        "--checkpoint.last_save_model_only",
                    ],
                ],
                "Checkpoint Integration Test - Save Model Only fp32",
                "last_save_model_only_fp32",
                supported_models=["flux"],
            ),
            # Parallelism tests.
            TestCaseConfigs(
                [
                    [
                        "--parallelism.data_parallel_shard_degree=4",
                        "--parallelism.data_parallel_replicate_degree=1",
                    ]
                ],
                "FSDP",
                "fsdp",
                ngpu=4,
                supported_models=["flux"],
            ),
            TestCaseConfigs(
                [
                    [
                        "--parallelism.data_parallel_shard_degree=2",
                        "--parallelism.data_parallel_replicate_degree=2",
                    ]
                ],
                "HSDP",
                "hsdp",
                ngpu=4,
                supported_models=["flux"],
            ),
        ]
    )
    return integration_tests_flavors


def _run_cmd(cmd):
    return subprocess.run([cmd], text=True, shell=True)


def run_single_test(
    test_flavor: TestCaseConfigs, model_name: str, full_path: str, output_dir: str
):
    assert model_name == "flux", "Only support flux model for FLUX integration tests"
    # run_test supports sequence of tests.
    test_name = test_flavor.test_name
    dump_folder_arg = f"--job.dump_folder {output_dir}/{test_name}"
    model_name_arg = f"--model.name {model_name}"

    # Random init encoder for offline testing
    random_init_encoder_arg = "--training.test_mode"
    clip_encoder_version_arg = "--encoder.clip_encoder torchtitan/experiments/flux/tests/assets/clip-vit-large-patch14/"
    t5_encoder_version_arg = (
        "--encoder.t5_encoder torchtitan/experiments/flux/tests/assets/t5-v1_1-xxl/"
    )
    hf_assets_path_arg = "--model.hf_assets_path tests/assets/tokenizer"

    all_ranks = ",".join(map(str, range(test_flavor.ngpu)))

    for idx, override_arg in enumerate(test_flavor.override_args):
        cmd = f"CONFIG_FILE={full_path} NGPU={test_flavor.ngpu} LOG_RANK={all_ranks} ./torchtitan/experiments/flux/run_train.sh"
        # dump compile trace for debugging purpose
        cmd = f'TORCH_TRACE="{output_dir}/{test_name}/compile_trace" ' + cmd
        cmd += " " + model_name_arg
        cmd += " " + dump_folder_arg
        cmd += " " + random_init_encoder_arg
        cmd += " " + clip_encoder_version_arg
        cmd += " " + t5_encoder_version_arg
        cmd += " " + hf_assets_path_arg
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
    # build integration tests list
    test_list = build_test_list()

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
        default="./torchtitan/experiments/flux/train_configs/debug_model.toml",
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
    run_tests(args)


if __name__ == "__main__":
    main()
