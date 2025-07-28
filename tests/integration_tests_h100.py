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
        OverrideDefinitions(
            [
                [
                    "--training.compile",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.enable_async_tensor_parallel",
                ],
            ],
            "2D async TP compile",
            "2d_asynctp_compile",
        ),
        OverrideDefinitions(
            [
                [
                    "--model.converters float8",
                    "--float8.enable_fsdp_float8_all_gather",
                    "--float8.precompute_float8_dynamic_scale_for_fsdp",
                ],
            ],
            "Float8 test",
            "float8",
        ),
        OverrideDefinitions(
            [
                [
                    "--training.compile",
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
                    "--training.compile",
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


def _run_cmd(cmd):
    return subprocess.run([cmd], text=True, shell=True)


def run_test(test_flavor: OverrideDefinitions, full_path: str, output_dir: str):
    # run_test supports sequence of tests.
    test_name = test_flavor.test_name
    dump_folder_arg = f"--job.dump_folder {output_dir}/{test_name}"
    all_ranks = ",".join(map(str, range(test_flavor.ngpu)))

    for idx, override_arg in enumerate(test_flavor.override_args):
        cmd = f"CONFIG_FILE={full_path} NGPU={test_flavor.ngpu} LOG_RANK={all_ranks} ./run_train.sh"
        # dump compile trace for debugging purpose
        cmd = f'TORCH_TRACE="{output_dir}/{test_name}/compile_trace" ' + cmd
        if test_name == "fsdp2_memory_estimation":
            cmd = (
                f"CONFIG_FILE={full_path} NGPU={test_flavor.ngpu} LOG_RANK={all_ranks} "
                "./scripts/estimate/run_memory_estimation.sh"
            )
        cmd += " " + dump_folder_arg
        if override_arg:
            cmd += " " + " ".join(override_arg)
        logger.info(
            f"=====Integration test, flavor : {test_flavor.test_descr}, command : {cmd}====="
        )

        # save checkpoint (idx == 0) and load it for generation (idx == 1)
        if test_name == "test_generate" and idx == 1:
            cmd = (
                f"CONFIG_FILE={full_path} NGPU={test_flavor.ngpu} LOG_RANK={all_ranks} "
                f"CHECKPOINT_DIR={output_dir}/{test_name}/checkpoint/step-10 "
                "PROMPT='What is the meaning of life?' "
                f"./scripts/generate/run_llama_generate.sh --out > {output_dir}/{test_name}/generated_output.json"
            )

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
        "--config_dir", default="./torchtitan/models/llama3/train_configs"
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
