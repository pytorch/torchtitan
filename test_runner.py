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
    test_name: str = "default"
    requires_seed_checkpoint: bool = False
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
                [],
            ],
            "default",
            "default",
        ),
        OverrideDefinitions(
            [
                [
                    "--training.compile",
                ],
            ],
            "1D compile",
            "1d_compile",
        ),
        OverrideDefinitions(
            [
                [
                    "--training.compile",
                    "--activation_checkpoint.mode selective",
                    "--activation_checkpoint.selective_ac_option op",
                ],
            ],
            "1D compile with selective op AC",
            "1d_compile_sac_op",
        ),
        OverrideDefinitions(
            [
                [
                    "--training.tensor_parallel_degree 2",
                ],
            ],
            "2D eager",
            "2d_eager",
        ),
        OverrideDefinitions(
            [
                [
                    "--training.compile",
                    "--training.tensor_parallel_degree 2",
                ],
            ],
            "2D compile",
            "2d_compile",
        ),
        OverrideDefinitions(
            [
                [
                    "--training.tensor_parallel_degree 2",
                    "--model.norm_type=fused_rmsnorm",
                ],
            ],
            "2D eager with fused_rmsnorm",
            "2d_eager_fused_rmsnorm",
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
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable_checkpoint",
                    "--experimental.pipeline_parallel_degree 4",
                    "--experimental.pipeline_parallel_schedule InterleavedZeroBubble",
                ],
            ],
            "PP looped zero bubble test",
            "pp_looped_zero_bubble",
            requires_seed_checkpoint=True,
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable_checkpoint",
                    "--experimental.pipeline_parallel_degree 2",
                    "--experimental.pipeline_parallel_schedule 1F1B",
                    "--training.data_parallel_shard_degree 1",
                ],
            ],
            "PP 1D test 1F1B",
            "pp_1f1b",
            requires_seed_checkpoint=True,
            ngpu=2,
        ),
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable_checkpoint",
                    "--experimental.pipeline_parallel_degree 2",
                    "--experimental.pipeline_parallel_schedule GPipe",
                    "--training.data_parallel_shard_degree 1",
                ],
            ],
            "PP 1D test GPipe",
            "pp_gpipe",
            requires_seed_checkpoint=True,
            ngpu=2,
        ),
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable_checkpoint",
                    "--experimental.pipeline_parallel_degree 2",
                    "--experimental.pipeline_parallel_schedule 1F1B",
                    "--training.data_parallel_shard_degree 2",
                ],
            ],
            "PP+DP 1F1B 2D test",
            "pp_dp_1f1b",
            requires_seed_checkpoint=True,
        ),
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable_checkpoint",
                    "--experimental.pipeline_parallel_degree 2",
                    "--experimental.pipeline_parallel_schedule GPipe",
                    "--training.data_parallel_shard_degree 2",
                ],
            ],
            "PP+DP GPipe 2D test",
            "pp_dp_gpipe",
            requires_seed_checkpoint=True,
        ),
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable_checkpoint",
                    "--experimental.pipeline_parallel_degree 2",
                    "--training.tensor_parallel_degree 2",
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
                    "--training.data_parallel_shard_degree 2",
                    "--training.tensor_parallel_degree 2",
                ],
                [
                    "--training.steps 20",
                    "--checkpoint.enable_checkpoint",
                    "--experimental.pipeline_parallel_degree 2",
                    "--training.data_parallel_shard_degree 2",
                    "--training.tensor_parallel_degree 2",
                ],
            ],
            "PP+DP+TP 3D test with save/load resume ckpt",
            "pp_dp_tp",
            requires_seed_checkpoint=True,
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--experimental.pipeline_parallel_degree 2",
                    "--training.data_parallel_shard_degree 2",
                    "--training.tensor_parallel_degree 2",
                    "--training.compile",
                ],
            ],
            "PP+DP+TP 3D test with torch.compile",
            "3d_compile",
            requires_seed_checkpoint=True,
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable_checkpoint",
                    "--experimental.pipeline_parallel_degree 4",
                    "--experimental.pipeline_parallel_schedule Interleaved1F1B",
                ],
            ],
            "PP looped 1F1B test",
            "pp_looped_1f1b",
            requires_seed_checkpoint=True,
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--optimizer.name Adam --optimizer.fused",
                    "--optimizer.name AdamW --optimizer.fused",
                ]
            ],
            "Fused Optimizer Test",
        ),
        OverrideDefinitions(
            [
                [
                    "--training.data_parallel_shard_degree=1",
                    "--training.data_parallel_replicate_degree=4",
                ]
            ],
            "DDP",
            "ddp",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--training.data_parallel_shard_degree=2",
                    "--training.data_parallel_replicate_degree=2",
                ]
            ],
            "HSDP",
            "hsdp",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--training.data_parallel_shard_degree=2",
                    "--training.data_parallel_replicate_degree=2",
                    "--training.tensor_parallel_degree=2",
                ]
            ],
            "HSDP+TP",
            "hsdp+tp",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--training.data_parallel_shard_degree=2",
                    "--experimental.context_parallel_degree=2",
                ]
            ],
            "FSDP+CP",
            "fsdp+cp",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--training.data_parallel_shard_degree=2",
                    "--training.data_parallel_replicate_degree=2",
                    "--experimental.context_parallel_degree=2",
                ]
            ],
            "HSDP+CP",
            "hsdp+cp",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--training.data_parallel_shard_degree=2",
                    "--training.tensor_parallel_degree=2",
                    "--experimental.context_parallel_degree=2",
                ]
            ],
            "FSDP+TP+CP",
            "fsdp+tp+cp",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--memory_estimation.enabled",
                ]
            ],
            "FSDP2 Memory Tracking and Estimation",
            "fsdp2_mem_tracker",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--experimental.pipeline_parallel_degree 2",
                    "--training.enable_cpu_offload True",
                ],
            ],
            "Enable CPU Offload with PP",
            "enable_cpu_offload+PP",
            ngpu=4,
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
    test_name = test_flavor.test_name
    dump_folder_arg = f"--job.dump_folder {output_dir}/{test_name}"
    model_flavor_arg = f"--model.flavor {test_flavor.model_flavor}"
    all_ranks = ",".join(map(str, range(test_flavor.ngpu)))

    if test_flavor.requires_seed_checkpoint:
        cmd = f"CONFIG_FILE={full_path} ./create_seed_checkpoint.sh {dump_folder_arg} {model_flavor_arg}"
        logger.info(
            f"=====Integration test, flavor : {test_flavor.test_descr}, command : {cmd}====="
        )
        result = _run_cmd(cmd)
        logger.info(result.stdout)

    for override_arg in test_flavor.override_args:
        cmd = f"CONFIG_FILE={full_path} NGPU={test_flavor.ngpu} LOG_RANK={all_ranks} ./run_llama_train.sh"
        if test_name == "fsdp2_mem_tracker":
            cmd = f"CONFIG_FILE={full_path} NGPU={test_flavor.ngpu} LOG_RANK={all_ranks} ./run_memory_estimation.sh"
        cmd += " " + dump_folder_arg
        cmd += " " + model_flavor_arg
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
                            elif args.ngpu == 8 and test_flavor.ngpu != 8:
                                logger.info(
                                    f"Skipping non-8gpu test {test_flavor.test_name} on 8-gpu runner"
                                )
                            else:
                                run_test(test_flavor, full_path, args.output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--config_dir", default="./train_configs")
    parser.add_argument(
        "--test",
        default="all",
        help="test to run, acceptable values: `test_name` in `build_test_list` (default: all)",
    )
    parser.add_argument("--ngpu", default=4, type=int)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.listdir(args.output_dir):
        raise RuntimeError("Please provide an empty output directory.")
    run_tests(args)


if __name__ == "__main__":
    main()
