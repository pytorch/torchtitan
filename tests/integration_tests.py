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
                    "--profiling.enable_profiling",
                    "--metrics.enable_tensorboard",
                ],
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
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "2D eager",
            "2d_eager",
        ),
        OverrideDefinitions(
            [
                [
                    "--training.compile",
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "2D compile",
            "2d_compile",
        ),
        # TODO: re-enable this test once the async TP issue is fixed
        # OverrideDefinitions(
        #     [
        #         [
        #             "--training.compile",
        #             "--parallelism.tensor_parallel_degree 2",
        #             "--parallelism.enable_async_tensor_parallel",
        #         ],
        #     ],
        #     "2D async TP compile",
        #     "2d_asynctp_compile",
        # ),
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
        # TODO: enable the following tests once they are fixed
        # OverrideDefinitions(
        #     [
        #         [
        #             "--parallelism.pipeline_parallel_degree 4",
        #             "--parallelism.pipeline_parallel_schedule InterleavedZeroBubble",
        #         ],
        #     ],
        #     "PP looped zero bubble test",
        #     "pp_looped_zero_bubble",
        #     ngpu=4,
        # ),
        # OverrideDefinitions(
        #     [
        #         [
        #             "--parallelism.pipeline_parallel_degree 2",
        #             "--parallelism.pipeline_parallel_schedule ZBVZeroBubble",
        #         ],
        #     ],
        #     "PP zero bubble test (v shaped)",
        #     "pp_zbv",
        #     ngpu=2,
        # ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule 1F1B",
                    "--parallelism.data_parallel_shard_degree 1",
                ],
            ],
            "PP 1D test 1F1B",
            "pp_1f1b",
            ngpu=2,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule GPipe",
                    "--parallelism.data_parallel_shard_degree 1",
                ],
            ],
            "PP 1D test GPipe",
            "pp_gpipe",
            ngpu=2,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule 1F1B",
                    "--parallelism.data_parallel_shard_degree 2",
                ],
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule 1F1B",
                    "--parallelism.pipeline_parallel_layers_per_stage 4",
                    "--parallelism.data_parallel_shard_degree 2",
                ],
            ],
            "PP+DP 1F1B 2D test",
            "pp_dp_1f1b",
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule GPipe",
                    "--parallelism.data_parallel_shard_degree 2",
                ],
            ],
            "PP+DP GPipe 2D test",
            "pp_dp_gpipe",
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "PP+TP 2D test",
            "pp_tp",
        ),
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable_checkpoint",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                ],
                [
                    "--training.steps 20",
                    "--checkpoint.enable_checkpoint",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "PP+DP+TP 3D test with save/load resume ckpt",
            "pp_dp_tp",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--training.compile",
                ],
            ],
            "PP+DP+TP 3D test with torch.compile",
            "3d_compile",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.pipeline_parallel_degree 4",
                    "--parallelism.pipeline_parallel_schedule Interleaved1F1B",
                ],
                [
                    "--parallelism.pipeline_parallel_degree 4",
                    "--parallelism.pipeline_parallel_schedule Interleaved1F1B",
                    "--parallelism.pipeline_parallel_layers_per_stage 1",
                ],
            ],
            "PP looped 1F1B test",
            "pp_looped_1f1b",
            ngpu=4,
        ),
        # OverrideDefinitions(
        #     [
        #         [
        #             "--parallelism.pipeline_parallel_degree 2",
        #             "--parallelism.pipeline_parallel_schedule PipelineScheduleMulti",
        #             "--parallelism.pipeline_parallel_schedule_csv ./tests/assets/custom_schedule.csv",
        #         ],
        #     ],
        #     "PP with custom pipeline schedule loaded from CSV file",
        #     "pp_custom_csv",
        #     ngpu=2,
        # ),
        OverrideDefinitions(
            [
                [
                    "--optimizer.name AdamW --optimizer.implementation foreach",
                ]
            ],
            "Foreach Optimizer Test",
            "optimizer_foreach",
            ngpu=2,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.data_parallel_shard_degree=1",
                    "--parallelism.data_parallel_replicate_degree=4",
                ]
            ],
            "DDP",
            "ddp",
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
        OverrideDefinitions(
            [
                [
                    "--parallelism.data_parallel_shard_degree=4",
                    "--activation_checkpoint.mode='full'",
                    "--model.flavor=debugmodel_flex_attn",
                ]
            ],
            "FSDP+FLEX_ATTN",
            "fsdp+flex_attn",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.context_parallel_degree=4",
                    "--parallelism.context_parallel_rotate_method='allgather'",
                ]
            ],
            "CP (allgather)",
            "cp_allgather",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.context_parallel_degree=4",
                    "--parallelism.context_parallel_rotate_method='alltoall'",
                ]
            ],
            "CP (alltoall)",
            "cp_alltoall",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.data_parallel_shard_degree=2",
                    "--parallelism.data_parallel_replicate_degree=2",
                    "--parallelism.tensor_parallel_degree=2",
                ]
            ],
            "HSDP+TP",
            "hsdp+tp",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.data_parallel_shard_degree=2",
                    "--parallelism.context_parallel_degree=2",
                ]
            ],
            "FSDP+CP",
            "fsdp+cp",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.data_parallel_shard_degree=1",
                    "--parallelism.data_parallel_replicate_degree=2",
                    "--parallelism.context_parallel_degree=2",
                ]
            ],
            "HSDP+CP (with dp_shard)",
            "hsdp+cp_without_dp_shard",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.data_parallel_shard_degree=2",
                    "--parallelism.data_parallel_replicate_degree=2",
                    "--parallelism.context_parallel_degree=2",
                ]
            ],
            "HSDP+CP (without dp_shard)",
            "hsdp+cp_with_dp_shard",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.data_parallel_shard_degree=2",
                    "--parallelism.tensor_parallel_degree=2",
                    "--parallelism.context_parallel_degree=2",
                ]
            ],
            "FSDP+TP+CP",
            "fsdp+tp+cp",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable_checkpoint",
                    "--parallelism.tensor_parallel_degree=2",
                    "--parallelism.context_parallel_degree=2",
                    "--training.enable_cpu_offload",
                    "--optimizer.early_step_in_backward",
                ],
                [
                    "--parallelism.tensor_parallel_degree=2",
                    "--parallelism.context_parallel_degree=2",
                    "--parallelism.data_parallel_replicate_degree=2",
                    "--training.enable_cpu_offload",
                    "--optimizer.early_step_in_backward",
                ],
            ],
            "Enable CPU Offload, Optimizer in backward with TP, DP, CP",
            "cpu_offload+opt_in_bwd+TP+DP+CP",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--memory_estimation.enabled",
                ]
            ],
            "FSDP2 Memory Tracking and Estimation",
            "fsdp2_memory_estimation",
            ngpu=2,
        ),
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable_checkpoint",
                ],
                [
                    # placeholder for the generation script's generate step
                ],
            ],
            "Generation script test",
            "test_generate",
            ngpu=2,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.fsdp_reshard_after_forward always",
                ],
            ],
            "Test always resharding after forward pass",
            "fsdp_reshard_always",
            ngpu=2,
        ),
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable_checkpoint",
                    "--training.steps 10",
                ],
                # Save at [dp:4] and load at [dp:2, tp:2]. Note that the dataloader should be
                # excluded during loading to avoid errors caused by mismatched dp_degree.
                [
                    "--checkpoint.enable_checkpoint",
                    "--checkpoint.exclude_from_loading lr_scheduler,dataloader,optimizer",
                    "--parallelism.tensor_parallel_degree 2",
                    "--training.steps 20",
                ],
            ],
            "Optional checkpoint",
            "optional_checkpoint",
        ),
    ]
    return integration_tests_flavors


def _run_cmd(cmd):
    return subprocess.run([cmd], text=True, shell=True)


def run_test(test_flavor: OverrideDefinitions, full_path: str, output_dir: str):
    # run_test supports sequence of tests.
    test_name = test_flavor.test_name
    dump_folder_arg = f"--job.dump_folder {output_dir}/{test_name}"
    model_flavor_arg = f"--model.flavor {test_flavor.model_flavor}"
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
        cmd += " " + model_flavor_arg
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
