# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Separate test runner for CooR precompile integration tests.

Each test has two steps:
1. Run precompile_main.py on a single process to generate a rank-agnostic
   compiled artifact.
2. Run training via graph_trainer/run_train_precompile.sh (which passes
   --virtual-local-rank to torchrun) to load and train with the artifact.

Usage:
    python -m torchtitan.experiments.graph_trainer.tests.run_precompile_tests \
        output_dir --ngpu 8
"""

import argparse
import os
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass

from torchtitan.tools.logging import logger


@dataclass
class PrecompileTestDefinition:
    precompile_command: str
    override_args: Sequence[str]
    test_descr: str
    test_name: str
    ngpu: int = 8


def _run_cmd(cmd):
    return subprocess.run([cmd], text=True, shell=True)


def _build_precompile_tests() -> list[PrecompileTestDefinition]:
    full_inductor_precompile_dir = tempfile.mkdtemp(prefix="precompile_")
    regional_precompile_dir = tempfile.mkdtemp(prefix="precompile_regional_")
    fx_trace_precompile_dir = tempfile.mkdtemp(prefix="fx_trace_precompile_")
    return [
        PrecompileTestDefinition(
            precompile_command=(
                "python -m torchtitan.experiments.graph_trainer.precompile_main"
                " --module graph_trainer.llama3"
                " --config graph_trainer_llama3_debugmodel"
                " --compile.mode aot"
                " --compile.passes full_inductor_compilation"
                " --compile.joint_passes inductor_decomposition"
                f" --compile.precompile_artifact_dir {full_inductor_precompile_dir}"
                " --parallelism.data_parallel_shard_degree 4"
                " --parallelism.tensor_parallel_degree 2"
            ),
            override_args=[
                "--module graph_trainer.llama3",
                "--config graph_trainer_llama3_debugmodel",
                "--compile.mode aot",
                "--compile.passes full_inductor_compilation",
                "--compile.joint_passes inductor_decomposition",
                f"--compile.precompile_artifact_dir {full_inductor_precompile_dir}",
                "--parallelism.data_parallel_shard_degree 4",
                "--parallelism.tensor_parallel_degree 2",
            ],
            test_descr="AOT llama3 precompile full_inductor_compilation",
            test_name="aot_llama3_precompile_full_inductor",
            ngpu=8,
        ),
        PrecompileTestDefinition(
            precompile_command=(
                "python -m torchtitan.experiments.graph_trainer.precompile_main"
                " --module graph_trainer.llama3"
                " --config graph_trainer_llama3_debugmodel_flex_attn"
                " --compile.mode aot"
                " --compile.passes regional_inductor"
                f" --compile.precompile_artifact_dir {regional_precompile_dir}"
                " --parallelism.data_parallel_shard_degree 4"
                " --parallelism.tensor_parallel_degree 2"
            ),
            override_args=[
                "--module graph_trainer.llama3",
                "--config graph_trainer_llama3_debugmodel_flex_attn",
                "--compile.mode aot",
                "--compile.passes regional_inductor",
                f"--compile.precompile_artifact_dir {regional_precompile_dir}",
                "--parallelism.data_parallel_shard_degree 4",
                "--parallelism.tensor_parallel_degree 2",
            ],
            test_descr="AOT llama3 precompile regional_inductor (flex_attn)",
            test_name="aot_llama3_precompile_regional_inductor",
            ngpu=8,
        ),
        PrecompileTestDefinition(
            precompile_command=(
                "python -m torchtitan.experiments.graph_trainer.precompile_main"
                " --module graph_trainer.llama3"
                " --config graph_trainer_llama3_debugmodel"
                " --compile.mode aot_fx_trace"
                f" --compile.precompile_artifact_dir {fx_trace_precompile_dir}"
                " --parallelism.data_parallel_shard_degree 4"
                " --parallelism.tensor_parallel_degree 2"
            ),
            override_args=[
                "--module graph_trainer.llama3",
                "--config graph_trainer_llama3_debugmodel",
                "--compile.mode aot_fx_trace",
                f"--compile.precompile_artifact_dir {fx_trace_precompile_dir}",
                "--parallelism.data_parallel_shard_degree 4",
                "--parallelism.tensor_parallel_degree 2",
            ],
            test_descr="aot_fx_trace llama3 precompile FSDP+TP",
            test_name="aot_fx_trace_llama3_precompile_fsdp_tp",
            ngpu=8,
        ),
    ]


RUN_TRAIN_SCRIPT = "torchtitan/experiments/graph_trainer/run_train_precompile.sh"


def run_precompile_tests(args):
    test_list = _build_precompile_tests()

    ran_any = False
    for test in test_list:
        if args.test_name != "all" and test.test_name != args.test_name:
            continue
        if args.ngpu < test.ngpu:
            logger.info(
                f"Skipping test {test.test_name} that requires {test.ngpu} gpus,"
                f" because --ngpu arg is {args.ngpu}"
            )
            continue

        ran_any = True
        all_ranks = ",".join(map(str, range(test.ngpu)))
        dump_folder_arg = f"--dump_folder {args.output_dir}/{test.test_name}"

        # Step 1: precompile
        logger.info(
            f"===== {time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"Precompile step for {test.test_descr}: "
            f"{test.precompile_command} ====="
        )
        result = _run_cmd(test.precompile_command)
        logger.info(result.stdout)
        if result.returncode != 0:
            raise Exception(
                f"Precompile step failed for: {test.test_descr}, "
                f"command: {test.precompile_command}"
            )

        # Step 2: training with the precompiled artifact
        cmd = f"NGPU={test.ngpu} LOG_RANK={all_ranks} " f"./{RUN_TRAIN_SCRIPT}"
        cmd = f'TORCH_TRACE="{args.output_dir}/{test.test_name}/compile_trace" ' + cmd
        cmd += " " + dump_folder_arg
        cmd += " " + " ".join(test.override_args)

        logger.info(
            f"===== {time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"Training step for {test.test_descr}: {cmd} ====="
        )
        result = _run_cmd(cmd)
        logger.info(result.stdout)
        if result.returncode != 0:
            raise Exception(
                f"Training step failed for: {test.test_descr}, command: {cmd}"
            )

    if not ran_any:
        available = [t.test_name for t in test_list]
        logger.warning(
            f"No precompile tests were run for --test_name '{args.test_name}'.\n"
            f"Available test names: {available}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument(
        "--test_name",
        default="all",
        help="Specific test name to run (default: all)",
    )
    parser.add_argument("--ngpu", default=8, type=int)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.listdir(args.output_dir):
        raise RuntimeError("Please provide an empty output directory.")

    run_precompile_tests(args)


if __name__ == "__main__":
    main()
