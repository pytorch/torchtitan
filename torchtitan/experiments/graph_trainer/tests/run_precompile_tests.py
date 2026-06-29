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
    disabled: bool = False


def _run_cmd(cmd):
    return subprocess.run([cmd], text=True, shell=True)


def _build_precompile_tests() -> list[PrecompileTestDefinition]:
    fx_trace_precompile_dir = tempfile.mkdtemp(prefix="fx_trace_precompile_")
    dsv3_fx_trace_precompile_dir = tempfile.mkdtemp(prefix="dsv3_fx_trace_precompile_")
    return [
        # Uses the SDPA backend: the default FlexAttention backend bakes a
        # BlockMask into the precompiled artifact, whose mask_mod closures are
        # Python code objects that pickle.dumps cannot serialize ("TypeError:
        # cannot pickle code objects" in precompile_fx_trace_save). SDPA carries
        # no such object, so it exercises the precompile machinery cleanly.
        # TODO: re-test on FlexAttention once BlockMask is excluded/rebuilt at
        # load time (or becomes picklable).
        PrecompileTestDefinition(
            precompile_command=(
                "python -m torchtitan.experiments.graph_trainer.precompile_main"
                " --module graph_trainer.llama3"
                " --config graph_trainer_llama3_debugmodel_sdpa"
                " --compile.mode aot_fx_trace"
                f" --compile.precompile_artifact_dir {fx_trace_precompile_dir}"
                " --parallelism.data_parallel_shard_degree 4"
                " --parallelism.tensor_parallel_degree 2"
            ),
            override_args=[
                "--module graph_trainer.llama3",
                "--config graph_trainer_llama3_debugmodel_sdpa",
                "--compile.mode aot_fx_trace",
                f"--compile.precompile_artifact_dir {fx_trace_precompile_dir}",
                "--parallelism.data_parallel_shard_degree 4",
                "--parallelism.tensor_parallel_degree 2",
            ],
            test_descr="aot_fx_trace llama3 precompile FSDP+TP",
            test_name="aot_fx_trace_llama3_precompile_fsdp_tp",
            ngpu=8,
        ),
        # TODO: disabled — precompile sharding propagation fails on aten.view
        # with a data-dependent unbacked symint ("Could not extract specialized
        # integer from u13") for DSv3 MoE. Separate from the empty_strided
        # shadow-node fix; re-enable once the precompile symint issue is fixed.
        PrecompileTestDefinition(
            precompile_command=(
                "python -m torchtitan.experiments.graph_trainer.precompile_main"
                " --module graph_trainer.deepseek_v3"
                " --config graph_trainer_deepseek_v3_debugmodel"
                " --compile.mode aot_fx_trace"
                f" --compile.precompile_artifact_dir {dsv3_fx_trace_precompile_dir}"
                " --parallelism.data_parallel_shard_degree 4"
                " --parallelism.tensor_parallel_degree 2"
                " --parallelism.expert_parallel_degree 4"
            ),
            override_args=[
                "--module graph_trainer.deepseek_v3",
                "--config graph_trainer_deepseek_v3_debugmodel",
                "--compile.mode aot_fx_trace",
                f"--compile.precompile_artifact_dir {dsv3_fx_trace_precompile_dir}",
                "--parallelism.data_parallel_shard_degree 4",
                "--parallelism.tensor_parallel_degree 2",
                "--parallelism.expert_parallel_degree 4",
            ],
            test_descr="aot_fx_trace deepseek_v3 precompile FSDP+TP+EP",
            test_name="aot_fx_trace_deepseek_v3_precompile_fsdp_tp_ep",
            ngpu=8,
            disabled=True,
        ),
        # GraphPP precompile: save every stage on one process, load per PP rank.
        # Validated bitwise-identical to live-trace for both supported V-shaped
        # schedules. ep=1: EP precompile is rejected because the MoE all-to-all
        # bakes an opaque ProcessGroup constant the GraphPickler cannot serialize.
        # selective_activation_remat is disabled because CooR fragments the
        # backward into many regions that pass rejects; remat is numerically
        # neutral, so disabling it does not change the loss. SDPA backend: the
        # FlexAttention default bakes an unpicklable BlockMask closure.
        *_graph_pp_precompile_tests(),
    ]


def _graph_pp_precompile_tests() -> list[PrecompileTestDefinition]:
    tests = []
    for schedule, slug in (
        ("Interleaved1F1B", "interleaved_1f1b"),
        ("ZBVZeroBubble", "zbv_zero_bubble"),
    ):
        artifact_dir = tempfile.mkdtemp(prefix=f"gpp_fx_trace_precompile_{slug}_")
        args = [
            "--module graph_trainer.deepseek_v3",
            "--config graph_trainer_deepseek_v3_debugmodel_sdpa",
            "--compile.mode aot_fx_trace",
            f"--compile.precompile_artifact_dir {artifact_dir}",
            "--compile.disable_passes selective_activation_remat_pass",
            "--parallelism.pipeline_parallel_degree 2",
            f"--parallelism.pipeline_parallel_schedule {schedule}",
            "--parallelism.data_parallel_shard_degree 4",
            "--parallelism.expert_parallel_degree 1",
        ]
        tests.append(
            PrecompileTestDefinition(
                precompile_command=(
                    "python -m torchtitan.experiments.graph_trainer.precompile_main "
                    + " ".join(args)
                ),
                override_args=args,
                test_descr=f"aot_fx_trace deepseek_v3 GraphPP precompile {schedule}",
                test_name=f"aot_fx_trace_deepseek_v3_graph_pp_precompile_{slug}",
                ngpu=8,
            )
        )
    return tests


RUN_TRAIN_SCRIPT = "torchtitan/experiments/graph_trainer/run_train_precompile.sh"


def run_precompile_tests(args):
    test_list = _build_precompile_tests()

    # A --test_name that matches no known test (a typo, or CI drifting from a
    # renamed slug) must fail loudly instead of silently running nothing and
    # exiting 0, which would mask zero coverage as a green CI run.
    known_names = {t.test_name for t in test_list}
    if args.test_name != "all" and args.test_name not in known_names:
        raise SystemExit(
            f"--test_name {args.test_name!r} matched no precompile test. "
            f"Available test names: {sorted(known_names)}"
        )

    ran_any = False
    for test in test_list:
        if args.test_name != "all" and test.test_name != args.test_name:
            continue
        if test.disabled:
            logger.info(f"Skipping disabled test: {test.test_name}")
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
        # _run_cmd inherits stdout/stderr so child output streams straight to
        # the CI log; result.stdout is None and is not logged here.
        result = _run_cmd(test.precompile_command)
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
