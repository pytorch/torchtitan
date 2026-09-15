# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

r"""Run extended scale tests locally or through Slurm.

Example Slurm launch::

    python3 -m tests.scale_tests.run_tests outputs/scale-validation \
        --test_suite gb300_dsv3 \
        --test_name deepseek_v3_mxfp8_pp2_ep8_loss_compile \
        --launcher slurm \
        --num_gpus_per_node 4 \
        --sbatch-args="--partition=<partition> --account=<account> \
        --qos=<qos> --exclusive"
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from tests.integration_tests import OverrideDefinitions, read_golden_spec
from tests.scale_tests.gb300 import build_gb300_dsv3_tests_list


GOLDEN_DIR = Path("tests/assets/losses/gb300")
DEFAULT_NUM_GPUS_PER_NODE = 4

_TEST_SUITES_FUNCTION = {
    "gb300_dsv3": build_gb300_dsv3_tests_list,
}


def _parse_test_suites(value: str) -> tuple[str, ...]:
    suites = tuple(part.strip() for part in value.split(",") if part.strip())
    if not suites:
        raise ValueError("--test_suite must contain at least one suite")
    unknown = tuple(suite for suite in suites if suite not in _TEST_SUITES_FUNCTION)
    if unknown:
        available = ", ".join(_TEST_SUITES_FUNCTION)
        raise ValueError(
            f"Unknown test suite(s): {', '.join(unknown)}. Available: {available}"
        )
    if len(set(suites)) != len(suites):
        raise ValueError("--test_suite must not contain duplicate suites")
    return suites


def _build_slurm_launcher_command(
    test: OverrideDefinitions,
    *,
    output_dir: Path,
    slurm_script: Path,
    sbatch_args: str,
    num_gpus_per_node: int,
) -> str:
    num_nodes = test.ngpu // num_gpus_per_node
    command = [
        "sbatch",
        "--wait",
        *shlex.split(sbatch_args),
        "--export=ALL",
        f"--job-name=tt-{test.test_name}",
        f"--nodes={num_nodes}",
        "--ntasks-per-node=1",
        f"--gpus-per-node={num_gpus_per_node}",
        f"--output={output_dir / 'slurm-%j.out'}",
        str(slurm_script),
    ]
    return shlex.join(command)


def _build_loss_compare_command(
    test: OverrideDefinitions,
    *,
    output_dir: Path,
    launcher_command: str,
    export_numerics: bool,
    input_options: tuple[str, ...],
) -> list[str]:
    if test.golden_numerics_path is None:
        raise ValueError(f"{test.test_name}: scale tests require golden numerics")
    config_fn = test.configs[0]
    golden_path = GOLDEN_DIR / test.golden_numerics_path
    if export_numerics:
        num_steps = config_fn().training.steps
        metrics = ("loss", "grad_norm")
    else:
        num_steps, metrics = read_golden_spec(golden_path)
    training_options = shlex.join(
        [
            *(
                token
                for fragment in test.override_args[0]
                for token in shlex.split(fragment)
            ),
            *input_options,
        ]
    )
    command = [
        sys.executable,
        "scripts/loss_compare.py",
        ".",
        ".",
        f"--baseline-module={config_fn.__module__}",
        f"--baseline-config={config_fn.__name__}",
        f"--baseline-options={training_options}",
        f"--test-module={config_fn.__module__}",
        f"--test-config={config_fn.__name__}",
        f"--test-options={training_options}",
        f"--steps={num_steps}",
        f"--metrics={','.join(metrics)}",
        f"--baseline-ngpus={test.ngpu}",
        f"--test-ngpus={test.ngpu}",
        f"--job-dump-folder={output_dir}",
        f"--output-folder={output_dir / 'loss_compare'}",
        f"--launcher-command={launcher_command}",
        "--no-seed-checkpoint",
    ]
    if export_numerics:
        command.append(f"--export-result={output_dir / 'numerics_result.txt'}")
    else:
        command.extend(
            [
                "--assert-equal",
                f"--import-result={golden_path}",
            ]
        )
    return command


def run_single_test(test: OverrideDefinitions, args: argparse.Namespace) -> None:
    if len(test.configs) != 1:
        raise ValueError(f"{test.test_name}: scale tests require exactly one config")
    if test.golden_numerics_path is None:
        raise ValueError(f"{test.test_name}: scale tests require golden numerics")
    if test.ngpu % args.num_gpus_per_node != 0:
        raise ValueError(
            f"{test.test_name}: {test.ngpu} GPUs cannot be divided across nodes "
            f"with {args.num_gpus_per_node} GPUs each"
        )

    num_nodes = test.ngpu // args.num_gpus_per_node
    output_dir = Path(args.output_dir) / test.test_name
    if output_dir.exists():
        raise ValueError(f"Test output directory already exists: {output_dir}")

    if not args.export_numerics and not args.dry_run:
        golden_path = GOLDEN_DIR / test.golden_numerics_path
        if not golden_path.is_file():
            raise FileNotFoundError(
                f"Missing golden numerics file: {golden_path}. Run with "
                "--export-numerics to generate a candidate before enabling "
                "comparison."
            )

    if args.launcher == "local":
        if num_nodes != 1:
            raise ValueError(
                f"Scale test '{test.test_name}' requires "
                f"{num_nodes} nodes and cannot use the local launcher"
            )
        launcher_command = "./run_train.sh"
    else:
        launcher_command = _build_slurm_launcher_command(
            test,
            output_dir=output_dir,
            slurm_script=args.slurm_script,
            sbatch_args=args.sbatch_args,
            num_gpus_per_node=args.num_gpus_per_node,
        )

    command = _build_loss_compare_command(
        test,
        output_dir=output_dir,
        launcher_command=launcher_command,
        export_numerics=args.export_numerics,
        input_options=_build_input_options(args),
    )
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return

    output_dir.mkdir(parents=True)
    env = os.environ.copy()
    env["NPROC_PER_NODE"] = str(args.num_gpus_per_node)
    env["OUTPUT_DIR"] = str(output_dir)
    env["RUN_ID"] = output_dir.name

    if args.launcher == "local":
        subprocess.run(
            [
                sys.executable,
                "scripts/collect_environment.py",
                "--output",
                str(output_dir / "environment.json"),
            ],
            env=env,
            check=True,
        )
    else:
        env["TORCHTITAN_COLLECT_ENVIRONMENT"] = "1"

    process = subprocess.run(command, env=env)
    if output_dir.exists():
        baseline_log = output_dir / "loss_compare" / "baseline_training.log"
        training_log = output_dir / "training.log"
        if (
            args.launcher == "local"
            and baseline_log.exists()
            and not training_log.exists()
        ):
            baseline_log.replace(training_log)
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)


def _filter_tests(
    args: argparse.Namespace, test_list: list[OverrideDefinitions]
) -> list[OverrideDefinitions]:
    return [test for test in test_list if test.test_name == args.test_name]


def _build_input_options(args: argparse.Namespace) -> tuple[str, ...]:
    options = []
    if args.checkpoint_initial_load_path is not None:
        options.extend(
            (
                "--checkpoint.enable",
                "--checkpoint.load_only",
                "--checkpoint.initial_load_in_hf",
                "--checkpoint.initial_load_path="
                f"{args.checkpoint_initial_load_path}",
            )
        )
    if args.dataloader_dataset_path is not None:
        options.extend(
            (
                "--dataloader.dataset=c4",
                f"--dataloader.dataset_path={args.dataloader_dataset_path}",
            )
        )
    if args.hf_assets_path is not None:
        options.append(f"--hf_assets_path={args.hf_assets_path}")
    return tuple(options)


def run_tests(args: argparse.Namespace, test_list: list[OverrideDefinitions]) -> None:
    runnable = _filter_tests(args, test_list)
    if not runnable:
        available = ", ".join(test.test_name for test in test_list)
        raise ValueError(
            f"Unknown test '{args.test_name}' in suite '{args.test_suite}'. "
            f"Available: {available}"
        )
    for test in runnable:
        run_single_test(test, args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir",
        help="Directory to dump results generated by tests.",
    )
    parser.add_argument(
        "--test_suite",
        default="gb300_dsv3",
        help="Comma-separated scale test suites to run.",
    )
    parser.add_argument(
        "--test_name",
        required=True,
        help="Specific scale test to run.",
    )
    parser.add_argument(
        "--launcher",
        choices=("local", "slurm"),
        default="slurm",
    )
    parser.add_argument(
        "--slurm-script",
        type=Path,
        default=Path("multinode_trainer.slurm"),
    )
    parser.add_argument(
        "--sbatch-args",
        default="",
        help="Site-specific arguments passed unchanged to sbatch.",
    )
    parser.add_argument(
        "--num_gpus_per_node",
        type=int,
        default=DEFAULT_NUM_GPUS_PER_NODE,
        help="Number of GPUs allocated and launched per Slurm node.",
    )
    parser.add_argument(
        "--checkpoint.initial_load_path",
        dest="checkpoint_initial_load_path",
        type=Path,
        help="Hugging Face checkpoint loaded for the scale test.",
    )
    parser.add_argument(
        "--dataloader.dataset_path",
        dest="dataloader_dataset_path",
        type=Path,
        help="Local C4 revision root used by the scale test.",
    )
    parser.add_argument(
        "--hf_assets_path",
        type=Path,
        help="Local Hugging Face tokenizer assets used by the scale test.",
    )
    parser.add_argument(
        "--export-numerics",
        action="store_true",
        help="Generate candidate numerics instead of comparing with a golden.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without launching training.",
    )
    args = parser.parse_args()

    try:
        test_suites = _parse_test_suites(args.test_suite)
    except ValueError as error:
        parser.error(str(error))
    if args.num_gpus_per_node <= 0:
        parser.error("--num_gpus_per_node must be greater than zero")

    output_dir = Path(args.output_dir).resolve()
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        if any(output_dir.iterdir()):
            raise RuntimeError("Please provide an empty output directory.")

    for test_suite in test_suites:
        suite_args = argparse.Namespace(**vars(args))
        suite_args.test_suite = test_suite
        suite_args.slurm_script = args.slurm_script.resolve()
        suite_args.output_dir = output_dir
        if len(test_suites) > 1:
            suite_args.output_dir = output_dir / test_suite
            if not args.dry_run:
                suite_args.output_dir.mkdir()

        test_list = _TEST_SUITES_FUNCTION[test_suite]()
        run_tests(suite_args, test_list)


if __name__ == "__main__":
    main()
