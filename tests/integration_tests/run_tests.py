# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import subprocess
import time
from dataclasses import dataclass

from torchtitan.tools.logging import logger

from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.features import build_features_test_list
from tests.integration_tests.h100 import build_h100_tests_list
from tests.integration_tests.models import build_model_tests_list


class _IntegrationTestFailure(RuntimeError):
    """Carries structured info about a failed integration test."""

    def __init__(self, test_descr: str, cmd: str, stderr: str):
        self.cmd = cmd
        self.stderr_text = stderr
        super().__init__(
            f"\nFailed test flavor: {test_descr}.\n"
            f"Command: {cmd}\n"
            f"stderr: {stderr}\n"
        )


@dataclass
class _FailureRecord:
    """Structured record of a single integration test failure."""

    test_name: str
    test_descr: str
    cmd: str
    stderr_tail: str


_TEST_SUITES_FUNCTION = {
    "features": build_features_test_list,
    "models": build_model_tests_list,
    "h100": build_h100_tests_list,
}


def _run_cmd(cmd, timeout=None):
    return subprocess.run(
        [cmd],
        encoding="utf-8",
        errors="replace",
        shell=True,
        capture_output=True,
        timeout=timeout,
    )


def run_single_test(
    test_flavor: OverrideDefinitions,
    output_dir: str,
    module: str | None = None,
    config: str | None = None,
):
    # run_test supports sequence of tests.
    test_name = test_flavor.test_name
    dump_folder_arg = f"--dump_folder {output_dir}/{test_name}"

    all_ranks = ",".join(map(str, range(test_flavor.ngpu)))

    for idx, override_arg in enumerate(test_flavor.override_args):
        cmd = ""
        if module is not None:
            cmd += f"MODULE={module} "
        if config is not None:
            cmd += f"CONFIG={config} "
        cmd = f"NGPU={test_flavor.ngpu} LOG_RANK={all_ranks} ./run_train.sh"

        # dump compile trace for debugging purpose
        cmd = f'TORCH_TRACE="{output_dir}/{test_name}/compile_trace" ' + cmd

        cmd += " " + dump_folder_arg
        if override_arg:
            cmd += " " + " ".join(override_arg)
        logger.info(
            f"===== {time.strftime('%Y-%m-%d %H:%M:%S')} Integration test, flavor : {test_flavor.test_descr}, command : {cmd}====="
        )

        # save checkpoint (idx == 0) and load it for generation (idx == 1)
        if test_name == "test_generate" and idx == 1:
            cmd = (
                f"NGPU={test_flavor.ngpu} LOG_RANK={all_ranks} "
                f"CHECKPOINT_DIR={output_dir}/{test_name}/checkpoint/step-10 "
                "PROMPT='What is the meaning of life?' "
                f"./scripts/generate/run_llama_generate.sh --out > {output_dir}/{test_name}/generated_output.json"
            )

        try:
            result = _run_cmd(cmd, timeout=test_flavor.timeout)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"\nTest timed out after {test_flavor.timeout}s: {test_flavor.test_descr}.\n"
                f"Command: {cmd}\n"
            ) from e
        if result.stdout:
            logger.info(result.stdout)
        if result.returncode != 0:
            raise _IntegrationTestFailure(test_flavor.test_descr, cmd, result.stderr)


def _print_failure_summary(failed_tests: list[_FailureRecord]):
    """Print a clear, scannable failure summary and raise RuntimeError.

    This is printed at the very end of the test run so it's easy to find
    in flooded CI logs. Each failure includes the test name and a
    copy-paste repro command.
    """
    sep = "=" * 70
    n = len(failed_tests)
    lines = [
        "",
        sep,
        f"  FAILURE SUMMARY: {n} integration test(s) failed",
        sep,
    ]
    for i, rec in enumerate(failed_tests, 1):
        lines.append("")
        lines.append(f"  [{i}/{n}] {rec.test_name} — {rec.test_descr}")
        lines.append("")
        lines.append("  Repro command:")
        lines.append(f"    {rec.cmd}")
        if rec.stderr_tail:
            lines.append("")
            lines.append("  Last lines of stderr:")
            for sline in rec.stderr_tail.splitlines():
                lines.append(f"    {sline}")
    lines.append("")
    lines.append(sep)
    summary = "\n".join(lines)
    logger.error(summary)
    raise RuntimeError(summary)


def run_tests(args, test_list: list[OverrideDefinitions], module=None, config=None):
    """Run all integration tests to test the core features of TorchTitan"""
    exclude_set = set()
    if hasattr(args, "exclude") and args.exclude:
        exclude_set = {name.strip() for name in args.exclude.split(",")}

    ran_any_test = False
    failed_tests: list[_FailureRecord] = []
    for test_flavor in test_list:
        # Filter by test_name if specified
        if args.test_name != "all" and test_flavor.test_name != args.test_name:
            continue

        if test_flavor.disabled or test_flavor.test_name in exclude_set:
            continue

        # Skip the test for ROCm
        if (
            getattr(args, "gpu_arch_type", "cuda") == "rocm"
            and test_flavor.skip_rocm_test
        ):
            continue

        # Check if we have enough GPUs
        if args.ngpu < test_flavor.ngpu:
            logger.info(
                f"Skipping test {test_flavor.test_name} that requires {test_flavor.ngpu} gpus,"
                f" because --ngpu arg is {args.ngpu}"
            )
        else:
            try:
                run_single_test(test_flavor, args.output_dir, module, config)
            except _IntegrationTestFailure as e:
                logger.error(str(e))
                # Keep last 30 lines of stderr for the summary
                stderr_lines = e.stderr_text.strip().splitlines()
                tail = "\n".join(stderr_lines[-30:])
                if len(stderr_lines) > 30:
                    tail = f"... ({len(stderr_lines) - 30} lines truncated)\n" + tail
                failed_tests.append(
                    _FailureRecord(
                        test_flavor.test_name,
                        test_flavor.test_descr,
                        e.cmd,
                        tail,
                    )
                )
            except Exception as e:
                logger.error(str(e))
                failed_tests.append(
                    _FailureRecord(
                        test_flavor.test_name,
                        test_flavor.test_descr,
                        cmd="<unknown>",
                        stderr_tail=str(e),
                    )
                )
            ran_any_test = True

    if failed_tests:
        _print_failure_summary(failed_tests)

    if not ran_any_test:
        available_tests = [t.test_name for t in test_list if not t.disabled]
        if hasattr(args, "test_suite"):
            logger.warning(
                f"No tests were run for --test_name '{args.test_name}' in test suite '{args.test_suite}'.\n"
                f"Available test names in '{args.test_suite}' suite: {available_tests}"
            )
        else:
            logger.warning(
                f"No tests were run for --test_name '{args.test_name}'.\n"
                f"Available test names: {available_tests}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir", help="Directory to dump results generated by tests"
    )
    parser.add_argument(
        "--gpu_arch_type",
        default="cuda",
        choices=["cuda", "rocm"],
        help="GPU architecture type. Must be specified as either 'cuda' or 'rocm'.",
    )
    parser.add_argument(
        "--test_suite",
        default="features",
        choices=["features", "models", "h100"],
        help="Which test suite to run. If not specified, torchtitan composability tests will be run",
    )
    parser.add_argument(
        "--module",
        default="llama3",
        help="Model module to use for training (default: llama3). "
        "This is passed as MODULE env var to run_train.sh.",
    )
    parser.add_argument(
        "--config",
        default="llama3_debugmodel",
        help="Config function to use for training (default: llama3_debugmodel). "
        "This is passed as CONFIG env var to run_train.sh.",
    )
    parser.add_argument(
        "--test_name",
        default="all",
        help="Specific test name to run (e.g., 'tp_only', 'full_checkpoint'). Use 'all' to run all tests (default: all)",
    )
    parser.add_argument(
        "--ngpu", default=8, type=int, help="Maximum number of GPUs to use"
    )
    parser.add_argument(
        "--exclude",
        default=None,
        help="Comma-separated list of test names to skip",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.listdir(args.output_dir):
        raise RuntimeError("Please provide an empty output directory.")

    assert (
        args.test_suite in _TEST_SUITES_FUNCTION
    ), f"Unknown test suite {args.test_suite}"

    test_list = _TEST_SUITES_FUNCTION[args.test_suite]()
    run_tests(args, test_list)


if __name__ == "__main__":
    main()
