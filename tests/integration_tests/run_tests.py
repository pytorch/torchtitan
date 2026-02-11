# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import subprocess
import tempfile
import time

from torchtitan.tools.logging import logger
from torchtitan.tools.loss_utils import compare_losses, extract_losses_from_log

from tests.integration_tests import OverrideDefinitions

from tests.integration_tests.features import build_features_test_list
from tests.integration_tests.h100 import build_h100_tests_list
from tests.integration_tests.models import build_model_tests_list


_TEST_SUITES_FUNCTION = {
    "features": build_features_test_list,
    "models": build_model_tests_list,
    "h100": build_h100_tests_list,
}


def _run_cmd(cmd):
    return subprocess.run([cmd], text=True, shell=True)


def run_single_test(test_flavor: OverrideDefinitions, full_path: str, output_dir: str):
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
            f"===== {time.strftime('%Y-%m-%d %H:%M:%S')} Integration test, flavor : {test_flavor.test_descr}, command : {cmd}====="
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


def run_determinism_test(
    test_flavor: OverrideDefinitions, full_path: str, output_dir: str
):
    """Run a test twice and verify losses are identical (run-to-run determinism).

    This runs the same configuration twice with deterministic settings enabled,
    then compares the losses from both runs to ensure they match exactly.
    """
    test_name = test_flavor.test_name
    all_ranks = ",".join(map(str, range(test_flavor.ngpu)))

    # Build the base command with determinism flags
    override_arg = test_flavor.override_args[0] if test_flavor.override_args else []
    override_str = " ".join(override_arg) if override_arg else ""

    base_cmd = (
        f"CONFIG_FILE={full_path} NGPU={test_flavor.ngpu} LOG_RANK={all_ranks} "
        f"./run_train.sh --job.dump_folder {output_dir}/{test_name}_determinism "
        f"--debug.deterministic --debug.seed=42 --training.steps=10"
    )
    if override_str:
        base_cmd += " " + override_str

    logger.info(
        f"===== {time.strftime('%Y-%m-%d %H:%M:%S')} Determinism test, flavor : {test_flavor.test_descr} ====="
    )

    # Create temp files for logs
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_run1.log", delete=False
    ) as log1_file:
        log1_path = log1_file.name
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_run2.log", delete=False
    ) as log2_file:
        log2_path = log2_file.name

    try:
        # Run 1
        logger.info(f"Determinism test run 1: {base_cmd}")
        cmd1 = f"{base_cmd} 2>&1 | tee {log1_path}"
        result1 = _run_cmd(cmd1)
        if result1.returncode != 0:
            raise Exception(
                f"Determinism test run 1 failed, flavor : {test_flavor.test_descr}"
            )

        # Run 2
        logger.info(f"Determinism test run 2: {base_cmd}")
        cmd2 = f"{base_cmd} 2>&1 | tee {log2_path}"
        result2 = _run_cmd(cmd2)
        if result2.returncode != 0:
            raise Exception(
                f"Determinism test run 2 failed, flavor : {test_flavor.test_descr}"
            )

        # Extract and compare losses
        losses1 = extract_losses_from_log(log1_path)
        losses2 = extract_losses_from_log(log2_path)

        success, message = compare_losses(losses1, losses2, "run1", "run2")
        if not success:
            raise Exception(
                f"Determinism test failed for {test_flavor.test_descr}: {message}"
            )

        logger.info(f"Determinism test passed for {test_flavor.test_descr}: {message}")

    finally:
        # Clean up temp files
        if os.path.exists(log1_path):
            os.remove(log1_path)
        if os.path.exists(log2_path):
            os.remove(log2_path)


def run_tests(args, test_list: list[OverrideDefinitions]):
    """Run all integration tests to test the core features of TorchTitan"""

    # Check if config file exists
    assert args.config_path.endswith(".toml"), "Base config path must end with .toml"
    assert os.path.exists(
        args.config_path
    ), f"Base config path {args.config_path} does not exist"

    ran_any_test = False
    for test_flavor in test_list:
        # Filter by test_name if specified
        if args.test_name != "all" and test_flavor.test_name != args.test_name:
            continue

        if test_flavor.disabled:
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
            run_single_test(test_flavor, args.config_path, args.output_dir)
            ran_any_test = True

            # Run determinism test if enabled (CUDA only)
            if (
                test_flavor.determinism_test
                and getattr(args, "gpu_arch_type", "cuda") == "cuda"
            ):
                run_determinism_test(test_flavor, args.config_path, args.output_dir)

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
        "--config_path",
        default="./tests/integration_tests/base_config.toml",
        help="Base config path for integration tests. This is the config that will be used as a base for all tests.",
    )
    parser.add_argument(
        "--test_name",
        default="all",
        help="Specific test name to run (e.g., 'tp_only', 'full_checkpoint'). Use 'all' to run all tests (default: all)",
    )
    parser.add_argument(
        "--ngpu", default=8, type=int, help="Maximum number of GPUs to use"
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
