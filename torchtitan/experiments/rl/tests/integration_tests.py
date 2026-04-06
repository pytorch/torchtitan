# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Integration tests for the RL unified workstream.

Runs the full GRPO training loop (simple_grpo.py) with different
parallelism configurations. Uses OverrideDefinitions from the shared
test infrastructure but with a custom runner since simple_grpo.py is
a Monarch script (run with ``python``, not ``torchrun``).

Usage:
    python -m torchtitan.experiments.rl.tests.integration_tests \
        $OUTPUT_DIR --ngpu 4
"""

import argparse
import os
import subprocess
import time

from tests.integration_tests import OverrideDefinitions

from torchtitan.tools.logging import logger


def build_rl_test_list() -> list[OverrideDefinitions]:
    return [
        OverrideDefinitions(
            [
                [
                    "--module rl",
                    "--config rl_grpo_qwen3_0_6b",
                    "--trainer.parallelism.tensor_parallel_degree 2",
                    "--generator.parallelism.tensor_parallel_degree 2",
                    "--generator.num_samples_per_prompt 2",
                    "--trainer.debug.no_batch_invariant",
                    "--generator.debug.no_batch_invariant",
                    "--trainer.compile.no-enable",
                    "--generator.compile.backend none",
                    "--generator.compile.cudagraph_mode none",
                ],
            ],
            "RL GRPO TP=2 no compile",
            "rl_grpo_tp2_no_compile",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module rl",
                    "--config rl_grpo_qwen3_0_6b",
                    "--trainer.parallelism.tensor_parallel_degree 2",
                    "--generator.parallelism.tensor_parallel_degree 2",
                    "--generator.num_samples_per_prompt 2",
                    "--trainer.debug.no_batch_invariant",
                    "--generator.debug.no_batch_invariant",
                ],
            ],
            "RL GRPO TP=2 compile",
            "rl_grpo_tp2_compile",
            ngpu=4,
        ),
    ]


def build_rl_h100_test_list() -> list[OverrideDefinitions]:
    return [
        OverrideDefinitions(
            [
                [
                    "--module rl",
                    "--config rl_grpo_qwen3_0_6b_batch_invariant",
                ],
            ],
            "RL GRPO TP=2 batch-invariant + deterministic",
            "rl_grpo_tp2_batch_invariant",
            ngpu=4,
        ),
    ]


_TEST_SUITES = {
    "default": build_rl_test_list,
    "h100": build_rl_h100_test_list,
}


def run_single_test(
    test_flavor: OverrideDefinitions,
    output_dir: str,
    hf_assets_path: str = "",
) -> None:
    """Run a single RL integration test.

    Unlike the standard run_tests which uses ``./run_train.sh`` (torchrun),
    this runs ``python simple_grpo.py`` directly since the RL script manages
    its own distributed setup via Monarch.
    """
    test_name = test_flavor.test_name
    dump_folder = os.path.join(output_dir, test_name)

    for override_arg in test_flavor.override_args:
        cmd_parts = [
            "python",
            "torchtitan/experiments/rl/simple_grpo_sum_digits.py",
            f"--dump_folder {dump_folder}",
        ]
        if hf_assets_path:
            cmd_parts.append(f"--hf_assets_path {hf_assets_path}")
        cmd_parts.extend(override_arg)
        cmd = " ".join(cmd_parts)

        logger.info(
            f"===== {time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"RL integration test: {test_flavor.test_descr}, command: {cmd} ====="
        )

        result = subprocess.run(cmd, text=True, shell=True)
        if result.returncode != 0:
            raise Exception(
                f"RL integration test failed: {test_flavor.test_descr}, command: {cmd}"
            )


def run_tests(args, test_list: list[OverrideDefinitions]) -> None:
    ran_any = False
    for test_flavor in test_list:
        if args.test_name != "all" and test_flavor.test_name != args.test_name:
            continue
        if test_flavor.disabled:
            continue
        if args.ngpu < test_flavor.ngpu:
            logger.info(
                f"Skipping test {test_flavor.test_name} (needs {test_flavor.ngpu} GPUs, "
                f"have {args.ngpu})"
            )
            continue

        run_single_test(test_flavor, args.output_dir, args.hf_assets_path)
        ran_any = True

    if not ran_any:
        available = [t.test_name for t in test_list if not t.disabled]
        logger.warning(
            f"No tests were run for --test_name '{args.test_name}'.\n"
            f"Available: {available}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="Directory to dump results")
    parser.add_argument(
        "--test_name",
        default="all",
        help="Specific test to run (default: all)",
    )
    parser.add_argument(
        "--test_suite",
        default="default",
        choices=list(_TEST_SUITES.keys()),
        help="Which test suite to run (default: default)",
    )
    parser.add_argument(
        "--ngpu",
        default=4,
        type=int,
        help="Maximum number of GPUs available",
    )
    parser.add_argument(
        "--hf_assets_path",
        default="",
        help="Path to HF model checkpoint (weights, tokenizer, config)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    test_list = _TEST_SUITES[args.test_suite]()
    run_tests(args, test_list)


if __name__ == "__main__":
    main()
