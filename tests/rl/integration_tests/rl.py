# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Integration tests for the RL unified workstream.

Runs the full GRPO training loop (train.py) with different
parallelism configurations. Uses IntegrationTestDefinition from the shared
test infrastructure but with a custom runner since train.py is
a Monarch script (run with ``python``, not ``torchrun``).

Usage:
    python -m tests.integration_tests.rl \
        $OUTPUT_DIR --ngpu 4
"""

import argparse

import logging
import os
import subprocess
import sys
import time

from torchtitan.observability.logging import init_logger
from torchtitan_recipes.tests.rl import (
    rl_grpo_0_6b_tp4_batch_invariant,
    rl_grpo_checkpoint_resume,
    rl_grpo_checkpoint_save,
    rl_grpo_fsdp2_gen_tp2_compile,
    rl_grpo_fsdp2_gen_tp2_no_compile,
    rl_grpo_kimi_k3_debug_batch_invariant,
    rl_grpo_moe_debug_tp4_ep4,
    rl_grpo_moe_debug_tp4_ep4_batch_invariant,
    rl_grpo_qwen3_5_debug_tp2_batch_invariant,
)

from tests.integration_tests import (
    get_importable_config_module,
    IntegrationTestDefinition,
)


logger = logging.getLogger(__name__)


def build_rl_test_list() -> list[IntegrationTestDefinition]:
    return [
        IntegrationTestDefinition(
            configs=[rl_grpo_fsdp2_gen_tp2_no_compile],
            test_descr="RL GRPO trainer FSDP=2 + gen TP=2 no compile",
            test_name="rl_grpo_fsdp2_gen_tp2_no_compile",
            ngpu=8,
        ),
        IntegrationTestDefinition(
            configs=[rl_grpo_fsdp2_gen_tp2_compile],
            test_descr="RL GRPO trainer FSDP=2 + gen TP=2 compile",
            test_name="rl_grpo_fsdp2_gen_tp2_compile",
            ngpu=8,
        ),
        IntegrationTestDefinition(
            configs=[rl_grpo_moe_debug_tp4_ep4],
            test_descr="RL GRPO GPT-OSS MoE varlen TP=4 EP=4",
            test_name="rl_grpo_moe_debug_tp4_ep4",
            ngpu=8,
        ),
        IntegrationTestDefinition(
            # Two runs sharing the same dump_folder, with different parallelism
            # to exercise resharding on resume:
            #   run 1: trainer DP=2 (TP=1) + 2 generators (TP=2); train 2 steps,
            #          write a full checkpoint at step 2.
            #   run 2: trainer TP=2 (DP=1) + 1 generator (TP=4); resume from the
            #          step-2 checkpoint and train through step 4.
            # This covers (1) trainer DCP checkpoint resharding (DP->TP),
            # (2) trainer->generator TorchStore resharding, and (3) the
            # multi-generator vs single-generator paths. The second run errors
            # if resume is broken. lr_scheduler.total_steps is pinned so the LR
            # is identical across save/load.
            configs=[rl_grpo_checkpoint_save, rl_grpo_checkpoint_resume],
            test_descr="RL GRPO checkpoint save + resume (resharding)",
            test_name="rl_grpo_checkpoint_resume",
            ngpu=8,
        ),
        IntegrationTestDefinition(
            configs=[rl_grpo_0_6b_tp4_batch_invariant],
            test_descr="RL GRPO 0.6B TP=4 batch-invariant + deterministic",
            test_name="rl_grpo_0_6b_tp4_batch_invariant",
            ngpu=8,
        ),
        IntegrationTestDefinition(
            configs=[rl_grpo_moe_debug_tp4_ep4_batch_invariant],
            test_descr="RL GRPO MoE TP=4 EP=4 batch-invariant",
            test_name="rl_grpo_moe_debug_tp4_ep4_batch_invariant",
            ngpu=8,
        ),
        IntegrationTestDefinition(
            configs=[rl_grpo_qwen3_5_debug_tp2_batch_invariant],
            test_descr="RL GRPO Qwen3.5 hybrid GDN TP=2 batch-invariant",
            test_name="rl_grpo_qwen3_5_debug_tp2_batch_invariant",
            ngpu=8,
        ),
    ]


def build_rl_kda_test_list() -> list[IntegrationTestDefinition]:
    """Build RL integration tests for Attention Gym KDA, which requires SM90+."""
    return [
        IntegrationTestDefinition(
            configs=[rl_grpo_kimi_k3_debug_batch_invariant],
            test_descr="RL GRPO Kimi K3 hybrid KDA batch-invariant",
            test_name="rl_grpo_kimi_k3_debug_batch_invariant",
            ngpu=4,
        ),
    ]


_TEST_SUITES_FUNCTION = {
    "default": build_rl_test_list,
    "kda": build_rl_kda_test_list,
}


def run_single_test(
    test_flavor: IntegrationTestDefinition,
    output_dir: str,
    hf_assets_path: str = "",
) -> None:
    """Run a single RL integration test.

    Unlike the standard run_tests which uses ``./run_train.sh`` (torchrun),
    this runs the RL training module directly since the RL script manages
    its own distributed setup via Monarch.
    """
    test_name = test_flavor.test_name
    dump_folder = os.path.join(output_dir, test_name)

    for config_fn in test_flavor.configs:
        cmd_parts = [
            sys.executable,
            "-m",
            "torchtitan.rl.train",
            "--module",
            get_importable_config_module(config_fn),
            "--config",
            config_fn.__name__,
            "--output-dir",
            dump_folder,
        ]
        env = os.environ.copy()
        if hf_assets_path:
            env["TORCHTITAN_TEST_HF_ASSETS_PATH"] = hf_assets_path
        cmd = " ".join(cmd_parts)

        logger.info(
            f"===== {time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"RL integration test: {test_flavor.test_descr}, command: {cmd} ====="
        )

        result = subprocess.run(cmd_parts, text=True, env=env)
        if result.returncode != 0:
            raise Exception(
                f"RL integration test failed: {test_flavor.test_descr}, command: {cmd}"
            )


def run_tests(args, test_list: list[IntegrationTestDefinition]) -> None:
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
    init_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="Directory to dump results")
    parser.add_argument(
        "--test_suite",
        default="default",
        choices=sorted(_TEST_SUITES_FUNCTION),
        help="Test suite to run (default: default)",
    )
    parser.add_argument(
        "--test_name",
        default="all",
        help="Specific test to run (default: all)",
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

    test_list = _TEST_SUITES_FUNCTION[args.test_suite]()
    run_tests(args, test_list)


if __name__ == "__main__":
    main()
