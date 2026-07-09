# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Integration tests for the RL unified workstream.

Runs the full GRPO training loop (train.py) with different
parallelism configurations. Uses OverrideDefinitions from the shared
test infrastructure but with a custom runner since train.py is
a Monarch script (run with ``python``, not ``torchrun``).

Usage:
    python -m torchtitan.experiments.rl.tests.integration_tests \
        $OUTPUT_DIR --ngpu 4
"""

import argparse
import os
import subprocess
import sys
import time

from tests.integration_tests import OverrideDefinitions

from torchtitan.tools.logging import logger


def build_rl_test_list() -> list[OverrideDefinitions]:
    return [
        OverrideDefinitions(
            [
                [
                    "--module alphabet_sort",
                    "--config rl_grpo_qwen3_0_6b_varlen",
                    "--async-loop.num-training-steps 5",
                    # trainer FSDP=2 (dp_shard=2, tp=1) + 3 generators TP=2 = 8 GPUs.
                    "--trainer.parallelism.data_parallel_shard_degree 2",
                    "--trainer.parallelism.tensor_parallel_degree 1",
                    "--generator.parallelism.tensor_parallel_degree 2",
                    "--num_generators 3",
                    "--async-loop.group-size 2",
                    "--async-loop.batcher.batch.seq-len 1024",
                    "--renderer.enable-thinking False",
                    "--generator.sampling.max_tokens 256",
                    "--trainer.debug.no_batch_invariant",
                    "--generator.debug.no_batch_invariant",
                    "--compile.no-enable",
                    "--generator.cudagraph.no-enable",
                    "--metrics.no-enable-wandb",
                ],
            ],
            "RL GRPO trainer FSDP=2 + gen TP=2 no compile",
            "rl_grpo_fsdp2_gen_tp2_no_compile",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module alphabet_sort",
                    "--config rl_grpo_qwen3_0_6b_varlen",
                    "--async-loop.num-training-steps 5",
                    # trainer FSDP=2 (dp_shard=2, tp=1) + 3 generators TP=2 = 8 GPUs.
                    "--trainer.parallelism.data_parallel_shard_degree 2",
                    "--trainer.parallelism.tensor_parallel_degree 1",
                    "--generator.parallelism.tensor_parallel_degree 2",
                    "--num_generators 3",
                    "--async-loop.group-size 2",
                    "--async-loop.batcher.batch.seq-len 1024",
                    "--renderer.enable-thinking False",
                    "--generator.sampling.max_tokens 256",
                    "--trainer.debug.no_batch_invariant",
                    "--generator.debug.no_batch_invariant",
                    "--metrics.no-enable-wandb",
                ],
            ],
            "RL GRPO trainer FSDP=2 + gen TP=2 compile",
            "rl_grpo_fsdp2_gen_tp2_compile",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module alphabet_sort",
                    "--config rl_grpo_gpt_oss_debug_varlen",
                    "--async-loop.num-training-steps 5",
                    "--hf_assets_path tests/assets/tokenizer",
                    "--trainer.parallelism.tensor_parallel_degree 4",
                    "--trainer.parallelism.expert_parallel_degree 4",
                    "--trainer.parallelism.data_parallel_shard_degree 1",
                    "--generator.parallelism.tensor_parallel_degree 4",
                    "--generator.parallelism.expert_parallel_degree 4",
                    "--generator.parallelism.data_parallel_degree 1",
                    "--async-loop.group-size 2",
                    "--async-loop.batcher.batch.seq-len 1024",
                    "--renderer.enable-thinking False",
                    "--generator.sampling.max_tokens 256",
                    "--trainer.debug.no_batch_invariant",
                    "--generator.debug.no_batch_invariant",
                    "--trainer.checkpoint.no-enable",  # use random-init weights
                    "--generator.checkpoint.no-enable",  # use random-init weights
                    "--compile.no-enable",
                    "--generator.cudagraph.no-enable",
                    "--metrics.no-enable-wandb",
                ],
            ],
            "RL GRPO GPT-OSS MoE varlen TP=4 EP=4",
            "rl_grpo_moe_debug_tp4_ep4",
            ngpu=8,
        ),
        OverrideDefinitions(
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
            [
                [
                    "--module alphabet_sort",
                    "--config rl_grpo_qwen3_0_6b_varlen",
                    "--async-loop.num-training-steps 2",
                    "--num_generators 2",
                    "--trainer.parallelism.data_parallel_shard_degree 2",
                    "--trainer.parallelism.tensor_parallel_degree 1",
                    "--generator.parallelism.tensor_parallel_degree 2",
                    "--async-loop.group-size 2",
                    "--async-loop.batcher.batch.seq-len 1024",
                    "--renderer.enable-thinking False",
                    "--generator.sampling.max_tokens 256",
                    "--trainer.debug.no_batch_invariant",
                    "--generator.debug.no_batch_invariant",
                    "--metrics.no-enable-wandb",
                    "--trainer.checkpoint.interval 2",
                    "--trainer.lr_scheduler.total_steps 4",
                ],
                [
                    "--module alphabet_sort",
                    "--config rl_grpo_qwen3_0_6b_varlen",
                    "--async-loop.num-training-steps 4",
                    "--num_generators 1",
                    "--trainer.parallelism.data_parallel_shard_degree 1",
                    "--trainer.parallelism.tensor_parallel_degree 2",
                    "--generator.parallelism.tensor_parallel_degree 4",
                    "--async-loop.group-size 2",
                    "--async-loop.batcher.batch.seq-len 1024",
                    "--renderer.enable-thinking False",
                    "--generator.sampling.max_tokens 256",
                    "--trainer.debug.no_batch_invariant",
                    "--generator.debug.no_batch_invariant",
                    "--metrics.no-enable-wandb",
                    "--trainer.checkpoint.interval 2",
                    "--trainer.lr_scheduler.total_steps 4",
                ],
            ],
            "RL GRPO checkpoint save + resume (resharding)",
            "rl_grpo_checkpoint_resume",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module alphabet_sort",
                    "--config rl_grpo_qwen3_0_6b_varlen_batch_invariant",
                    "--async-loop.num-training-steps 3",
                    # The config defaults to trainer TP=2 + 3 generators TP=2. Override
                    # to trainer TP=4 + 1 generator TP=4 so batch-invariant mode fits
                    # A10G: TP=2 shards less per GPU and OOMs with BI on.
                    "--trainer.parallelism.tensor_parallel_degree 4",
                    "--generator.parallelism.tensor_parallel_degree 4",
                    "--num_generators 1",
                    # On-policy (lockstep) + real weights that update each step:
                    # trainer/generator weights match, so bit_wise/logprob_diff/max == 0.
                    "--async-loop.max-offpolicy-steps 0",
                    "--async-loop.group-size 2",
                    "--async-loop.batcher.batch.seq-len 1024",
                    "--renderer.enable-thinking False",
                    "--generator.sampling.max_tokens 128",
                    "--metrics.no-enable-wandb",
                ],
            ],
            "RL GRPO 0.6B TP=4 batch-invariant + deterministic",
            "rl_grpo_0_6b_tp4_batch_invariant",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module alphabet_sort",
                    "--config rl_grpo_qwen3_moe_debug_varlen_batch_invariant",
                    "--async-loop.num-training-steps 5",
                    "--hf_assets_path tests/assets/tokenizer",
                    "--async-loop.group-size 2",
                    "--async-loop.batcher.batch.seq-len 1024",
                    "--renderer.enable-thinking False",
                    "--generator.sampling.max_tokens 256",
                    "--trainer.checkpoint.no-enable",  # use random-init weights
                    "--generator.checkpoint.no-enable",
                    "--metrics.no-enable-wandb",
                ],
            ],
            "RL GRPO MoE TP=4 EP=4 batch-invariant",
            "rl_grpo_moe_debug_tp4_ep4_batch_invariant",
            ngpu=8,
        ),
    ]


def run_single_test(
    test_flavor: OverrideDefinitions,
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

    for override_arg in test_flavor.override_args:
        cmd_parts = [
            sys.executable,
            "-m",
            "torchtitan.experiments.rl.train",
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

    test_list = build_rl_test_list()
    run_tests(args, test_list)


if __name__ == "__main__":
    main()
