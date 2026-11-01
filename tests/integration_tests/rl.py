# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""RL integration-test suites: the full async RL loop (``torchtitan.rl.train``).

Configurations live in ``torchtitan_recipes.tests.rl``. Run with, e.g.::

    python -m tests.integration_tests.run_tests $OUTPUT_DIR \
        --test_suite rl --ngpu 8
"""

from torchtitan_recipes.tests.rl.alphabet_sort import (
    rl_grpo_0_6b_tp4_batch_invariant,
    rl_grpo_checkpoint_resume,
    rl_grpo_checkpoint_save,
    rl_grpo_cp2_fsdp2_gen_tp2,
    rl_grpo_deepseek_v3_debug,
    rl_grpo_fsdp2_gen_tp2_compile,
    rl_grpo_fsdp2_gen_tp2_no_compile,
    rl_grpo_kimi_k3_debug_batch_invariant,
    rl_grpo_moe_debug_dist_moe_tp2_ep4,
    rl_grpo_moe_debug_tp4_ep4,
    rl_grpo_moe_debug_tp4_ep4_batch_invariant,
    rl_grpo_pp2_interleaved1f1b_fsdp2_gen_tp2,
    rl_grpo_qwen3_5_debug_tp2_batch_invariant,
)

from tests.integration_tests import IntegrationTestDefinition


RL_ENTRYPOINT = "torchtitan.rl.train"


def _rl_test(**kwargs) -> IntegrationTestDefinition:
    """Define an RL test: every config runs the RL entrypoint with real PGs."""
    return IntegrationTestDefinition(
        entrypoints=[RL_ENTRYPOINT] * len(kwargs["configs"]),
        use_real_pg=True,
        **kwargs,
    )


def build_rl_test_list() -> list[IntegrationTestDefinition]:
    return [
        _rl_test(
            configs=[rl_grpo_fsdp2_gen_tp2_no_compile],
            test_descr="RL GRPO trainer FSDP=2 + gen TP=2 no compile",
            test_name="rl_grpo_fsdp2_gen_tp2_no_compile",
            ngpu=8,
        ),
        _rl_test(
            configs=[rl_grpo_cp2_fsdp2_gen_tp2],
            test_descr="RL GRPO trainer FSDP=2 + Ulysses varlen CP=2 + gen TP=2",
            test_name="rl_grpo_cp2_fsdp2_gen_tp2",
            ngpu=8,
        ),
        _rl_test(
            configs=[rl_grpo_fsdp2_gen_tp2_compile],
            test_descr="RL GRPO trainer FSDP=2 + gen TP=2 compile",
            test_name="rl_grpo_fsdp2_gen_tp2_compile",
            ngpu=8,
        ),
        _rl_test(
            configs=[rl_grpo_pp2_interleaved1f1b_fsdp2_gen_tp2],
            test_descr="RL GRPO trainer PP=2 Interleaved1F1B FSDP=2 + gen TP=2",
            test_name="rl_grpo_pp2_interleaved1f1b_fsdp2_gen_tp2",
            ngpu=8,
        ),
        _rl_test(
            configs=[rl_grpo_moe_debug_tp4_ep4],
            test_descr="RL GRPO GPT-OSS MoE varlen TP=4 EP=4",
            test_name="rl_grpo_moe_debug_tp4_ep4",
            ngpu=8,
        ),
        _rl_test(
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
        _rl_test(
            configs=[rl_grpo_0_6b_tp4_batch_invariant],
            test_descr="RL GRPO 0.6B TP=4 batch-invariant + deterministic",
            test_name="rl_grpo_0_6b_tp4_batch_invariant",
            ngpu=8,
        ),
        _rl_test(
            configs=[rl_grpo_moe_debug_tp4_ep4_batch_invariant],
            test_descr="RL GRPO MoE TP=4 EP=4 batch-invariant",
            test_name="rl_grpo_moe_debug_tp4_ep4_batch_invariant",
            ngpu=8,
        ),
        _rl_test(
            configs=[rl_grpo_qwen3_5_debug_tp2_batch_invariant],
            test_descr="RL GRPO Qwen3.5 hybrid GDN TP=2 batch-invariant",
            test_name="rl_grpo_qwen3_5_debug_tp2_batch_invariant",
            ngpu=8,
        ),
    ]


def build_h100_rl_test_list() -> list[IntegrationTestDefinition]:
    """Build RL integration tests for Attention Gym KDA and vLLM's native MLA, which require SM90+."""
    return [
        _rl_test(
            configs=[rl_grpo_kimi_k3_debug_batch_invariant],
            test_descr="RL GRPO Kimi K3 hybrid KDA batch-invariant",
            test_name="rl_grpo_kimi_k3_debug_batch_invariant",
            ngpu=4,
        ),
        _rl_test(
            configs=[rl_grpo_deepseek_v3_debug],
            test_descr="RL GRPO DeepSeek V3 compressed MLA cache + decode CUDA graphs",
            test_name="rl_grpo_deepseek_v3_debug",
            ngpu=4,
        ),
    ]


def build_b200_rl_test_list() -> list[IntegrationTestDefinition]:
    """Build the B200 RL integration tests: the H100 tests, plus Dist-MoE, which requires SM100+."""
    return build_h100_rl_test_list() + [
        _rl_test(
            configs=[rl_grpo_moe_debug_dist_moe_tp2_ep4],
            test_descr=(
                "RL GRPO MoE Dist-MoE trainer FSDP=2 TP=2 EP=4 + generator "
                "DP=2 TP=2 EP=4 with CUDA graphs"
            ),
            test_name="rl_grpo_moe_debug_dist_moe_tp2_ep4",
            ngpu=8,
        ),
    ]
