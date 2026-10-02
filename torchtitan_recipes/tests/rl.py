# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurations for reinforcement-learning integration tests."""

import os

from torchtitan.rl.controller import Controller

from torchtitan_recipes.rl.alphabet_sort import (
    rl_grpo_gpt_oss_debug_varlen_no_compile,
    rl_grpo_kimi_k3_debug_varlen_batch_invariant,
    rl_grpo_qwen3_0_6b_varlen,
    rl_grpo_qwen3_0_6b_varlen_batch_invariant,
    rl_grpo_qwen3_0_6b_varlen_checkpoint_test,
    rl_grpo_qwen3_0_6b_varlen_no_compile,
    rl_grpo_qwen3_5_debug_varlen_batch_invariant,
    rl_grpo_qwen3_moe_debug_varlen_batch_invariant,
)


def _configure_ci(
    config: Controller.Config,
    *,
    steps: int,
    num_tokens_per_microbatch: int,
    max_generated_tokens: int,
    use_hf_assets_from_env: bool,
) -> Controller.Config:
    config.async_loop.num_training_steps = steps
    config.async_loop.num_samples_per_prompt = 2
    config.async_loop.training_sample_builder.drop_zero_std_reward_groups = False
    config.trainer.training.num_tokens_per_microbatch_per_dp_rank = (
        num_tokens_per_microbatch
    )
    config.generator.sampling.max_tokens = max_generated_tokens
    config.generator.gpu_memory_limit = 0.8
    config.metrics.enable_wandb = False
    config.rollouter.worker.token_env.max_rollout_tokens = (
        config.model.max_context_length
    )
    if use_hf_assets_from_env:
        assets_path = os.environ.get("TORCHTITAN_TEST_HF_ASSETS_PATH")
        if assets_path:
            config.hf_assets_path = assets_path
    return config


def rl_grpo_fsdp2_gen_tp2_no_compile() -> Controller.Config:
    config = rl_grpo_qwen3_0_6b_varlen_no_compile(seq_len=1024)
    config.num_generators = 3
    config.trainer.parallelism.data_parallel_shard_degree = 2
    config.trainer.parallelism.tensor_parallel_degree = 1
    config.generator.parallelism.tensor_parallel_degree = 2
    return _configure_ci(
        config,
        steps=5,
        num_tokens_per_microbatch=2048,
        max_generated_tokens=256,
        use_hf_assets_from_env=True,
    )


def rl_grpo_fsdp2_gen_tp2_compile() -> Controller.Config:
    config = rl_grpo_qwen3_0_6b_varlen(seq_len=1024)
    config.num_generators = 3
    config.trainer.parallelism.data_parallel_shard_degree = 2
    config.trainer.parallelism.tensor_parallel_degree = 1
    config.generator.parallelism.tensor_parallel_degree = 2
    return _configure_ci(
        config,
        steps=5,
        num_tokens_per_microbatch=2048,
        max_generated_tokens=256,
        use_hf_assets_from_env=True,
    )


def rl_grpo_pp2_fsdp2_gen_tp2() -> Controller.Config:
    config = rl_grpo_qwen3_0_6b_varlen_no_compile(seq_len=1024)
    # Tied embeddings cannot be split across pipeline stages.
    config.model.enable_weight_tying = False  # pyrefly: ignore[missing-attribute]
    config.num_generators = 2
    config.trainer.parallelism.pipeline_parallel_degree = 2
    config.trainer.parallelism.num_pp_microbatches = 2
    config.trainer.parallelism.data_parallel_shard_degree = 2
    config.trainer.parallelism.tensor_parallel_degree = 1
    config.generator.parallelism.tensor_parallel_degree = 2
    return _configure_ci(
        config,
        steps=3,
        num_tokens_per_microbatch=2048,
        max_generated_tokens=256,
        use_hf_assets_from_env=True,
    )


def rl_grpo_moe_debug_tp4_ep4() -> Controller.Config:
    config = rl_grpo_gpt_oss_debug_varlen_no_compile(seq_len=1024)
    config.trainer.parallelism.data_parallel_shard_degree = 1
    config.trainer.parallelism.tensor_parallel_degree = 4
    config.trainer.parallelism.expert_parallel_degree = 4
    config.generator.parallelism.data_parallel_degree = 1
    config.generator.parallelism.tensor_parallel_degree = 4
    config.generator.parallelism.expert_parallel_degree = 4
    # TODO: use FULL CUDA graphs once GPT-OSS CI enables a
    # CUDA-graph-compatible distributed MoE token dispatcher.
    config.generator.cuda_graph.mode = "NONE"
    return _configure_ci(
        config,
        steps=5,
        num_tokens_per_microbatch=2048,
        max_generated_tokens=256,
        use_hf_assets_from_env=False,
    )


def rl_grpo_checkpoint_save() -> Controller.Config:
    config = rl_grpo_qwen3_0_6b_varlen_checkpoint_test(seq_len=1024)
    config.num_generators = 2
    config.trainer.parallelism.data_parallel_shard_degree = 2
    config.trainer.parallelism.tensor_parallel_degree = 1
    config.generator.parallelism.tensor_parallel_degree = 2
    return _configure_ci(
        config,
        steps=2,
        num_tokens_per_microbatch=2048,
        max_generated_tokens=256,
        use_hf_assets_from_env=True,
    )


def rl_grpo_checkpoint_resume() -> Controller.Config:
    config = rl_grpo_qwen3_0_6b_varlen_checkpoint_test(seq_len=1024)
    config.num_generators = 1
    config.trainer.parallelism.data_parallel_shard_degree = 1
    config.trainer.parallelism.tensor_parallel_degree = 2
    config.generator.parallelism.tensor_parallel_degree = 4
    return _configure_ci(
        config,
        steps=4,
        num_tokens_per_microbatch=2048,
        max_generated_tokens=256,
        use_hf_assets_from_env=True,
    )


def rl_grpo_0_6b_tp4_batch_invariant() -> Controller.Config:
    config = rl_grpo_qwen3_0_6b_varlen_batch_invariant(seq_len=1024)
    config.num_generators = 1
    config.trainer.parallelism.tensor_parallel_degree = 4
    config.generator.parallelism.tensor_parallel_degree = 4
    return _configure_ci(
        config,
        steps=3,
        num_tokens_per_microbatch=2048,
        max_generated_tokens=128,
        use_hf_assets_from_env=True,
    )


def rl_grpo_moe_debug_tp4_ep4_batch_invariant() -> Controller.Config:
    config = rl_grpo_qwen3_moe_debug_varlen_batch_invariant(seq_len=1024)
    return _configure_ci(
        config,
        steps=5,
        num_tokens_per_microbatch=2048,
        max_generated_tokens=256,
        use_hf_assets_from_env=False,
    )


def rl_grpo_qwen3_5_debug_tp2_batch_invariant() -> Controller.Config:
    config = rl_grpo_qwen3_5_debug_varlen_batch_invariant(seq_len=1024)
    config.num_generators = 2
    config.trainer.parallelism.data_parallel_shard_degree = 2
    config.trainer.parallelism.tensor_parallel_degree = 2
    config.generator.parallelism.tensor_parallel_degree = 2
    return _configure_ci(
        config,
        steps=3,
        num_tokens_per_microbatch=1024,
        max_generated_tokens=128,
        use_hf_assets_from_env=False,
    )


def rl_grpo_kimi_k3_debug_batch_invariant() -> Controller.Config:
    config = rl_grpo_kimi_k3_debug_varlen_batch_invariant(seq_len=1024)
    config.num_generators = 2
    config.trainer.parallelism.data_parallel_shard_degree = 2
    config.trainer.parallelism.tensor_parallel_degree = 1
    config.generator.parallelism.tensor_parallel_degree = 1
    return _configure_ci(
        config,
        steps=3,
        num_tokens_per_microbatch=1024,
        max_generated_tokens=128,
        use_hf_assets_from_env=False,
    )
