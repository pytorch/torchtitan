# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Config entry points for the RL/unified experiment.

Each function returns a complete ``RLTrainer.Config`` and is discoverable by
``ConfigManager`` via ``--module rl --config <function_name>``.
"""

from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config.configs import (
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.rl.actors.generator import SamplingConfig, VLLMGenerator
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.grpo import GRPOLoss, RLTrainer
from torchtitan.experiments.rl.sum_digits import SumDigitsEnv
from torchtitan.models.qwen3 import model_registry


def rl_grpo_qwen3_0_6b() -> RLTrainer.Config:
    """GRPO training config for Qwen3-0.6B (6 GPUs: 4 gen + 2 train)."""
    group_size = 8
    return RLTrainer.Config(
        model_spec=model_registry("0.6B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        num_steps=10,
        num_prompts_per_step=5,
        num_validation_samples=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        env=SumDigitsEnv.Config(seed=42, correctness_reward=1.0, format_reward=0.3),
        validation_env=SumDigitsEnv.Config(
            seed=99, correctness_reward=1.0, format_reward=0.3
        ),
        trainer=PolicyTrainer.Config(
            optimizer=OptimizersContainer.Config(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
                disable_loss_parallel=True,
            ),
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=ParallelismConfig(
                tensor_parallel_degree=4,
                data_parallel_replicate_degree=1,
            ),
            sampling=SamplingConfig(
                n=group_size,
                temperature=0.8,
                top_p=0.95,
                max_tokens=100,
            ),
        ),
    )


def rl_grpo_qwen3_1_7b() -> RLTrainer.Config:
    """GRPO training config for Qwen3-1.7B (6 GPUs: 4 gen + 2 train)."""
    group_size = 8
    return RLTrainer.Config(
        model_spec=model_registry("1.7B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-1.7B",
        num_steps=10,
        num_prompts_per_step=5,
        num_validation_samples=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        env=SumDigitsEnv.Config(seed=42, correctness_reward=1.0, format_reward=0.3),
        validation_env=SumDigitsEnv.Config(
            seed=99, correctness_reward=1.0, format_reward=0.3
        ),
        trainer=PolicyTrainer.Config(
            optimizer=OptimizersContainer.Config(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
                disable_loss_parallel=True,
            ),
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=4,
                data_parallel_replicate_degree=1,
            ),
            sampling=SamplingConfig(
                n=group_size,
                temperature=0.8,
                top_p=0.95,
                max_tokens=100,
            ),
        ),
    )


def rl_grpo_qwen3_14b() -> RLTrainer.Config:
    """GRPO training config for Qwen3-14B (16 GPUs: 8 gen + 8 train)."""
    group_size = 8
    return RLTrainer.Config(
        model_spec=model_registry("14B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-14B",
        num_steps=10,
        num_prompts_per_step=5,
        num_validation_samples=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        env=SumDigitsEnv.Config(seed=42, correctness_reward=1.0, format_reward=0.3),
        validation_env=SumDigitsEnv.Config(
            seed=99, correctness_reward=1.0, format_reward=0.3
        ),
        trainer=PolicyTrainer.Config(
            optimizer=OptimizersContainer.Config(lr=1e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(dtype="bfloat16"),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=8,
                disable_loss_parallel=True,
            ),
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=ParallelismConfig(
                tensor_parallel_degree=8,
                data_parallel_replicate_degree=1,
            ),
            sampling=SamplingConfig(
                n=group_size,
                temperature=0.8,
                top_p=0.95,
                max_tokens=100,
            ),
        ),
    )


def rl_grpo_qwen3_debug() -> RLTrainer.Config:
    """Debug config for quick iteration -- small model, few steps (2 GPUs: 1 gen + 1 train)."""
    group_size = 4
    return RLTrainer.Config(
        model_spec=model_registry("debugmodel", attn_backend="varlen"),
        num_steps=5,
        num_prompts_per_step=5,
        num_validation_samples=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        env=SumDigitsEnv.Config(seed=42, correctness_reward=1.0, format_reward=0.3),
        validation_env=SumDigitsEnv.Config(
            seed=99, correctness_reward=1.0, format_reward=0.3
        ),
        trainer=PolicyTrainer.Config(
            optimizer=OptimizersContainer.Config(lr=8e-4),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=1,
                data_parallel_replicate_degree=1,
            ),
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            parallelism=ParallelismConfig(
                tensor_parallel_degree=1,
                data_parallel_replicate_degree=1,
            ),
            sampling=SamplingConfig(
                n=group_size,
                temperature=1.0,
                top_p=0.95,
                max_tokens=50,
            ),
        ),
    )


def rl_grpo_qwen3_0_6b_batch_invariant() -> RLTrainer.Config:
    """On-policy GRPO config for Qwen3-0.6B under same parallelism (4 GPUs: 2 gen + 2 train).

    Enables deterministic + batch-invariant mode for true on-policy RL training.
    """
    batch_invariant_config = DebugConfig(batch_invariant=True, deterministic=True)
    group_size = 8
    return RLTrainer.Config(
        model_spec=model_registry("0.6B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        num_steps=10,
        num_prompts_per_step=5,
        num_validation_samples=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        env=SumDigitsEnv.Config(seed=42, correctness_reward=1.0, format_reward=0.3),
        validation_env=SumDigitsEnv.Config(
            seed=99, correctness_reward=1.0, format_reward=0.3
        ),
        trainer=PolicyTrainer.Config(
            optimizer=OptimizersContainer.Config(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            # bfloat16 is needed for trainer to align with generator dtype
            # TODO: replace bfloat16 enablement with FSDP2+TP2
            training=TrainingConfig(dtype="bfloat16"),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            debug=batch_invariant_config,
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=ParallelismConfig(
                tensor_parallel_degree=2,
                data_parallel_replicate_degree=1,
            ),
            sampling=SamplingConfig(
                n=group_size,
                temperature=0.8,
                top_p=0.95,
                max_tokens=100,
            ),
            debug=batch_invariant_config,
        ),
    )
