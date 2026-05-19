# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config entrypoints for the RL experiment.

Each function returns a complete :class:`RLTrainer.Config` and is
discoverable via ``--module rl --config <function_name>``.
"""

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.rl.actors.generator import SamplingConfig, VLLMGenerator
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.alphabet_sort import (
    AlphabetSortBuilder,
    AlphabetSortDataset,
)
from torchtitan.experiments.rl.envs.token_env import TokenEnvConfig
from torchtitan.experiments.rl.grpo import GRPOLoss, RLTrainer
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.replay import ReplayBuffer
from torchtitan.experiments.rl.sum_digits import SumDigitsBuilder, SumDigitsDataset
from torchtitan.models.qwen3 import model_registry


def rl_grpo_qwen3_0_6b() -> RLTrainer.Config:
    """GRPO config for Qwen3-0.6B (6 GPUs: 4 gen + 2 train) — SumDigits smoke.

    10-step gate. Reward starts ~0.3, should reach ~0.7 by step 10.
    """
    return RLTrainer.Config(
        model_spec=model_registry("0.6B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        num_steps=10,
        num_prompts_per_step=5,
        rollout_group_size=8,
        num_rollout_tasks=4,
        max_rollout_turns=1,
        num_validation_samples=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        train_dataset=SumDigitsDataset.Config(seed=42),
        train_builder=SumDigitsBuilder.Config(),
        validation_dataset=SumDigitsDataset.Config(seed=99),
        validation_builder=SumDigitsBuilder.Config(),
        renderer=RendererConfig(name="auto"),
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
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=ParallelismConfig(
                tensor_parallel_degree=4,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                n=1,  # GRPO siblings come from rollout_group_size
                temperature=0.8,
                top_p=0.95,
                max_tokens=200,
            ),
        ),
        replay_buffer=ReplayBuffer.Config(
            batch_size=5,
            dp_size=1,
            max_policy_age=1,
            max_buffer_size=256,
        ),
    )


def rl_grpo_qwen3_1_7b() -> RLTrainer.Config:
    """GRPO config for Qwen3-1.7B (6 GPUs: 4 gen + 2 train) — SumDigits or AlphabetSort."""
    return RLTrainer.Config(
        model_spec=model_registry("1.7B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-1.7B",
        num_steps=10,
        num_prompts_per_step=5,
        rollout_group_size=8,
        num_rollout_tasks=4,
        max_rollout_turns=1,
        num_validation_samples=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        train_dataset=SumDigitsDataset.Config(seed=42),
        train_builder=SumDigitsBuilder.Config(),
        validation_dataset=SumDigitsDataset.Config(seed=99),
        validation_builder=SumDigitsBuilder.Config(),
        renderer=RendererConfig(name="auto"),
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
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=4,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                n=1,
                temperature=0.8,
                top_p=0.95,
                max_tokens=300,
            ),
        ),
        replay_buffer=ReplayBuffer.Config(
            batch_size=5,
            dp_size=1,
            max_policy_age=1,
            max_buffer_size=256,
        ),
    )


def rl_grpo_qwen3_0_6b_batch_invariant() -> RLTrainer.Config:
    """On-policy GRPO config for Qwen3-0.6B (4 GPUs: 2 gen + 2 train).

    Enables deterministic + batch-invariant mode for true on-policy RL.
    Trainer and generator use the same TP degree.
    """
    batch_invariant_config = DebugConfig(batch_invariant=True, deterministic=True)
    return RLTrainer.Config(
        model_spec=model_registry("0.6B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        num_steps=10,
        num_prompts_per_step=5,
        rollout_group_size=8,
        num_rollout_tasks=2,
        max_rollout_turns=1,
        num_validation_samples=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        train_dataset=SumDigitsDataset.Config(seed=42),
        train_builder=SumDigitsBuilder.Config(),
        validation_dataset=SumDigitsDataset.Config(seed=99),
        validation_builder=SumDigitsBuilder.Config(),
        renderer=RendererConfig(name="auto"),
        trainer=PolicyTrainer.Config(
            optimizer=OptimizersContainer.Config(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(dtype="bfloat16"),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            debug=batch_invariant_config,
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=ParallelismConfig(
                tensor_parallel_degree=2,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                n=1,
                temperature=0.8,
                top_p=0.95,
                max_tokens=200,
            ),
            debug=batch_invariant_config,
        ),
        replay_buffer=ReplayBuffer.Config(
            batch_size=5,
            dp_size=1,
            max_policy_age=0,
            max_buffer_size=128,
        ),
    )


def rl_grpo_qwen3_1_7b_alphabet() -> RLTrainer.Config:
    """AlphabetSort acceptance gate — Qwen3-1.7B, 100 steps, 4-GPU split.

    Multi-turn (3-5 turns/episode), reward 0.0-1.0 via Ratcliff/Obershelp
    similarity. Holistic-power mode (matches prime-rl's run config).
    Target: avg reward materially higher than the baseline; aim for
    >=0.5 by step 100.
    """
    return RLTrainer.Config(
        model_spec=model_registry("1.7B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-1.7B",
        num_steps=100,
        num_prompts_per_step=8,
        rollout_group_size=8,
        num_rollout_tasks=4,
        max_rollout_turns=8,
        num_validation_samples=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        train_dataset=AlphabetSortDataset.Config(
            seed=1337420,
            min_turns=3,
            max_turns=5,
            min_names_per_turn=1,
            max_names_per_turn=5,
        ),
        train_builder=AlphabetSortBuilder.Config(
            similarity_power=4,
            power_per_turn=False,
        ),
        validation_dataset=AlphabetSortDataset.Config(
            seed=99,
            min_turns=3,
            max_turns=3,
            min_names_per_turn=1,
            max_names_per_turn=4,
        ),
        validation_builder=AlphabetSortBuilder.Config(
            similarity_power=8,
            power_per_turn=False,
        ),
        renderer=RendererConfig(name="auto"),
        token_env=TokenEnvConfig(
            failed_parse_reward=0.0,
            context_overflow_reward=0.0,
            max_trajectory_tokens=2048,
            max_generation_tokens=768,
        ),
        trainer=PolicyTrainer.Config(
            optimizer=OptimizersContainer.Config(lr=1e-5),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=4,
                decay_type="linear",
            ),
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=50,
                last_save_model_only=False,
            ),
            loss=GRPOLoss.Config(clip_eps=0.2),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                n=1,
                temperature=0.8,
                top_p=0.95,
                max_tokens=768,
            ),
        ),
        replay_buffer=ReplayBuffer.Config(
            batch_size=8,
            dp_size=1,
            max_policy_age=1,
            max_buffer_size=512,
        ),
    )
