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
from torchtitan.experiments.rl.actors.generator import (
    SamplingConfig,
    VLLMCudagraphConfig,
    VLLMGenerator,
)
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


def rl_grpo_qwen3_moe_debug_ep() -> RLTrainer.Config:
    """Debug MoE config with EP+TP on generator (4 GPUs: 2 gen + 2 train).

    Generator uses TP=2 for dense layers and EP=2 for MoE experts.
    The RL loop auto-rebuilds the model spec with AllToAllTokenDispatcher
    when generator EP > 1.

    Generate the debug checkpoint with:
        python scripts/create_debug_moe_ckpt.py
    """
    return RLTrainer.Config(
        model_spec=model_registry("debugmodel_moe", attn_backend="varlen"),
        hf_assets_path="/tmp/debug_moe_ckpt",
        num_steps=5,
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
                tensor_parallel_degree=2,
                data_parallel_replicate_degree=1,
                expert_parallel_degree=2,
            ),
            compile=CompileConfig(enable=False),
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            # Disable torch.compile + CUDA graph capture: the EP all-to-all
            # path issues an unpinned D2H copy of split sizes that the
            # piecewise/full graph capture rejects.
            cudagraph=VLLMCudagraphConfig(enable=False),
            parallelism=ParallelismConfig(
                tensor_parallel_degree=2,
                data_parallel_replicate_degree=1,
                expert_parallel_degree=2,
            ),
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=0.95,
                max_tokens=50,
            ),
        ),
    )


def rl_grpo_qwen3_moe_debug_ep_batch_invariant() -> RLTrainer.Config:
    """Batch-invariant MoE EP config for bitwise parity testing (8 GPUs).

    Trainer: TP=4, EP=4 (4 GPUs). Generator: TP=4, EP=4 (4 GPUs).

    Uses the bundled bootstrap directory ``tests/assets/qwen3_moe_debug``
    (config.json + Qwen3 tokenizer files, no weights) together with
    ``debug.random_init=True`` so no checkpoint is needed.
    """
    debug_config = DebugConfig(
        batch_invariant=True, deterministic=True, random_init=True
    )
    return RLTrainer.Config(
        model_spec=model_registry(
            "debugmodel_moe", attn_backend="varlen", moe_comm_backend="standard"
        ),
        hf_assets_path="tests/assets/qwen3_moe_debug",
        num_steps=5,
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
            training=TrainingConfig(dtype="bfloat16"),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=4,
                expert_parallel_degree=4,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            compile=CompileConfig(enable=False, backend="aot_eager"),
            debug=debug_config,
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            cudagraph=VLLMCudagraphConfig(enable=False),
            parallelism=ParallelismConfig(
                tensor_parallel_degree=4,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                expert_parallel_degree=4,
            ),
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=0.95,
                max_tokens=50,
            ),
            debug=debug_config,
        ),
    )


def rl_grpo_qwen3_30b_a3b() -> RLTrainer.Config:
    """GRPO training config for Qwen3-30B-A3B MoE (8 GPUs: 4 gen + 4 train).

    Generator uses TP=4 for dense layers and EP=4 for MoE experts.
    Trainer uses TP=4 for all layers.

    Note: Qwen3-30B-A3B has 4 KV heads, so TP degree cannot exceed 4.
    """
    return RLTrainer.Config(
        model_spec=model_registry("30B-A3B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-30B-A3B",
        num_steps=10,
        trainer=PolicyTrainer.Config(
            optimizer=OptimizersContainer.Config(lr=1e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(dtype="bfloat16"),
            parallelism=ParallelismConfig(
                tensor_parallel_degree=8,
                disable_loss_parallel=True,
            ),
            compile=CompileConfig(enable=True, backend="aot_eager"),
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            cudagraph=VLLMCudagraphConfig(enable=True),
            parallelism=ParallelismConfig(
                tensor_parallel_degree=8,
                data_parallel_replicate_degree=1,
                expert_parallel_degree=8,
            ),
            sampling=SamplingConfig(
                n=8,
                temperature=0.8,
                top_p=0.95,
                max_tokens=100,
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
