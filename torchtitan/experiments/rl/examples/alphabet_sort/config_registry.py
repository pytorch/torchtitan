# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config entry points for the alphabet-sort example.

Each function returns a complete ``RLTrainer.Config``, discoverable by
``ConfigManager`` via
``--module torchtitan.experiments.rl.examples.alphabet_sort --config rl_grpo_qwen3_*``.
"""

import dataclasses

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import (
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
from torchtitan.experiments.rl.batch_invariance import BatchInvariantFlexConverter
from torchtitan.experiments.rl.batcher import BatchConfig, Batcher
from torchtitan.experiments.rl.examples.alphabet_sort import AlphabetSortRollouter
from torchtitan.experiments.rl.generator_router import (
    GeneratorRouter,
    LeastLoadedRoutingStrategy,
    StickySessionRoutingStrategy,
)
from torchtitan.experiments.rl.losses import GRPOLoss
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.trainer import RLTrainer
from torchtitan.models.qwen3 import model_registry

_BATCH_INVARIANT_DEBUG = DebugConfig(batch_invariant=True, deterministic=True)


def rl_grpo_qwen3_0_6b_varlen() -> RLTrainer.Config:
    """GRPO training config for Qwen3-0.6B (6 GPUs: 4 gen + 2 train)."""
    group_size = 8
    return RLTrainer.Config(
        model_spec=model_registry("0.6B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        num_steps=10,
        num_groups_per_rollout_batch=5,
        num_validation_samples=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=AlphabetSortRollouter.Config(),
        group_size=group_size,
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        generator_router=GeneratorRouter.Config(
            strategy=StickySessionRoutingStrategy.Config(
                fallback_strategy=LeastLoadedRoutingStrategy.Config()
            )
        ),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=2, global_batch_size=8, seq_len=2048),
        ),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=2e-6),
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
                temperature=0.8,
                top_p=0.95,
                max_tokens=700,
            ),
        ),
    )


def rl_grpo_qwen3_0_6b_flex() -> RLTrainer.Config:
    """GRPO training config for Qwen3-0.6B with flex attention (4 GPUs: 2 gen + 2 train)."""
    group_size = 8
    return RLTrainer.Config(
        model_spec=model_registry("0.6B", attn_backend="flex"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        num_steps=10,
        num_groups_per_rollout_batch=5,
        num_validation_samples=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=AlphabetSortRollouter.Config(),
        group_size=group_size,
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=2, global_batch_size=8, seq_len=2048),
        ),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(dtype="bfloat16"),
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
                tensor_parallel_degree=2,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=100,
            ),
        ),
    )


def rl_grpo_qwen3_0_6b_flex_batch_invariant() -> RLTrainer.Config:
    """GRPO training config for Qwen3-0.6B with flex attention and batch invariance
    for bitwise-identical numerics between trainer and generator (4 GPUs: 2 gen + 2 train).
    """
    config = rl_grpo_qwen3_0_6b_flex()
    config.model_spec = model_registry(
        "0.6B",
        attn_backend="flex",
        converters=[BatchInvariantFlexConverter.Config()],
    )
    block_size = config.model_spec.model.layers[0].attention.inner_attention.block_size
    config.batcher = dataclasses.replace(
        config.batcher, per_sample_pad_multiple=block_size
    )
    config.trainer = dataclasses.replace(
        config.trainer,
        debug=_BATCH_INVARIANT_DEBUG,
        parallelism=dataclasses.replace(
            config.trainer.parallelism, enable_sequence_parallel=False
        ),
    )
    config.generator = dataclasses.replace(
        config.generator, debug=_BATCH_INVARIANT_DEBUG
    )
    return config


def rl_grpo_qwen3_1_7b() -> RLTrainer.Config:
    """GRPO training config for Qwen3-1.7B (6 GPUs: 4 gen + 2 train)."""
    group_size = 8
    return RLTrainer.Config(
        model_spec=model_registry("1.7B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-1.7B",
        num_steps=10,
        num_groups_per_rollout_batch=5,
        num_validation_samples=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=AlphabetSortRollouter.Config(),
        group_size=group_size,
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=2, global_batch_size=8, seq_len=2048),
        ),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=2e-6),
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
                temperature=0.8,
                top_p=0.95,
                max_tokens=700,
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
        num_groups_per_rollout_batch=5,
        num_validation_samples=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=AlphabetSortRollouter.Config(),
        group_size=group_size,
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=2, global_batch_size=8, seq_len=2048),
        ),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=1e-6),
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
                tensor_parallel_degree=8,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=700,
            ),
        ),
    )


def rl_grpo_qwen3_moe_debug_varlen() -> RLTrainer.Config:
    """Debug MoE config with EP+TP on generator (8 GPUs: 4 gen + 4 train).

    Generator uses TP=4 for dense layers and EP=4 for MoE experts.
    """
    group_size = 8
    return RLTrainer.Config(
        model_spec=model_registry("debugmodel_moe", attn_backend="varlen"),
        hf_assets_path="tests/assets/tokenizer",
        num_steps=5,
        num_groups_per_rollout_batch=5,
        num_validation_samples=20,
        # MoE EP all-to-all path issues unpinned D2H copies that block
        # torch.compile and CUDA graph capture; disable both.
        compile=CompileConfig(enable=False),
        rollouter=AlphabetSortRollouter.Config(),
        group_size=group_size,
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=2, global_batch_size=8, seq_len=2048),
        ),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=8e-4),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=4,
                data_parallel_replicate_degree=1,
                disable_loss_parallel=True,
                expert_parallel_degree=4,
            ),
            checkpoint=CheckpointManager.Config(
                enable=False,
                interval=10,
                last_save_model_only=False,
            ),
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            # Disable torch.compile + CUDA graph capture: the EP all-to-all
            # path issues an unpinned D2H copy of split sizes that the
            # piecewise/full graph capture rejects.
            cudagraph=VLLMCudagraphConfig(enable=False),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=4,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
                expert_parallel_degree=4,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=0.95,
                max_tokens=50,
            ),
        ),
    )


def rl_grpo_qwen3_moe_debug_varlen_batch_invariant() -> RLTrainer.Config:
    """Batch-invariant MoE EP config for bitwise parity testing (8 GPUs).

    Trainer: TP=4, EP=4 (4 GPUs). Generator: TP=4, EP=4 (4 GPUs).
    """
    group_size = 8
    return RLTrainer.Config(
        model_spec=model_registry(
            "debugmodel_moe", attn_backend="varlen", moe_comm_backend="standard"
        ),
        hf_assets_path="tests/assets/tokenizer",
        num_steps=10,
        num_groups_per_rollout_batch=5,
        num_validation_samples=20,
        # MoE EP all-to-all path issues unpinned D2H copies that block
        # torch.compile and CUDA graph capture; disable both.
        compile=CompileConfig(enable=False),
        rollouter=AlphabetSortRollouter.Config(),
        group_size=group_size,
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=2, global_batch_size=8, seq_len=2048),
        ),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=8e-4),
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
            checkpoint=CheckpointManager.Config(
                enable=False,
                interval=10,
                last_save_model_only=False,
            ),
            debug=_BATCH_INVARIANT_DEBUG,
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            cudagraph=VLLMCudagraphConfig(enable=False),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=4,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
                expert_parallel_degree=4,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=0.95,
                max_tokens=50,
            ),
            debug=_BATCH_INVARIANT_DEBUG,
        ),
    )


def rl_grpo_qwen3_30b_a3b_varlen() -> RLTrainer.Config:
    """GRPO training config for Qwen3-30B-A3B MoE (8 GPUs: 4 gen + 4 train).

    Generator uses TP=4 for dense layers and EP=4 for MoE experts.
    Trainer uses TP=4 for all layers.

    Note: Qwen3-30B-A3B has 4 KV heads, so TP degree cannot exceed 4.
    """
    group_size = 8
    return RLTrainer.Config(
        model_spec=model_registry("30B-A3B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-30B-A3B",
        num_steps=10,
        num_groups_per_rollout_batch=5,
        num_validation_samples=20,
        compile=CompileConfig(enable=False),
        rollouter=AlphabetSortRollouter.Config(),
        group_size=group_size,
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=2, global_batch_size=8, seq_len=2048),
        ),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=1e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(dtype="bfloat16"),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=4,
                disable_loss_parallel=True,
                expert_parallel_degree=4,
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
            cudagraph=VLLMCudagraphConfig(enable=False),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=4,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
                expert_parallel_degree=4,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=700,
            ),
        ),
    )


def rl_grpo_qwen3_0_6b_varlen_batch_invariant() -> RLTrainer.Config:
    """On-policy GRPO config for Qwen3-0.6B (4 GPUs: 2 gen + 2 train).

    Enables deterministic + batch-invariant mode for true on-policy RL training.
    """
    batch_invariant_config = DebugConfig(batch_invariant=True, deterministic=True)
    group_size = 8
    return RLTrainer.Config(
        model_spec=model_registry("0.6B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        num_steps=5,
        num_groups_per_rollout_batch=5,
        num_validation_samples=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=AlphabetSortRollouter.Config(),
        group_size=group_size,
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=2, global_batch_size=8, seq_len=2048),
        ),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=2e-6),
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
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=700,
            ),
            debug=batch_invariant_config,
        ),
    )
