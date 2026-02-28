# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Config entry points for the RL/unified experiment.

Each function returns a complete ``RLTrainer.Config`` and is discoverable by
``ConfigManager`` via ``--module rl.unified --config <function_name>``.
"""

from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config.configs import ParallelismConfig, TrainingConfig
from torchtitan.experiments.rl.unified.actors.generator import (
    SamplingConfig,
    VLLMGenerator,
)
from torchtitan.experiments.rl.unified.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.unified.simple_grpo import RLTrainer
from torchtitan.models.qwen3 import model_registry


def rl_grpo_qwen3_0_6b() -> RLTrainer.Config:
    """GRPO training config for Qwen3-0.6B."""
    return RLTrainer.Config(
        model_spec=model_registry("0.6B"),
        hf_assets_path="/data/users/jianiw/model/qwen3-0.6b",
        num_steps=10,
        batch_invariant_mode=True,
        trainer=PolicyTrainer.Config(
            optimizer=OptimizersContainer.Config(lr=1e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(
                local_batch_size=4,
                seq_len=4096,
            ),
            parallelism=ParallelismConfig(
                tensor_parallel_degree=2,
                data_parallel_replicate_degree=1,
            ),
            hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            gpu_memory_limit=0.5,
            enforce_eager=True,
            parallelism=ParallelismConfig(
                tensor_parallel_degree=2,
            ),
            num_samples_per_prompt=8,
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=100,
            ),
            attention_backend="FLASH_ATTN",
        ),
    )


def rl_grpo_qwen3_debug() -> RLTrainer.Config:
    """Debug config for quick iteration — small model, few steps."""
    return RLTrainer.Config(
        model_spec=model_registry("debugmodel"),
        num_steps=5,
        batch_invariant_mode=False,
        trainer=PolicyTrainer.Config(
            optimizer=OptimizersContainer.Config(lr=8e-4),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(
                local_batch_size=2,
                seq_len=2048,
            ),
            parallelism=ParallelismConfig(
                tensor_parallel_degree=1,
                data_parallel_replicate_degree=1,
            ),
        ),
        generator=VLLMGenerator.Config(
            gpu_memory_limit=0.3,
            enforce_eager=True,
            parallelism=ParallelismConfig(
                tensor_parallel_degree=1,
            ),
            num_samples_per_prompt=4,
            sampling=SamplingConfig(
                temperature=1.0,
                max_tokens=50,
            ),
        ),
    )
