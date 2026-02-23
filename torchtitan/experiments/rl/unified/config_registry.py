# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Config entry points for the RL/unified experiment.

Each function returns a complete ``RLTrainer.Config`` and is discoverable by
``ConfigManager`` via ``--module rl.unified --config <function_name>``.

TODO: Once the config branch lands, replace the ``JobConfig`` sub-dataclass
imports (``Checkpoint``, ``Optimizer``, ``LRScheduler``, ``Training``,
``Parallelism``, ``ActivationCheckpoint``) with their config-branch
counterparts (``CheckpointManager.Config``, ``OptimizersContainer.Config``,
etc.) and replace ``from torchtitan.experiments.rl.unified.job_config import
JobConfig`` with ``from torchtitan.trainer import Trainer``.
"""

from torchtitan.config.job_config import (
    ActivationCheckpoint,
    Checkpoint,
    JobConfig,
    LRScheduler,
    Model,
    Optimizer,
    Parallelism,
    Training,
)

from torchtitan.experiments.rl.unified.actors.generator import Generator, VLLMEngine
from torchtitan.experiments.rl.unified.configs import (
    PolicyOptimizationConfig,
    RLTrainer,
    VLLMSamplingConfig,
)


def rl_grpo_qwen3_0_6b() -> RLTrainer.Config:
    """GRPO training config for Qwen3-0.6B."""
    return RLTrainer.Config(
        trainer=JobConfig(
            model=Model(
                name="qwen3",
                flavor="0.6B",
            ),
            optimizer=Optimizer(lr=1e-6),
            lr_scheduler=LRScheduler(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=Training(
                local_batch_size=4,
                seq_len=4096,
                steps=10,
            ),
            parallelism=Parallelism(
                tensor_parallel_degree=1,
                data_parallel_replicate_degree=2,
            ),
            checkpoint=Checkpoint(
                initial_load_path="/data/users/jianiw/model/qwen3-0.6b",
                initial_load_model_only=True,
                initial_load_in_hf=True,
            ),
            activation_checkpoint=ActivationCheckpoint(
                mode="selective",
                selective_ac_option="op",
            ),
        ),
        batch_invariant_mode=True,
        policy_optimization=PolicyOptimizationConfig(
            beta=0.1,
            group_size=8,
            use_stable=False,
        ),
        generator=Generator.Config(
            vllm_engine=VLLMEngine.Config(
                dtype="bfloat16",
                gpu_memory_limit=0.5,
                enforce_eager=True,
                seed=42,
                parallelism=Parallelism(
                    tensor_parallel_degree=1,
                ),
                sampling=VLLMSamplingConfig(
                    temperature=0.8,
                    top_p=0.95,
                    max_tokens=100,
                ),
            ),
            vllm_attention_backend="FLASH_ATTN",
        ),
    )


def rl_grpo_qwen3_debug() -> RLTrainer.Config:
    """Debug config for quick iteration — small model, few steps."""
    return RLTrainer.Config(
        trainer=JobConfig(
            model=Model(
                name="qwen3",
                flavor="debugmodel",
            ),
            optimizer=Optimizer(lr=8e-4),
            lr_scheduler=LRScheduler(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=Training(
                local_batch_size=2,
                seq_len=2048,
                steps=5,
            ),
            parallelism=Parallelism(
                tensor_parallel_degree=1,
                data_parallel_replicate_degree=1,
            ),
            checkpoint=Checkpoint(
                interval=5,
            ),
        ),
        batch_invariant_mode=False,
        policy_optimization=PolicyOptimizationConfig(
            beta=0.1,
            group_size=4,
            use_stable=False,
        ),
        generator=Generator.Config(
            vllm_engine=VLLMEngine.Config(
                gpu_memory_limit=0.3,
                enforce_eager=True,
                parallelism=Parallelism(
                    tensor_parallel_degree=1,
                ),
                sampling=VLLMSamplingConfig(
                    temperature=1.0,
                    max_tokens=50,
                ),
            ),
        ),
    )
