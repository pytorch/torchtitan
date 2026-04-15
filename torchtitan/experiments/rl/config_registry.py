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
    GeneratorCompileConfig,
    SamplingConfig,
    VLLMGenerator,
)
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.simple_grpo_sum_digits import GRPOLoss, RLTrainer
from torchtitan.models.qwen3 import model_registry


def rl_grpo_qwen3_0_6b() -> RLTrainer.Config:
    """GRPO training config for Qwen3-0.6B (6 GPUs: 4 gen + 2 train)."""
    model_spec = model_registry("0.6B_varlen")
    return RLTrainer.Config(
        model_spec=model_spec,
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        num_steps=10,
        trainer=PolicyTrainer.Config(
            optimizer=OptimizersContainer.Config(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                tensor_parallel_degree=2,
            ),
            compile=CompileConfig(enable=True, backend="aot_eager"),
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            compile=GeneratorCompileConfig(
                backend="eager",
                cudagraph_mode="piecewise",
            ),
            parallelism=ParallelismConfig(
                tensor_parallel_degree=4,
                data_parallel_replicate_degree=1,
            ),
            num_samples_per_prompt=8,
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=100,
            ),
        ),
    )


def rl_grpo_qwen3_1_7b() -> RLTrainer.Config:
    """GRPO training config for Qwen3-1.7B (6 GPUs: 4 gen + 2 train)."""
    model_spec = model_registry("1.7B_varlen")
    return RLTrainer.Config(
        model_spec=model_spec,
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-1.7B",
        num_steps=10,
        trainer=PolicyTrainer.Config(
            optimizer=OptimizersContainer.Config(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                tensor_parallel_degree=2,
            ),
            compile=CompileConfig(enable=True, backend="aot_eager"),
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            compile=GeneratorCompileConfig(
                backend="eager",
                cudagraph_mode="piecewise",
            ),
            parallelism=ParallelismConfig(
                tensor_parallel_degree=4,
                data_parallel_replicate_degree=1,
            ),
            num_samples_per_prompt=8,
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=100,
            ),
        ),
    )


def rl_grpo_qwen3_debug() -> RLTrainer.Config:
    """Debug config for quick iteration -- small model, few steps (2 GPUs: 1 gen + 1 train)."""
    model_spec = model_registry("debugmodel_varlen")
    return RLTrainer.Config(
        model_spec=model_spec,
        num_steps=5,
        trainer=PolicyTrainer.Config(
            optimizer=OptimizersContainer.Config(lr=8e-4),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                tensor_parallel_degree=1,
                data_parallel_replicate_degree=1,
            ),
            compile=CompileConfig(enable=True, backend="aot_eager"),
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            compile=GeneratorCompileConfig(
                backend="eager",
                cudagraph_mode="piecewise",
            ),
            parallelism=ParallelismConfig(
                tensor_parallel_degree=1,
                data_parallel_replicate_degree=1,
            ),
            num_samples_per_prompt=4,
            sampling=SamplingConfig(
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
    model_spec = model_registry("0.6B_varlen")
    batch_invariant_config = DebugConfig(batch_invariant=True, deterministic=True)
    return RLTrainer.Config(
        model_spec=model_spec,
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        num_steps=10,
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
                tensor_parallel_degree=2,
            ),
            compile=CompileConfig(enable=True, backend="aot_eager"),
            debug=batch_invariant_config,
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            compile=GeneratorCompileConfig(
                backend="eager",
                cudagraph_mode="piecewise",
            ),
            parallelism=ParallelismConfig(
                tensor_parallel_degree=2,
                data_parallel_replicate_degree=1,
            ),
            num_samples_per_prompt=8,
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=100,
            ),
            debug=batch_invariant_config,
        ),
    )
