# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end smoke configuration for the Terminal-Bench example."""

from __future__ import annotations

import dataclasses

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.config import CompileConfig, ParallelismConfig
from torchtitan.experiments.rl.actors.generator import SamplingConfig
from torchtitan.experiments.rl.components.batcher import BatchConfig, Batcher
from torchtitan.experiments.rl.controller import (
    AsyncLoopConfig,
    Controller,
    ValidationConfig,
)
from torchtitan.experiments.rl.environment import TokenEnv
from torchtitan.experiments.rl.examples.alphabet_sort.config_registry import (
    rl_grpo_qwen3_0_6b_varlen,
)
from torchtitan.experiments.rl.examples.terminal_bench.data import TerminalBenchDataset
from torchtitan.experiments.rl.examples.terminal_bench.rollouter import (
    TerminalBenchRollouter,
)
from torchtitan.experiments.rl.models.vllm_registry import InferenceParallelismConfig
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor


def rl_grpo_qwen3_0_6b_terminal_bench_smoke() -> Controller.Config:
    """One-step, two-GPU Terminal-Bench smoke run using local Docker sandboxes."""
    config = rl_grpo_qwen3_0_6b_varlen()
    dataset = TerminalBenchDataset.Config(shuffle=False)

    config.dump_folder = "outputs/rl/terminal_bench"
    config.async_loop = AsyncLoopConfig(
        num_training_steps=1,
        num_prompts_per_train_step=1,
        num_samples_per_prompt=1,
        target_offpolicy_steps=0,
        window_fraction=None,
        validation=ValidationConfig(num_samples=0),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=1, seq_len=8192),
        ),
    )
    config.compile = CompileConfig(enable=False)
    config.rollouter = TerminalBenchRollouter.Config(
        train_dataset=dataset,
        validation_dataset=dataclasses.replace(dataset),
        token_env=TokenEnv.Config(
            max_rollout_tokens=8192,
            max_num_turns=8,
        ),
    )
    config.metrics = MetricsProcessor.Config(enable_wandb=False)
    config.num_generators = 1
    config.trainer = dataclasses.replace(
        config.trainer,
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=1,
            tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            enable=True,
            initial_load_in_hf=True,
            interval=1,
            last_save_model_only=True,
        ),
    )
    config.generator = dataclasses.replace(
        config.generator,
        parallelism=InferenceParallelismConfig(
            data_parallel_degree=1,
            tensor_parallel_degree=1,
        ),
        sampling=SamplingConfig(
            temperature=0.8,
            top_p=0.95,
            max_tokens=256,
        ),
    )
    return config
