# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TerminalBench RL recipes backed by a Verifiers environment service."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.experiments.rl.actors.generator import (
    SamplingConfig,
    VLLMCudagraphConfig,
    VLLMGenerator,
)
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.components.batcher import BatchConfig, Batcher
from torchtitan.experiments.rl.controller import (
    AsyncLoopConfig,
    Controller,
    ValidationConfig,
)
from torchtitan.experiments.rl.examples.alphabet_sort.config_registry import (
    rl_grpo_qwen3_0_6b_varlen,
)
from torchtitan.experiments.rl.losses import GRPOLoss
from torchtitan.experiments.rl.models.vllm_registry import InferenceParallelismConfig
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.rollout.advantage import AdvantageEstimator
from torchtitan.experiments.rl.rollout.verifiers.dataset import VerifiersTaskDataset
from torchtitan.experiments.rl.rollout.verifiers.env_server import VerifiersEnvServer
from torchtitan.experiments.rl.rollout.verifiers.rollouter import (
    VerifiersRewardFn,
    VerifiersRollouter,
)
from torchtitan.experiments.rl.rubrics import Rubric
from torchtitan.models.qwen3 import model_registry


def rl_grpo_qwen3_8b_terminal_bench():
    """Qwen3-8B GRPO on TerminalBench through a Verifiers EnvServer."""
    context_tokens = 32768
    hf_assets_path = "torchtitan/experiments/rl/example_checkpoint/Qwen3-8B"
    return Controller.Config(
        model_spec=model_registry("8B", attn_backend="varlen"),
        hf_assets_path=hf_assets_path,
        verifiers_env_server=VerifiersEnvServer.Config(
            config_path=str(Path(__file__).with_name("verifiers_env.toml")),
            bind_address="tcp://127.0.0.1:0",
            startup_timeout_sec=600.0,
        ),
        async_loop=AsyncLoopConfig(
            num_training_steps=500,
            num_prompts_per_train_step=4,
            num_samples_per_prompt=4,
            validation=ValidationConfig(num_samples=32),
            batcher=Batcher.Config(
                batch=BatchConfig(local_batch_size=1, seq_len=context_tokens),
            ),
        ),
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=VerifiersRollouter.Config(
            train_dataset=VerifiersTaskDataset.Config(
                taskset_id="harbor",
                num_tasks=89,
                taskset_args={"dataset": "terminal-bench/terminal-bench-2"},
                seed=42,
            ),
            validation_dataset=VerifiersTaskDataset.Config(
                taskset_id="harbor",
                num_tasks=89,
                taskset_args={"dataset": "terminal-bench/terminal-bench-2"},
                seed=99,
                shuffle=False,
            ),
            rubric=Rubric.Config(
                reward_fns=[VerifiersRewardFn.Config(weight=1.0)],
                error_reward=0.0,
                truncation_reward=0.0,
            ),
            advantage=AdvantageEstimator.Config(should_std_normalize=True),
            model_name="Qwen/Qwen3-8B",
            renderer_model_name=hf_assets_path,
            renderer_name="qwen3",
            renderer_kwargs={
                "enable_thinking": True,
                "thinking_retention": "all",
            },
            renderer_multiplex=256,
            max_model_len=context_tokens,
            connection_timeout_sec=600.0,
        ),
        renderer=RendererConfig(
            name="qwen3",
            enable_thinking=True,
            preserve_all_thinking=True,
            preserve_thinking_between_tool_calls=True,
        ),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=1e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=4,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=50,
                last_save_model_only=False,
                keep_latest_k=3,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=GRPOLoss.Config(),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            gpu_memory_limit=0.6,
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=2,
            ),
            cudagraph=VLLMCudagraphConfig(enable=True),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=4096,
            ),
        ),
    )


def rl_grpo_qwen3_0_6b_verifiers_dummy():
    """Run the production RL path against a minimal Verifiers environment."""
    config = rl_grpo_qwen3_0_6b_varlen()
    # CI uses a lightweight server-side environment: the Verifiers null harness,
    # a subprocess runtime, and the local fixture taskset. This exercises the same
    # EnvServer and model-adapter path without requiring a Docker sandbox.
    fixture_config = (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "fixtures"
        / "verifiers_env.toml"
    )
    dataset = VerifiersTaskDataset.Config(
        taskset_id="torchtitan_verifiers_fixture",
        num_tasks=1,
        shuffle=False,
    )

    config.verifiers_env_server = VerifiersEnvServer.Config(
        config_path=str(fixture_config),
        bind_address="tcp://127.0.0.1:0",
        startup_timeout_sec=120.0,
    )
    config.rollouter = VerifiersRollouter.Config(
        train_dataset=dataset,
        validation_dataset=dataset,
        rubric=Rubric.Config(
            reward_fns=[VerifiersRewardFn.Config(weight=1.0)],
            error_reward=0.0,
            truncation_reward=0.0,
        ),
        advantage=AdvantageEstimator.Config(),
        model_name="Qwen/Qwen3-0.6B",
        renderer_model_name="Qwen/Qwen3-0.6B",
        renderer_name="qwen3",
        renderer_kwargs={"enable_thinking": False},
        renderer_multiplex=1,
        max_model_len=512,
        connection_timeout_sec=120.0,
    )
    config.renderer = RendererConfig(name="qwen3", enable_thinking=False)
    config.async_loop = dataclasses.replace(
        config.async_loop,
        num_training_steps=1,
        num_prompts_per_train_step=1,
        num_samples_per_prompt=1,
        target_offpolicy_steps=0,
        validation=ValidationConfig(num_samples=0),
        batcher=dataclasses.replace(
            config.async_loop.batcher,
            batch=dataclasses.replace(
                config.async_loop.batcher.batch,
                local_batch_size=1,
                seq_len=512,
            ),
        ),
    )
    config.trainer = dataclasses.replace(
        config.trainer,
        parallelism=dataclasses.replace(
            config.trainer.parallelism,
            data_parallel_shard_degree=1,
            tensor_parallel_degree=1,
        ),
        checkpoint=dataclasses.replace(config.trainer.checkpoint, enable=False),
    )
    config.generator = dataclasses.replace(
        config.generator,
        parallelism=dataclasses.replace(
            config.generator.parallelism,
            data_parallel_degree=1,
            tensor_parallel_degree=1,
        ),
        sampling=dataclasses.replace(
            config.generator.sampling,
            temperature=0.0,
            top_p=1.0,
            max_tokens=32,
        ),
    )
    config.compile = dataclasses.replace(config.compile, enable=False)
    config.metrics = dataclasses.replace(config.metrics, enable_wandb=False)
    return config
