# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5 terminal-agent recipes using Verifiers and TitanRL."""

import os
from dataclasses import replace
from pathlib import Path

from renderers import Qwen35RendererConfig

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.components.renderer import from_renderers
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.config.transform import LMHeadCastConverter
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3_5 import model_registry
from torchtitan.rl.controller import AsyncLoopConfig, ValidationConfig
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
from torchtitan.rl.distributed.routing.inter_generator import InterGeneratorRouter
from torchtitan.rl.distributed.routing.strategies import (
    LeastLoadedRoutingStrategy,
    StickySessionRoutingStrategy,
)
from torchtitan.rl.experiments.verifiers.terminal_bench.controller import (
    TerminalBenchController,
)
from torchtitan.rl.experiments.verifiers.terminal_bench.rollouter import (
    terminal_bench_rollouter_config,
)
from torchtitan.rl.generator import SamplingConfig, VLLMCudaGraphConfig, VLLMGenerator
from torchtitan.rl.losses import GRPOLoss
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.trainer import Trainer


def _tasks_root(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Set {name} to a frozen Harbor task-tree directory")
    return Path(value)


def _image_overrides_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def _terminal_agent_config(
    *, train_tasks_root: Path, eval_tasks_root: Path, eval_only: bool
) -> TerminalBenchController.Config:
    max_context_length = 65536
    model_config = model_registry(
        "9B",
        seq_len=max_context_length,
        attn_backend="varlen",
        converters=[LMHeadCastConverter.Config()],
    )
    return TerminalBenchController.Config(
        eval_only=eval_only,
        model=model_config,
        hf_assets_path="torchtitan/rl/example_checkpoint/Qwen3.5-9B",
        dump_folder="outputs/rl/qwen35_9b_terminal_bench",
        async_loop=AsyncLoopConfig(
            num_training_steps=0 if eval_only else 100,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=32,
            target_offpolicy_steps=4,
            validation=ValidationConfig(num_samples=89),
        ),
        rollouter=terminal_bench_rollouter_config(
            train_tasks_root,
            eval_tasks_root,
            train_images_path=(
                _image_overrides_path("TERMINAL_BENCH_EVAL_IMAGES")
                if eval_only
                else _image_overrides_path("TERMINAL_BENCH_TRAIN_IMAGES")
            ),
            validation_images_path=_image_overrides_path("TERMINAL_BENCH_EVAL_IMAGES"),
            eval_only=eval_only,
        ),
        renderer=from_renderers(
            Qwen35RendererConfig(
                enable_thinking=True,
                thinking_retention="all",
            )
        ),
        num_generators=8,
        generator_router=InterGeneratorRouter.Config(
            strategy=StickySessionRoutingStrategy.Config(
                fallback_strategy=LeastLoadedRoutingStrategy.Config()
            )
        ),
        metrics=MetricsProcessor.Config(
            console_log_keys_validation=[
                "validation_reward/_mean",
                "validation_reward/_max",
                "timing/validate",
            ],
        ),
        trainer=Trainer.Config(
            optimizer=default_adamw(
                lr=1e-6,
                betas=(0.9, 0.999),
                weight_decay=0.0,
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=0,
                min_lr_factor=1.0,
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=max_context_length,
                max_context_length=max_context_length,
                dtype="float32",
            ),
            parallelism=ParallelismConfig(
                data_parallel_replicate_degree=1,
                data_parallel_shard_degree=8,
                tensor_parallel_degree=1,
            ),
            activation_checkpoint=FullAC.Config(),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=20,
                keep_latest_k=3,
                async_mode="async",
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=32,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(model_config)
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            cuda_graph=VLLMCudaGraphConfig(mode="FULL_DECODE_ONLY"),
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=1,
            ),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=16384,
            ),
        ),
    )


def rl_grpo_qwen35_9b_terminal_bench() -> TerminalBenchController.Config:
    """Train on frozen terminal tasks and validate on Terminal-Bench 2.1."""
    return _terminal_agent_config(
        train_tasks_root=_tasks_root("TERMINAL_BENCH_TRAIN_TASKS_ROOT"),
        eval_tasks_root=_tasks_root("TERMINAL_BENCH_EVAL_TASKS_ROOT"),
        eval_only=False,
    )


def rl_grpo_qwen35_9b_terminal_bench_eval() -> TerminalBenchController.Config:
    """Score the 89 Terminal-Bench 2.1 tasks without optimizer steps."""
    eval_root = _tasks_root("TERMINAL_BENCH_EVAL_TASKS_ROOT")
    config = _terminal_agent_config(
        train_tasks_root=eval_root,
        eval_tasks_root=eval_root,
        eval_only=True,
    )
    checkpoint = os.environ.get("TERMINAL_BENCH_CHECKPOINT")
    if checkpoint:
        config.trainer = replace(
            config.trainer,
            checkpointer=replace(
                config.trainer.checkpointer,
                initial_load_path=checkpoint,
                initial_load_in_hf=False,
                initial_load_model_only=True,
            ),
        )
    return config
