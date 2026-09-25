# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DAPO math recipes using Verifiers for rollout orchestration."""

from __future__ import annotations

from typing import Literal

import verifiers.v1 as vf

from renderers import Qwen3RendererConfig
from verifiers.v1.harnesses.null import NullHarnessConfig as VerifiersNullHarnessConfig

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.components.renderer import from_renderers
from torchtitan.config import CompileConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import LMHeadCastConverter
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3 import model_registry
from torchtitan.rl.controller import AsyncLoopConfig, Controller, ValidationConfig
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
from torchtitan.rl.distributed.routing.inter_generator import InterGeneratorRouter
from torchtitan.rl.distributed.routing.strategies import LeastLoadedRoutingStrategy
from torchtitan.rl.examples.verifiers import (
    GenerationServer,
    RewardFromVerifiers,
    VerifiersEnvServer,
    VerifiersRollouter,
    VerifiersTaskDataset,
)
from torchtitan.rl.examples.verifiers.dapo_math.data import VerifiersMathTasksetConfig
from torchtitan.rl.examples.verifiers.data import register_local_taskset_alias
from torchtitan.rl.generator import SamplingConfig, VLLMCudaGraphConfig, VLLMGenerator
from torchtitan.rl.losses import DAPOLoss
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.rubric import Rubric
from torchtitan.rl.trainer import Trainer

# TODO: Enable CUDA graphs for RL trainers after eager/graph numerics parity is
# verified.


def _math_taskset_config(
    dataset: Literal["dapo_math", "aime2025"],
) -> VerifiersMathTasksetConfig:
    taskset_id = register_local_taskset_alias(VerifiersMathTasksetConfig.__module__)
    return VerifiersMathTasksetConfig(id=taskset_id, dataset=dataset)


def _verifiers_math_rollouter_config(
    *, max_rollout_tokens: int
) -> VerifiersRollouter.Config:
    return VerifiersRollouter.Config(
        train_dataset=VerifiersTaskDataset.Config(
            verifiers_taskset=_math_taskset_config("dapo_math"),
            seed=42,
        ),
        validation_dataset=VerifiersTaskDataset.Config(
            verifiers_taskset=_math_taskset_config("aime2025"),
            seed=99,
            shuffle=False,
        ),
        verifiers_env_server=VerifiersEnvServer.Config(
            environment=vf.SingleAgentEnvConfig(
                agent=vf.AgentConfig(
                    runtime=vf.SubprocessConfig(),
                    max_turns=1,
                    harness=VerifiersNullHarnessConfig(id="null"),
                ),
            ),
            serve=vf.ServeConfig(
                pool=vf.StaticPoolConfig(num_workers=1),
                address="tcp://127.0.0.1:0",
            ),
        ),
        rubric=Rubric.Config(
            reward_fns=[RewardFromVerifiers.Config(weight=1.0)],
            error_reward=0.0,
        ),
        generation_server=GenerationServer.Config(
            max_rollout_tokens=max_rollout_tokens
        ),
    )


def _qwen3_4b_verifiers_config(
    *,
    max_response_tokens: int,
    max_total_tokens: int,
    dump_folder: str,
) -> Controller.Config:
    """Build the Qwen3-4B DAPO-Math configuration using Verifiers."""
    num_validation_samples = 30
    model_config = model_registry(
        "4B",
        seq_len=max_total_tokens,
        attn_backend="varlen",
        converters=[LMHeadCastConverter.Config()],
    )
    return Controller.Config(
        model=model_config,
        hf_assets_path="torchtitan/rl/example_checkpoint/Qwen3-4B-Base",
        dump_folder=dump_folder,
        async_loop=AsyncLoopConfig(
            num_training_steps=150,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=16,
            target_offpolicy_steps=4,
            validation=ValidationConfig(num_samples=num_validation_samples),
        ),
        compile=CompileConfig(backend="aot_eager"),
        rollouter=_verifiers_math_rollouter_config(max_rollout_tokens=max_total_tokens),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=True)),
        num_generators=6,
        generator_router=InterGeneratorRouter.Config(
            strategy=LeastLoadedRoutingStrategy.Config()
        ),
        metrics=MetricsProcessor.Config(
            enable_wandb=True,
            console_log_keys_validation=[
                "validation_reward/_mean",
                "validation_reward/_max",
                "validation/response_length/mean",
                "timing/validate",
            ],
        ),
        trainer=Trainer.Config(
            optimizer=default_adamw(
                lr=1e-6,
                betas=(0.9, 0.98),
                weight_decay=0.1,
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=0,
                min_lr_factor=1.0,
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=max_total_tokens,
                max_context_length=max_total_tokens,
            ),
            parallelism=ParallelismConfig(
                data_parallel_replicate_degree=1,
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
            ),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=100,
                last_save_model_only=False,
                keep_latest_k=3,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=DAPOLoss.Config(
                    ratio_clip_low=0.2,
                    ratio_clip_high=0.28,
                    global_vocab_size=decoder_vocab_size(model_config),
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=1,
            ),
            cuda_graph=VLLMCudaGraphConfig(mode="FULL"),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=max_response_tokens,
            ),
        ),
    )


def rl_dapo_qwen3_4b_verifiers_8k() -> Controller.Config:
    """Run the DAPO 8K recipe with Verifiers managing math episodes."""
    return _qwen3_4b_verifiers_config(
        max_response_tokens=8192,
        max_total_tokens=10240,
        dump_folder="outputs/rl/qwen3_4b_verifiers_8k",
    )


def rl_dapo_qwen3_4b_verifiers_32k() -> Controller.Config:
    """Run the DAPO 32K recipe with Verifiers managing math episodes."""
    return _qwen3_4b_verifiers_config(
        max_response_tokens=32768,
        max_total_tokens=34816,
        dump_folder="outputs/rl/qwen3_4b_verifiers_32k",
    )
