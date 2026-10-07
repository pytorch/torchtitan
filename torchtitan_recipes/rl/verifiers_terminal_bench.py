# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5 terminal-agent recipes using Verifiers and TitanRL.

Both recipes train on ``local/tmax@v1``, the TMax tasks exported by
``prepare_tmax.py`` into ``~/.cache/harbor/local_tmax_v1``, and validate on the
Terminal-Bench 2.1 Harbor dataset, which the ``harbor`` CLI downloads into
``~/.cache/harbor`` on first use. Edit the ids below to use other datasets.
"""

import math

import verifiers.v1 as vf

from renderers import Qwen35RendererConfig

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.renderer import from_renderers
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import LMHeadCastConverter
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.rl.controller import AsyncLoopConfig, Controller, ValidationConfig
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
from torchtitan.rl.distributed.routing.inter_generator import InterGeneratorRouter
from torchtitan.rl.distributed.routing.strategies import (
    LeastLoadedRoutingStrategy,
    StickySessionRoutingStrategy,
)
from torchtitan.rl.examples.verifiers import (
    GenerationServer,
    RewardFromVerifiers,
    VerifiersEnvServer,
    VerifiersRollouter,
    VerifiersTaskDataset,
)
from torchtitan.rl.examples.verifiers.data import register_local_taskset_alias
from torchtitan.rl.examples.verifiers.terminal_bench.harness import (
    register_harness_alias,
    TerminalBenchTerminusHarnessConfig,
)
from torchtitan.rl.examples.verifiers.terminal_bench.taskset import (
    TerminalTasksetConfig,
)
from torchtitan.rl.generator import SamplingConfig, VLLMCudaGraphConfig, VLLMGenerator
from torchtitan.rl.losses import GRPOLoss
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.observability.rollout_recorder import (
    KeepExtremeRewardsFilter,
    RolloutSampleRecorder,
)
from torchtitan.rl.rubric import Rubric
from torchtitan.rl.trainer import Trainer
from verifiers.v1.configs.agent import TimeoutConfig as AgentTimeoutConfig
from verifiers.v1.tasksets.harbor import HarborEnvConfig

_ENV_SERVER_WORKERS = 16


def _terminal_bench_rollouter_config(
    train_dataset: str,
    validation_dataset: str,
    *,
    max_context_length: int,
    max_turns: int,
    max_concurrent_rollouts: int,
) -> VerifiersRollouter.Config:
    """Select Harbor datasets by id.

    ``max_context_length`` is the generator's sequence length; the generation
    server caps each rollout at it. ``max_turns`` is the agent turn limit,
    which Verifiers enforces. ``max_concurrent_rollouts`` sizes the env server;
    set it to the number of rollouts the controller keeps in flight, or the
    excess queues in the env server and the generators idle.
    """
    if train_dataset == validation_dataset:
        raise ValueError(
            "Training and Terminal-Bench evaluation must use different datasets"
        )

    taskset_id = register_local_taskset_alias(TerminalTasksetConfig.__module__)
    return VerifiersRollouter.Config(
        train_dataset=VerifiersTaskDataset.Config(
            verifiers_taskset=TerminalTasksetConfig(
                id=taskset_id, dataset=train_dataset
            ),
            seed=42,
            shuffle=True,
        ),
        validation_dataset=VerifiersTaskDataset.Config(
            verifiers_taskset=TerminalTasksetConfig(
                id=taskset_id, dataset=validation_dataset
            ),
            seed=99,
            shuffle=False,
        ),
        verifiers_env_server=VerifiersEnvServer.Config(
            environment=HarborEnvConfig(
                agent=vf.AgentConfig(
                    harness=TerminalBenchTerminusHarnessConfig(
                        id=register_harness_alias(), version="0.22.0"
                    ),
                    runtime=vf.DockerConfig(),
                    max_turns=max_turns,
                    timeout=AgentTimeoutConfig(
                        setup=600,
                        rollout=7200,
                        scoring=12000,
                    ),
                ),
            ),
            serve=vf.ServeConfig(
                pool=vf.StaticPoolConfig(num_workers=_ENV_SERVER_WORKERS),
                max_concurrent=math.ceil(max_concurrent_rollouts / _ENV_SERVER_WORKERS),
                address="tcp://127.0.0.1:0",
            ),
        ),
        rubric=Rubric.Config(
            reward_fns=[RewardFromVerifiers.Config(weight=1.0)],
            error_reward=0.0,
        ),
        generation_server=GenerationServer.Config(
            max_rollout_tokens=max_context_length
        ),
        connection_timeout_sec=1800.0,
    )


def rl_grpo_qwen35_9b_terminal_bench() -> Controller.Config:
    """Qwen3.5-9B: train on TMax (``local/tmax@v1``), validate on Terminal-Bench 2.1.

    16 GPUs: 8 trainer (FSDP=8) and 8 one-GPU generators.
    """
    # Agent turns average about 1.1K tokens (completion plus terminal output),
    # so 120 turns need about 128K.
    max_context_length = 131072
    max_output_tokens = 16384
    async_loop = AsyncLoopConfig(
        num_training_steps=100,
        num_prompts_per_train_step=12,
        num_samples_per_prompt=32,
        target_offpolicy_steps=3,
        validation=ValidationConfig(num_samples=89),
    )
    model_config = build_model_config(
        "9B",
        seq_len=max_context_length,
        attn_backend="varlen",
        converters=[LMHeadCastConverter.Config()],
    )
    return Controller.Config(
        model=model_config,
        hf_assets_path="torchtitan/rl/example_checkpoint/Qwen3.5-9B",
        dump_folder="outputs/rl/qwen35_9b_terminal_bench",
        rollout_recorder=RolloutSampleRecorder.Config(
            filter=KeepExtremeRewardsFilter.Config(keep_errors=True)
        ),
        async_loop=async_loop,
        rollouter=_terminal_bench_rollouter_config(
            train_dataset="local/tmax@v1",
            validation_dataset="terminal-bench/terminal-bench-2-1",
            max_context_length=max_context_length,
            max_turns=120,
            max_concurrent_rollouts=async_loop.max_active_rollout_groups
            * async_loop.num_samples_per_prompt,
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
            optim=Optim.Config(
                optimizer=OptimizersContainer.Config(
                    optimizers=[
                        AdamW.Config(
                            pattern=r".*",
                            lr=1e-6,
                            betas=(0.9, 0.999),
                            weight_decay=0.0,
                        )
                    ]
                ),
                lr_scheduler=LRSchedulersContainer.Config(
                    warmup_steps=0,
                    min_lr_factor=1.0,
                ),
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
                max_tokens=max_output_tokens,
            ),
        ),
    )


def rl_grpo_qwen35_35b_a3b_terminal_bench() -> Controller.Config:
    """Qwen3.5-35B-A3B: train on TMax (``local/tmax@v1``), validate on Terminal-Bench 2.1.

    16 GPUs: 8 trainer (FSDP=4, TP=2, EP=8) and 2 generators of 4 GPUs
    (DP=2, TP=2, EP=4).

    The layout is constrained from both sides. The model has 2 KV heads, so TP
    is at most 2 in either role. Trainer EP must be at least TP, divide the 256
    experts and divide ``dp_shard * tp``; EP=8 spans the whole sparse region, so
    each rank holds 32 experts. The generator's DP axis only supplies ranks for
    expert parallelism, so its EP equals DP x TP; 256 experts over 4 ranks is 64
    each.

    The trainer keeps fp32 master weights, the default. That is about 70 GB of
    model states per GPU across 8 GPUs before activations, so it needs GPUs with
    well over 80 GB of memory.

    Generator CUDA graphs are off because of the standard all-to-all MoE token
    dispatcher, not DistMoE. That dispatcher copies the split sizes to the host,
    which CUDA graph capture does not allow
    ("Cannot copy between CPU and CUDA tensors during CUDA graph capture"). Turn
    capture back on together with a dispatcher that avoids the host read, such as
    HybridEP with ``non_blocking_capacity_factor``.

    TODO: migrate to DistMoE and capture generator CUDA graphs in ``FULL`` mode.
    """
    max_context_length = 65536
    max_output_tokens = 16384
    async_loop = AsyncLoopConfig(
        num_training_steps=100,
        num_prompts_per_train_step=8,
        num_samples_per_prompt=32,
        target_offpolicy_steps=4,
        validation=ValidationConfig(num_samples=89),
    )
    # TODO: update to distMoE model config
    model_config = build_model_config(
        "35B-A3B",
        seq_len=max_context_length,
        attn_backend="varlen",
        converters=[LMHeadCastConverter.Config()],
    )
    return Controller.Config(
        model=model_config,
        hf_assets_path="torchtitan/rl/example_checkpoint/Qwen3.5-35B-A3B",
        dump_folder="outputs/rl/qwen35_35b_a3b_terminal_bench",
        rollout_recorder=RolloutSampleRecorder.Config(
            filter=KeepExtremeRewardsFilter.Config(keep_errors=True)
        ),
        async_loop=async_loop,
        rollouter=_terminal_bench_rollouter_config(
            train_dataset="local/tmax@v1",
            validation_dataset="terminal-bench/terminal-bench-2-1",
            max_context_length=max_context_length,
            max_turns=120,
            max_concurrent_rollouts=async_loop.max_active_rollout_groups
            * async_loop.num_samples_per_prompt,
        ),
        renderer=from_renderers(
            Qwen35RendererConfig(
                enable_thinking=True,
                thinking_retention="all",
            )
        ),
        num_generators=2,
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
            optim=Optim.Config(
                optimizer=OptimizersContainer.Config(
                    optimizers=[
                        AdamW.Config(
                            pattern=r".*",
                            lr=1e-6,
                            betas=(0.9, 0.999),
                            weight_decay=0.0,
                        )
                    ]
                ),
                lr_scheduler=LRSchedulersContainer.Config(
                    warmup_steps=0,
                    min_lr_factor=1.0,
                ),
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=max_context_length,
                max_context_length=max_context_length,
                dtype="float32",
            ),
            parallelism=ParallelismConfig(
                data_parallel_replicate_degree=1,
                data_parallel_shard_degree=4,
                tensor_parallel_degree=2,
                expert_parallel_degree=8,
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
            cuda_graph=VLLMCudaGraphConfig(mode="NONE"),
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=2,
                tensor_parallel_degree=2,
                expert_parallel_degree=4,
            ),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=max_output_tokens,
            ),
        ),
    )
