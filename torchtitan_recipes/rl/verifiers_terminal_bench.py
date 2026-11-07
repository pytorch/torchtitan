# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Terminal-agent recipes using Verifiers and TitanRL.

The recipes train on ``local/tmax@v1``, the TMax tasks exported by
``prepare_tmax.py`` into ``~/.cache/harbor/local_tmax_v1``, and validate on the
Terminal-Bench 2.1 Harbor dataset, which the ``harbor`` CLI downloads into
``~/.cache/harbor`` on first use. Edit the ids below to use other datasets.
The Muse Glimmer example is capped at its model's 16K context length.
"""

import math

import verifiers.v1 as vf

from renderers import CustomRendererConfig, Qwen35RendererConfig

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.renderer import from_renderers
from torchtitan.config import PrecisionConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import apply_transforms
from torchtitan.config.transform.dist_moe import DistMoeTransform
from torchtitan.distributed.activation_remat import FullActivationRematConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.common.dist_moe import DistMoeRuntime
from torchtitan.models.muse_glimmer import (
    build_model_config as build_muse_glimmer_model_config,
)
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.rl.controller.controller import (
    AsyncLoopConfig,
    Controller,
    ValidationConfig,
)
from torchtitan.rl.examples.verifiers.data import VerifiersTaskDataset
from torchtitan.rl.examples.verifiers.env_server import VerifiersEnvServer
from torchtitan.rl.examples.verifiers.generation_server import GenerationServer
from torchtitan.rl.examples.verifiers.rollouter import (
    RewardFromVerifiers,
    VerifiersRollouter,
)
from torchtitan.rl.examples.verifiers.terminal_bench.harness import (
    register_harness_alias,
    TerminalBenchTerminusHarnessConfig,
)
from torchtitan.rl.examples.verifiers.terminal_bench.taskset import (
    TerminalTasksetConfig,
)
from torchtitan.rl.examples.verifiers.worker_setup import register_local_taskset_alias
from torchtitan.rl.generator.generator import Generator
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.generator.vllm.config import VLLMCudaGraphConfig
from torchtitan.rl.generator.vllm.engine import VLLMEngine
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.rollout.data import IterableRolloutDataLoader
from torchtitan.rl.rollout.recorder import (
    KeepExtremeRewardsFilter,
    RolloutSampleRecorder,
)
from torchtitan.rl.rollout.rubric import Rubric
from torchtitan.rl.routing.inter_generator import InterGeneratorRouter
from torchtitan.rl.routing.strategies import (
    LeastLoadedRoutingStrategy,
    StickySessionRoutingStrategy,
)
from torchtitan.rl.trainer.losses import DAPOLoss, GRPOLoss
from torchtitan.rl.trainer.trainer import Trainer
from verifiers.v1.configs.agent import TimeoutConfig as AgentTimeoutConfig
from verifiers.v1.tasksets.harbor import HarborEnvConfig

from torchtitan_recipes.rl.model_configs import build_rl_model_configs

_ENV_SERVER_WORKERS = 16


def terminal_bench_rollouter_config(
    train_dataset: str,
    validation_dataset: str,
    *,
    max_context_length: int,
    max_turns: int,
    max_concurrent_rollouts: int,
    taskset_config_cls: type[TerminalTasksetConfig] = TerminalTasksetConfig,
) -> VerifiersRollouter.Config:
    """Build the Terminal-Bench rollouter: Harbor datasets by id, local Docker, in-sandbox Terminus-2.

    Args:
        train_dataset: Harbor dataset id for training, e.g. ``"local/tmax@v1"``.
        validation_dataset: Harbor dataset id for validation; must differ from ``train_dataset``.
        max_context_length: The generator's sequence length; the generation server caps
            each rollout at it.
        max_turns: The agent turn limit, which Verifiers enforces.
        max_concurrent_rollouts: Sizes the env server. Set it to the number of rollouts
            the controller keeps in flight, or the excess queues in the env server and
            the generators idle.
        taskset_config_cls: ``TerminalTasksetConfig`` or a subclass. Every env-server worker
            imports its module, which must export a Taskset bound to it and an Env (e.g.
            ``HarborEnv``), like ``terminal_bench/taskset.py``. Runtime and harness are
            fields of the returned config.
    """
    if train_dataset == validation_dataset:
        raise ValueError(
            "Training and Terminal-Bench evaluation must use different datasets"
        )

    taskset_id = register_local_taskset_alias(taskset_config_cls.__module__)
    return VerifiersRollouter.Config(
        training_dataloader=IterableRolloutDataLoader.Config(
            dataset=VerifiersTaskDataset.Config(
                verifiers_taskset=taskset_config_cls(
                    id=taskset_id, dataset=train_dataset, ignore_timeouts=False
                ),
                seed=42,
                shuffle=True,
            ),
        ),
        validation_dataset=VerifiersTaskDataset.Config(
            verifiers_taskset=taskset_config_cls(
                id=taskset_id, dataset=validation_dataset, ignore_timeouts=False
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
                    # Setup and agent timeouts apply to every task. Grading uses the task's
                    # verifier timeout from task.toml (ignore_timeouts=False), else Harbor's 600 s.
                    timeout=AgentTimeoutConfig(setup=600, rollout=7200),
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
        validation=ValidationConfig(steps=-1),
    )
    trainer_parallelism = ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=8,
        tensor_parallel_degree=1,
    )
    generator_parallelism = InferenceParallelismConfig(
        data_parallel_degree=1,
        tensor_parallel_degree=1,
    )
    trainer_model, generator_model = build_rl_model_configs(
        build_model_config,
        "9B",
        seq_len=max_context_length,
        attn_backend="varlen",
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
    )
    return Controller.Config(
        hf_assets_path="./assets/hf/Qwen3.5-9B",
        dump_folder="outputs/rl/qwen35_9b_terminal_bench",
        rollout_recorder=RolloutSampleRecorder.Config(
            filter=KeepExtremeRewardsFilter.Config(keep_errors=True)
        ),
        async_loop=async_loop,
        rollouter=terminal_bench_rollouter_config(
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
        precision=PrecisionConfig(dtype="float32"),
        trainer=Trainer.Config(
            model=trainer_model,
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
                disable_accelerator_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=max_context_length,
            ),
            parallelism=trainer_parallelism,
            activation_remat=FullActivationRematConfig(),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=20,
                keep_latest_k=3,
                async_mode="async",
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=32,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(trainer_model)
                ),
            ),
        ),
        generator=Generator.Config(
            model=generator_model,
            engine=VLLMEngine.Config(
                cuda_graph=VLLMCudaGraphConfig(mode="FULL_DECODE_ONLY")
            ),
            parallelism=generator_parallelism,
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=max_output_tokens,
            ),
        ),
    )


def rl_dapo_qwen35_35b_a3b_terminal_bench_dist_moe() -> Controller.Config:
    """Qwen3.5-35B-A3B: train on TMax (``local/tmax@v1``), validate on Terminal-Bench 2.1.

    Dist-MoE routed experts in both roles, so it needs SM100+ GPUs. The 4-GPU trainer
    needs 288 GB GPUs (GB300, B300). A B200 (180 GB) would keep only about 40 GB per GPU
    for activations. There, set the trainer's ``data_parallel_shard_degree=4`` (8 GPUs, EP 4
    and the scratch factor unchanged).

    20 GPUs on GB300 (4 GPUs per host): 5 GPU hosts plus a controller.
        trainer:     4 GPUs, FSDP 2 x TP 2 x EP 4. Model state is 35.1B x 16 B (fp32
                     weights, gradients, AdamW m and v) / 4 = 140 GB per GPU.
        generators:  16 one-GPU engines, each with all 256 experts. Weights are
                     35.1B x 2 B (bf16) = 70 GB per GPU. The rest holds the KV cache.

    Terminal-Bench is generation-bound: a rollout makes up to 120 model calls in a row,
    with a sandbox command between each, while the trainer runs one forward and backward
    pass per token. The trainer state fits one 4-GPU host, so the other 16 GPUs generate.
    A one-GPU engine needs no expert all-to-all or TP all-reduce, and Dist-MoE lets it
    capture FULL CUDA graphs.

    The model has 2 KV heads, so TP is at most 2. Trainer EP must be at least TP and
    divide both the 256 experts and FSDP x TP.
    """
    max_context_length = 65536
    max_output_tokens = 16384
    async_loop = AsyncLoopConfig(
        num_training_steps=100,
        num_prompts_per_train_step=8,
        num_samples_per_prompt=32,
        target_offpolicy_steps=4,
        validation=ValidationConfig(steps=-1),
    )
    trainer_parallelism = ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=2,
        tensor_parallel_degree=2,
        expert_parallel_degree=4,
    )
    generator_parallelism = InferenceParallelismConfig(
        data_parallel_degree=1,
        tensor_parallel_degree=1,
    )
    trainer_model, generator_model = build_rl_model_configs(
        build_model_config,
        "35B-A3B",
        seq_len=max_context_length,
        attn_backend="varlen",
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
    )
    config = Controller.Config(
        hf_assets_path="./assets/hf/Qwen3.5-35B-A3B",
        dump_folder="outputs/rl/qwen35_35b_a3b_terminal_bench",
        rollout_recorder=RolloutSampleRecorder.Config(
            filter=KeepExtremeRewardsFilter.Config(keep_errors=True)
        ),
        async_loop=async_loop,
        rollouter=terminal_bench_rollouter_config(
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
        num_generators=16,
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
        precision=PrecisionConfig(dtype="float32"),
        trainer=Trainer.Config(
            model=trainer_model,
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
                disable_accelerator_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=max_context_length,
            ),
            parallelism=trainer_parallelism,
            # Factor EP=4 keeps the worst case (every token routed to one rank) in device scratch.
            dist_moe_runtime=DistMoeRuntime.Config(scratch_capacity_factor=4.0),
            activation_remat=FullActivationRematConfig(),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=20,
                keep_latest_k=3,
                async_mode="async",
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=32,
                loss_fn=DAPOLoss.Config(
                    ratio_clip_low=0.2,
                    ratio_clip_high=0.28,
                    global_vocab_size=decoder_vocab_size(trainer_model),
                ),
            ),
        ),
        generator=Generator.Config(
            model=generator_model,
            engine=VLLMEngine.Config(cuda_graph=VLLMCudaGraphConfig(mode="FULL")),
            parallelism=generator_parallelism,
            # At EP 1 every routed token stays on this GPU: the default factor 1.0 is the worst case.
            dist_moe_runtime=DistMoeRuntime.Config(inference=True),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=max_output_tokens,
            ),
        ),
    )
    # Dist-MoE routed experts in both roles.
    config.trainer = apply_transforms(config.trainer, [DistMoeTransform()])
    config.generator = apply_transforms(config.generator, [DistMoeTransform()])
    return config


def rl_grpo_muse_glimmer_30b_terminal_bench() -> Controller.Config:
    """Muse Glimmer 30B: short-context TMax/Terminal-Bench example.

    16 GPUs: 8 trainer (FSDP=4 x TP=2) and 4 generators (TP=2 each).
    The 30B model flavor supports at most 16K tokens, so this caps the agent
    at 8 turns. Long Terminal-Bench trajectories can still exceed that limit;
    this is not a full 120-turn benchmark configuration.
    """
    max_context_length = 16384
    max_output_tokens = 4096
    async_loop = AsyncLoopConfig(
        num_training_steps=100,
        num_prompts_per_train_step=8,
        num_samples_per_prompt=8,
        validation=ValidationConfig(steps=-1),
    )
    trainer_parallelism = ParallelismConfig(
        data_parallel_shard_degree=4,
        tensor_parallel_degree=2,
    )
    generator_parallelism = InferenceParallelismConfig(
        data_parallel_degree=1,
        tensor_parallel_degree=2,
    )
    trainer_model, generator_model = build_rl_model_configs(
        build_muse_glimmer_model_config,
        "30B",
        seq_len=max_context_length,
        attn_backend="varlen",
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
        # Muse's SoftCappedLinear head cannot use LMHeadFP32OutputConverter.
        fp32_lm_head=False,
    )
    return Controller.Config(
        hf_assets_path="./assets/hf/Muse-Glimmer-30B",
        dump_folder="outputs/rl/muse_glimmer_30b_terminal_bench",
        rollout_recorder=RolloutSampleRecorder.Config(
            filter=KeepExtremeRewardsFilter.Config(keep_errors=True)
        ),
        async_loop=async_loop,
        rollouter=terminal_bench_rollouter_config(
            train_dataset="local/tmax@v1",
            validation_dataset="terminal-bench/terminal-bench-2-1",
            max_context_length=max_context_length,
            max_turns=8,
            max_concurrent_rollouts=async_loop.max_active_rollout_groups
            * async_loop.num_samples_per_prompt,
        ),
        renderer=from_renderers(
            CustomRendererConfig(
                import_path="torchtitan.models.muse_glimmer.renderer.MuseGlimmerRenderer",
                reasoning_strength="low",
            )
        ),
        num_generators=4,
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
            model=trainer_model,
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
                disable_accelerator_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=max_context_length,
            ),
            parallelism=trainer_parallelism,
            activation_remat=FullActivationRematConfig(),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=20,
                keep_latest_k=3,
                async_mode="async",
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(trainer_model)
                ),
            ),
        ),
        generator=Generator.Config(
            model=generator_model,
            engine=VLLMEngine.Config(cuda_graph=VLLMCudaGraphConfig(mode="NONE")),
            parallelism=generator_parallelism,
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=max_output_tokens,
            ),
        ),
    )
