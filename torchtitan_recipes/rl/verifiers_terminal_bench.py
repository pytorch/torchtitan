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
from torchtitan.distributed.activation_remat import FullActivationRematConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
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
from torchtitan.rl.examples.verifiers.rollouter import (
    RewardFromVerifiers,
    VerifiersRolloutWorker,
)
from torchtitan.rl.examples.verifiers.terminal_bench.harness import (
    TerminalBenchTerminusHarnessConfig,
)
from torchtitan.rl.examples.verifiers.terminal_bench.taskset import (
    TerminalTasksetConfig,
)
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
from torchtitan.rl.rollout.rollouter import Rollouter
from torchtitan.rl.rollout.rubric import Rubric
from torchtitan.rl.routing.inter_generator import InterGeneratorRouter
from torchtitan.rl.routing.strategies import (
    LeastLoadedRoutingStrategy,
    StickySessionRoutingStrategy,
)
from torchtitan.rl.trainer.losses import GRPOLoss
from torchtitan.rl.trainer.trainer import Trainer
from verifiers.v1.configs.agent import TimeoutConfig as AgentTimeoutConfig
from verifiers.v1.tasksets.harbor import HarborEnvConfig

from torchtitan_recipes.rl.model_configs import build_rl_model_configs

# Verifiers id of the example's taskset and harness: the package that exports both
# (`torchtitan/rl/examples/verifiers/terminal_bench/pyproject.toml`).
TERMINAL_BENCH_PACKAGE_ID = "torchtitan-verifiers-terminal-bench"


def terminal_bench_env_config(dataset: str, *, max_turns: int) -> HarborEnvConfig:
    """Verifiers' Harbor env on a Harbor dataset: Terminus-2 in a local Docker container.

    Docker is a placeholder for local runs. Set another runtime with
    ``env.model_copy(update={"agent": env.agent.model_copy(update={"runtime": ...})})``.

    Args:
        dataset: Harbor dataset id, e.g. ``"local/tmax@v1"``.
        max_turns: The agent's turn limit.
    """
    return HarborEnvConfig(
        taskset=TerminalTasksetConfig(id=TERMINAL_BENCH_PACKAGE_ID, dataset=dataset),
        agent=vf.AgentConfig(
            harness=TerminalBenchTerminusHarnessConfig(
                id=TERMINAL_BENCH_PACKAGE_ID, version="0.22.0"
            ),
            runtime=vf.DockerConfig(),
            max_turns=max_turns,
            # The only time limits: the taskset config's `ignore_timeouts=True` (Verifiers'
            # default) drops each task's own. Why: `examples/verifiers/terminal_bench/README.md`.
            timeout=AgentTimeoutConfig(setup=600, rollout=7200, scoring=12000),
        ),
    )


def terminal_bench_rollouter_config(
    train_env: vf.EnvConfig,
    validation_env: vf.EnvConfig,
    *,
    max_context_length: int,
) -> Rollouter.Config:
    """Train on ``train_env``'s tasks, shuffled; validate on ``validation_env``'s.

    ``max_context_length`` is the generator's sequence length.
    """
    if train_env.taskset == validation_env.taskset:
        raise ValueError("Training and validation must use different tasksets")
    return Rollouter.Config(
        training_dataloader=IterableRolloutDataLoader.Config(
            dataset=VerifiersTaskDataset.Config(
                env=train_env, select=vf.SelectConfig(shuffle=True, seed=42)
            ),
        ),
        validation_dataset=VerifiersTaskDataset.Config(env=validation_env),
        worker=VerifiersRolloutWorker.Config(
            rubric=Rubric.Config(
                reward_fns=[RewardFromVerifiers.Config(weight=1.0)],
                error_reward=0.0,
            ),
            max_rollout_tokens=max_context_length,
        ),
        # Python's default pool, 32 threads on 28+ CPUs: each render that waits for the renderer
        # lock holds a thread, so a small pool stalls other thread work such as sandbox cleanup.
        # TODO: 4 threads are enough once Verifiers waits for its renderer lock on the event loop.
        #   Today it waits in a thread.
        num_threads_per_worker=None,
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
        validation=ValidationConfig(steps=89),
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
            train_env=terminal_bench_env_config(dataset="local/tmax@v1", max_turns=120),
            validation_env=terminal_bench_env_config(
                dataset="terminal-bench/terminal-bench-2-1", max_turns=120
            ),
            max_context_length=max_context_length,
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
        validation=ValidationConfig(steps=89),
    )
    # TODO: update to distMoE model config
    trainer_parallelism = ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=4,
        tensor_parallel_degree=2,
        expert_parallel_degree=8,
    )
    generator_parallelism = InferenceParallelismConfig(
        data_parallel_degree=2,
        tensor_parallel_degree=2,
        expert_parallel_degree=4,
    )
    trainer_model, generator_model = build_rl_model_configs(
        build_model_config,
        "35B-A3B",
        seq_len=max_context_length,
        attn_backend="varlen",
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
    )
    return Controller.Config(
        hf_assets_path="./assets/hf/Qwen3.5-35B-A3B",
        dump_folder="outputs/rl/qwen35_35b_a3b_terminal_bench",
        rollout_recorder=RolloutSampleRecorder.Config(
            filter=KeepExtremeRewardsFilter.Config(keep_errors=True)
        ),
        async_loop=async_loop,
        rollouter=terminal_bench_rollouter_config(
            train_env=terminal_bench_env_config(dataset="local/tmax@v1", max_turns=120),
            validation_env=terminal_bench_env_config(
                dataset="terminal-bench/terminal-bench-2-1", max_turns=120
            ),
            max_context_length=max_context_length,
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
        validation=ValidationConfig(steps=89),
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
            train_env=terminal_bench_env_config(dataset="local/tmax@v1", max_turns=8),
            validation_env=terminal_bench_env_config(
                dataset="terminal-bench/terminal-bench-2-1", max_turns=8
            ),
            max_context_length=max_context_length,
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
