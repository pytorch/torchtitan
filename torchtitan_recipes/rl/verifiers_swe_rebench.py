# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SWE-rebench V2.1 coding-agent recipe using Verifiers and TitanRL.

Tasks come from the SWE-rebench V2.1 Harbor dataset, which Verifiers downloads
into ``~/.cache/harbor`` on first use. Training and validation take disjoint
projects. See ``torchtitan/rl/examples/verifiers/swe_rebench/README.md``.
"""

import math
from typing import Literal

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
from torchtitan.config import PrecisionConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import FullActivationRematConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
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
    VerifiersRolloutWorker,
)
from torchtitan.rl.examples.verifiers.swe_rebench.taskset import (
    SWERebenchV21TasksetConfig,
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
from torchtitan.rl.trainer.losses import GRPOLoss
from torchtitan.rl.trainer.trainer import Trainer
from verifiers.v1.configs.agent import TimeoutConfig as AgentTimeoutConfig
from verifiers.v1.harnesses import BashHarnessConfig

from torchtitan_recipes.rl.model_configs import build_rl_model_configs

_ENV_SERVER_WORKERS = 16


def _swe_rebench_rollouter_config(
    *,
    max_context_length: int,
    max_turns: int,
    max_concurrent_rollouts: int,
) -> VerifiersRollouter.Config:
    """Train and validate on SWE-rebench V2.1 with Verifiers' bash harness in Docker.

    `max_context_length` is the generator's sequence length; the generation
    server caps each rollout at it. `max_turns` is the agent turn limit, which
    Verifiers enforces. `max_concurrent_rollouts` sizes the env server; set it to
    the number of rollouts the controller keeps in flight, or the excess queues in
    the env server and the generators idle.
    """
    taskset_id = register_local_taskset_alias(SWERebenchV21TasksetConfig.__module__)

    def taskset(
        partition: Literal["train", "validation"]
    ) -> SWERebenchV21TasksetConfig:
        return SWERebenchV21TasksetConfig(id=taskset_id, partition=partition)

    return VerifiersRollouter.Config(
        training_dataloader=IterableRolloutDataLoader.Config(
            dataset=VerifiersTaskDataset.Config(
                verifiers_taskset=taskset("train"), seed=42, shuffle=True
            ),
        ),
        # Validation runs the first `ValidationConfig.steps` tasks of this order. The
        # taskset yields tasks sorted by name, so shuffle (with a fixed seed, so every
        # validation sees the same tasks) to sample across projects.
        validation_dataset=VerifiersTaskDataset.Config(
            verifiers_taskset=taskset("validation"), seed=99, shuffle=True
        ),
        verifiers_env_server=VerifiersEnvServer.Config(
            # The rollouter fills in the training taskset.
            environment=vf.SingleAgentEnvConfig(
                agent=vf.AgentConfig(
                    # Verifiers 0.4.0's bash harness summarizes the history by default
                    # when the context nears its limit. Off: the episode ends at the
                    # limit instead; compaction has not been tried with TitanRL.
                    harness=BashHarnessConfig(compaction=None),
                    runtime=vf.DockerConfig(),
                    max_turns=max_turns,
                    # Seconds. setup covers pulling the task image. scoring covers
                    # restoring and applying the test patch and running the tests;
                    # V2.1's own verifier allows 2400 s, its test command 1800 s.
                    timeout=AgentTimeoutConfig(
                        setup=1500, rollout=14400, finalize=300, scoring=2400
                    ),
                ),
            ),
            serve=vf.ServeConfig(
                pool=vf.StaticPoolConfig(num_workers=_ENV_SERVER_WORKERS),
                max_concurrent=math.ceil(max_concurrent_rollouts / _ENV_SERVER_WORKERS),
                address="tcp://127.0.0.1:0",
            ),
        ),
        worker=VerifiersRolloutWorker.Config(
            rubric=Rubric.Config(
                reward_fns=[RewardFromVerifiers.Config(weight=1.0)],
                error_reward=0.0,
            ),
            generation_server=GenerationServer.Config(
                max_rollout_tokens=max_context_length
            ),
        ),
    )


def rl_grpo_qwen35_9b_swe_rebench() -> Controller.Config:
    """Qwen3.5-9B on SWE-rebench V2.1 with the bash harness, on one 8-GPU host.

    8 GPUs: 4 trainer (FSDP=4) and 4 one-GPU generators. Batches of 8 tasks x
    8 rollouts, up to 100 agent turns at 64K context.

    Not run end to end yet.
    """
    max_context_length = 65536
    async_loop = AsyncLoopConfig(
        num_training_steps=50,
        num_prompts_per_train_step=8,
        num_samples_per_prompt=8,
        target_offpolicy_steps=2,
        validation=ValidationConfig(steps=50, freq=10),
    )
    trainer_parallelism = ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=4,
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
        dump_folder="outputs/rl/qwen35_9b_swe_rebench",
        rollout_recorder=RolloutSampleRecorder.Config(
            filter=KeepExtremeRewardsFilter.Config(keep_errors=True)
        ),
        async_loop=async_loop,
        rollouter=_swe_rebench_rollouter_config(
            max_context_length=max_context_length,
            max_turns=100,
            max_concurrent_rollouts=async_loop.max_active_rollout_groups
            * async_loop.num_samples_per_prompt,
        ),
        renderer=from_renderers(
            Qwen35RendererConfig(enable_thinking=True, thinking_retention="all")
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
                interval=10,
                keep_latest_k=3,
                async_mode="async",
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=16,
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
            sampling=SamplingConfig(temperature=1.0, top_p=1.0, max_tokens=16384),
        ),
    )
