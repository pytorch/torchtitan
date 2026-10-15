# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verified DAPO-Math recipes for Qwen3 and Muse Spark Open."""

from __future__ import annotations

from typing import cast

from renderers import Qwen3RendererConfig

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.checkpointer.torch_checkpointing import (
    TorchCheckpointingManager,
)
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

from torchtitan.config.transform import apply_transforms
from torchtitan.config.transform.dist_moe import DistMoeTransform
from torchtitan.distributed.activation_remat import ActivationRematConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.common.dist_moe import DistMoeRuntime
from torchtitan.models.muse_spark import (
    build_model_config as build_muse_spark_model_config,
)
from torchtitan.models.muse_spark.renderer import MuseSparkRendererConfig
from torchtitan.models.muse_spark.tokenizer import MuseSparkTokenizer
from torchtitan.models.qwen3 import build_model_config
from torchtitan.rl.controller.controller import (
    AsyncLoopConfig,
    Controller,
    ValidationConfig,
)
from torchtitan.rl.examples.dapo_math.data import AIME2025Dataset, DapoMathDataset
from torchtitan.rl.examples.dapo_math.env import DapoMathEnv
from torchtitan.rl.examples.dapo_math.rubric import RewardMathVerify
from torchtitan.rl.generator.generator import (
    SamplingConfig,
    VLLMCudaGraphConfig,
    VLLMGenerator,
)
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.rollout.advantage import AdvantageEstimator
from torchtitan.rl.rollout.data import IterableRLDataLoader
from torchtitan.rl.rollout.rollouter import Rollouter, RolloutWorker
from torchtitan.rl.rollout.rubric import Rubric
from torchtitan.rl.rollout.token_env import TokenEnv
from torchtitan.rl.trainer.batcher import Batcher
from torchtitan.rl.trainer.losses import DAPOLoss
from torchtitan.rl.trainer.trainer import Trainer
from torchtitan.rl.trainer.training_sample_builder import TrainingSampleBuilder

from torchtitan_recipes.models.muse_spark import MUSE_SPARK_SPECIAL_TOKENS

from torchtitan_recipes.rl.model_configs import build_rl_model_configs

# TODO: Enable CUDA graphs for RL trainers after eager/graph numerics parity is
# verified.


def _dapo_math_rollouter_config(
    *,
    validation_dataset: AIME2025Dataset.Config,
    token_env: TokenEnv.Config,
) -> Rollouter.Config:
    return Rollouter.Config(
        training_dataloader=IterableRLDataLoader.Config(
            dataset=DapoMathDataset.Config()
        ),
        validation_dataset=validation_dataset,
        worker=RolloutWorker.Config(
            rubric=Rubric.Config(
                reward_fns=[RewardMathVerify.Config(weight=1.0)],
                error_reward=0.0,
            ),
            message_env=DapoMathEnv.Config(),
            token_env=token_env,
            advantage=AdvantageEstimator.Config(should_std_normalize=False),
        ),
    )


def _qwen3_4b_dapo_math_config(
    *,
    max_response_tokens: int,
    max_total_tokens: int,
    dump_folder: str,
) -> Controller.Config:
    """Build the shared Qwen3-4B DAPO-Math configuration."""
    num_validation_samples = 30
    validation_dataset = AIME2025Dataset.Config()
    trainer_parallelism = ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=1,
        tensor_parallel_degree=2,
    )
    generator_parallelism = InferenceParallelismConfig(
        data_parallel_degree=1,
        tensor_parallel_degree=1,
    )
    trainer_model, generator_model = build_rl_model_configs(
        build_model_config,
        "4B",
        seq_len=max_total_tokens,
        attn_backend="varlen",
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
    )
    return Controller.Config(
        hf_assets_path="./assets/hf/Qwen3-4B-Base",
        dump_folder=dump_folder,
        async_loop=AsyncLoopConfig(
            num_training_steps=150,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=16,
            target_offpolicy_steps=4,
            validation=ValidationConfig(
                steps=num_validation_samples,
            ),
        ),
        rollouter=_dapo_math_rollouter_config(
            validation_dataset=validation_dataset,
            token_env=TokenEnv.Config(
                max_rollout_tokens=max_total_tokens,
                max_num_turns=1,
            ),
        ),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=True)),
        num_generators=6,
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
            model=trainer_model,
            optim=Optim.Config(
                optimizer=OptimizersContainer.Config(
                    optimizers=[
                        AdamW.Config(
                            pattern=r".*",
                            lr=1e-6,
                            betas=(0.9, 0.98),
                            weight_decay=0.1,
                        )
                    ]
                ),
                lr_scheduler=LRSchedulersContainer.Config(
                    warmup_steps=0,
                    min_lr_factor=1.0,
                ),
            ),
            # A minimum factor of 1 keeps the learning rate constant.
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=max_total_tokens,
                max_context_length=max_total_tokens,
            ),
            parallelism=trainer_parallelism,
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
                    global_vocab_size=decoder_vocab_size(trainer_model),
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model=generator_model,
            model_dtype="bfloat16",
            parallelism=generator_parallelism,
            cuda_graph=VLLMCudaGraphConfig(mode="FULL"),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=max_response_tokens,
            ),
        ),
    )


def rl_dapo_qwen3_4b_math_8k() -> Controller.Config:
    """Run 8K responses on one node: one TP=2 trainer and six TP=1 generators."""
    return _qwen3_4b_dapo_math_config(
        max_response_tokens=8192,
        max_total_tokens=10240,
        dump_folder="outputs/rl/qwen3_4b_dapo_math_8k",
    )


def rl_dapo_qwen3_4b_math_32k() -> Controller.Config:
    """Run 32K responses on one node: one TP=2 trainer and six TP=1 generators."""
    return _qwen3_4b_dapo_math_config(
        max_response_tokens=32768,
        max_total_tokens=34816,
        dump_folder="outputs/rl/qwen3_4b_dapo_math_32k",
    )


def muse_spark_816b_a42b_dapo_math_gb300_config(
    *,
    hf_assets_path: str,
    dapo_math_repo_id: str,
    aime2025_repo_id: str,
) -> Controller.Config:
    """DAPO-Math for Muse Spark Open 816B-A42B on GB300 with Dist-MoE in both roles.

    Runs five single-turn, on-policy steps with 8 prompts and 8 samples per prompt,
    a 65,536-token context and up to 63,488 response tokens. Validation is disabled.
    Checks trainer/generator logprob agreement with padding in both roles.

    Trainer: 256 GPUs (FSDP(shard)=64, TP=4, EP=4), fp32 master weights and a bf16
    forward, no CUDA graphs. Generator: 16 GPUs (DP=4, TP=4, EP=16), bf16 weights,
    full CUDA graphs.

    Supply the checkpoint directory and dataset repo IDs or local directories;
    the launcher owns cluster placement.
    """
    max_response_tokens = 63488
    max_total_tokens = 65536
    trainer_parallelism = ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=64,
        tensor_parallel_degree=4,
        expert_parallel_degree=4,
        enable_sequence_parallel=True,
    )
    generator_parallelism = InferenceParallelismConfig(
        data_parallel_degree=4,
        tensor_parallel_degree=4,
        expert_parallel_degree=16,
    )
    trainer_model, generator_model = build_rl_model_configs(
        build_muse_spark_model_config,
        "816B-A42B",
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
        seq_len=max_total_tokens,
        moe_comm_backend="standard",
        attn_backend="varlen",
    )
    rollouter = _dapo_math_rollouter_config(
        validation_dataset=AIME2025Dataset.Config(
            repo_id=aime2025_repo_id,
            num_samples=8,
        ),
        token_env=TokenEnv.Config(
            max_rollout_tokens=max_total_tokens,
            max_num_turns=1,
        ),
    )
    rollouter.training_dataloader.dataset = DapoMathDataset.Config(
        repo_id=dapo_math_repo_id
    )
    config = Controller.Config(
        hf_assets_path=hf_assets_path,
        dump_folder="outputs/rl/muse_spark_816b_a42b_dapo_math_gb300",
        tokenizer=MuseSparkTokenizer.Config(**MUSE_SPARK_SPECIAL_TOKENS),
        async_loop=AsyncLoopConfig(
            num_training_steps=5,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=8,
            target_offpolicy_steps=0,
            # Skip the pre- and post-training validation passes.
            validation=ValidationConfig(steps=0),
            # Keep all-pass / all-fail groups so every step trains exactly
            # num_prompts_per_train_step groups.
            batcher=Batcher.Config(
                training_sample_builder=TrainingSampleBuilder.Config(
                    drop_zero_std_reward_groups=False
                )
            ),
        ),
        rollouter=rollouter,
        renderer=MuseSparkRendererConfig(reasoning_effort="low"),
        num_generators=1,
        trainer=Trainer.Config(
            model=trainer_model,
            optim=Optim.Config(
                optimizer=OptimizersContainer.Config(
                    optimizers=[
                        AdamW.Config(
                            pattern=r".*",
                            lr=1e-6,
                            betas=(0.9, 0.95),
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
                num_tokens_per_microbatch_per_dp_rank=max_total_tokens,
                max_context_length=max_total_tokens,
                dtype="float32",
                mixed_precision_param="bfloat16",
            ),
            parallelism=trainer_parallelism,
            # EP=4 confines each expert group to one host; factor 4 covers
            # its worst-case routing imbalance.
            dist_moe_runtime=DistMoeRuntime.Config(scratch_capacity_factor=4.0),
            # Test run: load the HF weights, never write a checkpoint.
            checkpointer=cast(
                CheckpointManager.Config,
                TorchCheckpointingManager.Config(
                    initial_load_in_hf=True,
                    load_only=True,
                ),
            ),
            # Dist-MoE's eager expert-output callback (the trainable Muse norm)
            # rejects TorchDispatchMode-based checkpointing. torch_remat uses
            # saved-tensor hooks for whole-block recomputation instead.
            activation_remat=ActivationRematConfig(save_regions=[]),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=DAPOLoss.Config(
                    ratio_clip_low=0.2,
                    ratio_clip_high=0.28,
                    global_vocab_size=decoder_vocab_size(trainer_model),
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model=generator_model,
            model_dtype="bfloat16",
            parallelism=generator_parallelism,
            cuda_graph=VLLMCudaGraphConfig(mode="FULL"),
            # vLLM's profiling dummy batch can route every token to one expert.
            dist_moe_runtime=DistMoeRuntime.Config(
                inference=True, scratch_capacity_factor=16.0
            ),
            checkpointer=None,
            gpu_memory_limit=0.9,
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=max_response_tokens,
            ),
        ),
    )
    # Dist-MoE routed experts in both roles.
    config.trainer = apply_transforms(
        config.trainer, [DistMoeTransform(inplace_wgrad_accum=True)]
    )
    config.generator = apply_transforms(
        config.generator, [DistMoeTransform(inplace_wgrad_accum=True)]
    )
    return config
