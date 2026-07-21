# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Single-node Qwen3-4B-Base DAPO-Math recipes."""

from __future__ import annotations

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import default_adamw
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
from torchtitan.experiments.rl.examples.dapo_math.rollouter import DapoMathRollouter
from torchtitan.experiments.rl.losses import DAPOLoss
from torchtitan.experiments.rl.models.cast_linear import LMHeadCastConverter
from torchtitan.experiments.rl.models.vllm_registry import InferenceParallelismConfig
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.routing.inter_generator_router import (
    InterGeneratorRouter,
)
from torchtitan.experiments.rl.routing.strategies import LeastLoadedRoutingStrategy
from torchtitan.models.qwen3 import model_registry


_NUM_TRAINING_STEPS = 1000


def _qwen3_4b_dapo_math_8k_config(
    *, trainer_parallelism: ParallelismConfig
) -> Controller.Config:
    """Build the 8K DAPO-Math recipe for one trainer parallelism layout."""
    return Controller.Config(
        model_spec=model_registry(
            "4B",
            attn_backend="varlen",
            # Compute vocabulary logits in fp32; the rest of the forward uses bf16.
            converters=[LMHeadCastConverter.Config()],
        ),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-4B-Base",
        dump_folder="outputs/rl/qwen3_4b_dapo_math_8k",
        async_loop=AsyncLoopConfig(
            num_training_steps=_NUM_TRAINING_STEPS,
            num_groups_per_train_step=8,
            group_size=16,
            # Four versions of policy lag plus the current version allow 40 active groups.
            max_offpolicy_steps=4,
            validation=ValidationConfig(num_samples=30),
            batcher=Batcher.Config(
                # The dataset is filtered to 2K prompts; generation contributes up to 8K.
                batch=BatchConfig(local_batch_size=1, seq_len=10240),
            ),
        ),
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=DapoMathRollouter.Config(),
        renderer=RendererConfig(name="qwen3", enable_thinking=True),
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
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(
                lr=1e-6,
                betas=(0.9, 0.98),
                weight_decay=0.1,
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=0,
                total_steps=_NUM_TRAINING_STEPS,
                decay_type="linear",
                min_lr_factor=1.0,
            ),
            training=TrainingConfig(),
            parallelism=trainer_parallelism,
            checkpoint=CheckpointManager.Config(
                enable=True,
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
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=1,
            ),
            cudagraph=VLLMCudagraphConfig(enable=True),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=8192,
            ),
        ),
    )


def rl_dapo_qwen3_4b_math_8k() -> Controller.Config:
    """Use two replicated trainer ranks and six single-GPU generators."""
    return _qwen3_4b_dapo_math_8k_config(
        trainer_parallelism=ParallelismConfig(
            data_parallel_replicate_degree=2,
            data_parallel_shard_degree=1,
            tensor_parallel_degree=1,
        )
    )


def rl_dapo_qwen3_4b_math_8k_tp2() -> Controller.Config:
    """Use one TP=2 trainer mesh and six single-GPU generators."""
    return _qwen3_4b_dapo_math_8k_config(
        trainer_parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=1,
            tensor_parallel_degree=2,
        )
    )
