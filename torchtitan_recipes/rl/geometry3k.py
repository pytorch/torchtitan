# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GRPO recipe for Geometry3K (geometry diagram + problem -> final answer).

Data: Hugging Face ``hiyouga/geometry3k`` (2,101 train / 601 test problems, one diagram
each). The dataset writes the diagrams to ``./assets/datasets/geometry3k/images`` on first
use: the renderer reads each image's header from its file, and the generator and the
trainer decode the image file the prompt carries. Answers are graded by Math-Verify on the
last ``\\boxed{}``.

Run with ``--module torchtitan_recipes.rl.geometry3k --config rl_grpo_qwen3_5_0_8b_geometry3k``.
"""

from renderers import CustomRendererConfig

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
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
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3_5 import (
    build_model_config as build_qwen3_5_model_config,
    QWEN3_5_VISION_PROCESSOR_KWARGS,
)
from torchtitan.observability.metrics import WandBLogger
from torchtitan.rl.controller.controller import (
    AsyncLoopConfig,
    Controller,
    ValidationConfig,
)
from torchtitan.rl.examples.dapo_math import RewardMathVerify
from torchtitan.rl.examples.geometry3k import Geometry3kDataset, Geometry3kEnv
from torchtitan.rl.generator.generator import Generator
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.rollout.data import IterableRolloutDataLoader
from torchtitan.rl.rollout.rollouter import Rollouter, RolloutWorker
from torchtitan.rl.rollout.rubric import Rubric
from torchtitan.rl.trainer.batcher import Batcher
from torchtitan.rl.trainer.losses import GRPOLoss
from torchtitan.rl.trainer.trainer import Trainer

from torchtitan_recipes.rl.model_configs import build_rl_model_configs

GEOMETRY3K_IMAGE_DIR = "./assets/datasets/geometry3k/images"


def rl_grpo_qwen3_5_0_8b_geometry3k() -> Controller.Config:
    """Qwen3.5-0.8B on Geometry3K with the diagram (4 GPUs).

    The vision tower trains with the rest of the model. Trainer FSDP=2 + 2 generators at
    TP=1; validation runs all 601 test problems at the start and the end.
    """
    # Diagrams expand to 64-700 vision tokens (median 96) at the default pixel budget, plus
    # a short problem and up to 1024 response tokens.
    seq_len = 2048
    trainer_parallelism = ParallelismConfig(data_parallel_shard_degree=2)
    generator_parallelism = InferenceParallelismConfig(tensor_parallel_degree=1)
    trainer_model, generator_model = build_rl_model_configs(
        build_qwen3_5_model_config,
        "0.8B",
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
        seq_len=seq_len,
        attn_backend="varlen",
    )
    return Controller.Config(
        hf_assets_path="./assets/hf/Qwen3.5-0.8B",
        num_generators=2,
        async_loop=AsyncLoopConfig(
            num_training_steps=100,
            num_prompts_per_train_step=16,
            num_samples_per_prompt=8,
            # One pass over the 601 test problems (the dataset reshuffles each epoch).
            validation=ValidationConfig(steps=601),
            # Qwen3.5's image tokens take M-RoPE positions.
            batcher=Batcher.Config(build_mrope_positions=True),
        ),
        rollouter=Rollouter.Config(
            training_dataloader=IterableRolloutDataLoader.Config(
                dataset=Geometry3kDataset.Config(
                    split="train", image_dir=GEOMETRY3K_IMAGE_DIR, seed=42
                )
            ),
            validation_dataset=Geometry3kDataset.Config(
                split="test", image_dir=GEOMETRY3K_IMAGE_DIR, seed=99
            ),
            worker=RolloutWorker.Config(
                rubric=Rubric.Config(reward_fns=[RewardMathVerify.Config(weight=1.0)]),
                message_env=Geometry3kEnv.Config(),
            ),
        ),
        vision_processor=VisionProcessor.Config(**QWEN3_5_VISION_PROCESSOR_KWARGS),
        renderer=from_renderers(
            CustomRendererConfig(
                import_path="torchtitan.models.qwen3_5.renderer.Qwen35ImageRefRenderer",
                enable_thinking=False,
            )
        ),
        metrics=MetricsProcessor.Config(loggers=[WandBLogger.Config()]),
        trainer=Trainer.Config(
            model=trainer_model,
            optim=Optim.Config(
                optimizer=OptimizersContainer.Config(
                    optimizers=[AdamW.Config(pattern=r".*", lr=2e-6)]
                ),
                lr_scheduler=LRSchedulersContainer.Config(
                    warmup_steps=0,
                    min_lr_factor=1.0,
                ),
            ),
            training=TrainingConfig(
                disable_accelerator_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
            ),
            parallelism=trainer_parallelism,
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=100,
                last_save_model_only=False,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=4,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(trainer_model)
                ),
            ),
        ),
        generator=Generator.Config(
            model=generator_model,
            # Linear-attention (GDN) CUDA graphs need capture sizes <= max_model_len.
            max_num_batched_tokens=seq_len,
            parallelism=generator_parallelism,
            checkpointer=None,
            sampling=SamplingConfig(temperature=1.0, max_tokens=1024),
        ),
    )
