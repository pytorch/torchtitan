# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verified DeepSeek V3 model recipes."""

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.deepseek_v3 import build_model_config
from torchtitan.trainer import Trainer


def deepseek_v3_16b(seq_len: int | None = None) -> Trainer.Config:
    model_config = build_model_config("16B", seq_len=seq_len, attn_backend="flex")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./assets/hf/deepseek-moe-16b-base",
        model=model_config,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=2.2e-4)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                decay_ratio=0.8,
                decay_type="cosine",
                min_lr_factor=0.1,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=4 * model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=1000,
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
            expert_parallel_degree=8,
        ),
        checkpointer=None,
        activation_checkpoint=SelectiveAC.Config(),
    )


def deepseek_v3_671b(seq_len: int | None = None) -> Trainer.Config:
    model_config = build_model_config(
        "671B",
        seq_len=seq_len,
        attn_backend="flex",
    )
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./assets/hf/DeepSeek-V3.1-Base",
        model=model_config,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=2.2e-4)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2000,
                decay_ratio=0.8,
                decay_type="cosine",
                min_lr_factor=0.1,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=4 * model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=10000,
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
            expert_parallel_degree=2,
        ),
        checkpointer=None,
        activation_checkpoint=SelectiveAC.Config(),
    )
