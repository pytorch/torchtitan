# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.components.validate import Validator
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.trainer import Trainer

from . import model_registry


def _gpt_oss_debugmodel(
    seq_len: int | None = None, attn_backend: str = "varlen"
) -> Trainer.Config:
    model_spec = model_registry(
        "debugmodel", seq_len=seq_len, attn_backend=attn_backend
    )
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=8 * model_spec.max_context_length,
            max_context_length=model_spec.max_context_length,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=None,
        validator=Validator.Config(
            freq=5,
            steps=10,
        ),
    )


def gpt_oss_debugmodel(seq_len: int | None = None) -> Trainer.Config:
    return _gpt_oss_debugmodel(seq_len=seq_len)


def gpt_oss_debugmodel_flex(seq_len: int | None = None) -> Trainer.Config:
    return _gpt_oss_debugmodel(seq_len=seq_len, attn_backend="flex")


def gpt_oss_20b(seq_len: int | None = None) -> Trainer.Config:
    model_spec = model_registry("20b", seq_len=seq_len)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/gpt-oss-20b",
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=1 * model_spec.max_context_length,
            max_context_length=model_spec.max_context_length,
            steps=10000,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=FullAC.Config(),
    )


def gpt_oss_120b(seq_len: int | None = None) -> Trainer.Config:
    model_spec = model_registry("120b", seq_len=seq_len)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/gpt-oss-120b",
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=1 * model_spec.max_context_length,
            max_context_length=model_spec.max_context_length,
            steps=10000,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=FullAC.Config(),
    )
