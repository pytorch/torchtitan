# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
)
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.components.validate import Validator
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.tools.profiler import Profiler
from torchtitan.trainer import Trainer

from . import model_registry


def gemma4_debugmodel() -> Trainer.Config:
    model_spec = model_registry("debugmodel")
    packed = ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"])
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_spec,
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=8 * 2048,
            max_context_length=2048,
            steps=10,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=packed,
            shuffle=False,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(pipeline_parallel_schedule="Interleaved1F1B"),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
        validator=Validator.Config(
            freq=5,
            steps=10,
            dataloader=GrainDataLoader.Config(
                dataset=packed,
                shuffle=False,
            ),
        ),
    )


def gemma4_debugmodel_varlen_attn() -> Trainer.Config:
    config = gemma4_debugmodel()
    config.model_spec = model_registry("debugmodel", attn_backend="varlen")
    config.training.disable_cuda_graphs = True
    return config


def gemma4_12b() -> Trainer.Config:
    model_spec = model_registry("12b")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/gemma-4-12b",
        profiler=Profiler.Config(
            enable_profiling=True,
            profile_freq=100,
        ),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=True,
        ),
        model_spec=model_spec,
        optimizer=default_adamw(lr=1e-4),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=1 * 8192,
            max_context_length=8192,
            steps=10000,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=SelectiveAC.Config(),
        validator=Validator.Config(
            freq=500,
            steps=1200,
        ),
    )


def gemma4_12b_1node_full() -> Trainer.Config:
    config = gemma4_12b()
    config.compile = CompileConfig(enable=True, components=["model"])
    config.parallelism = ParallelismConfig(
        tensor_parallel_degree=1,
    )
    return config


def gemma4_12b_multinode() -> Trainer.Config:
    config = gemma4_12b()
    config.compile = CompileConfig(enable=True, components=["model"])
    config.parallelism = ParallelismConfig(
        tensor_parallel_degree=2,
    )
    return config


def gemma4_12b_long_context() -> Trainer.Config:
    config = gemma4_12b()
    config.training.max_context_length = 16384
    config.training.num_tokens_per_microbatch_per_dp_rank = 1 * 16384
    config.parallelism = ParallelismConfig(
        enable_sequence_parallel=True,
        tensor_parallel_degree=2,
    )
    return config


def gemma4_31b() -> Trainer.Config:
    model_spec = model_registry("31b")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/gemma-4-31b",
        profiler=Profiler.Config(
            enable_profiling=True,
            profile_freq=100,
        ),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=True,
        ),
        model_spec=model_spec,
        optimizer=default_adamw(lr=1e-4),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=1 * 8192,
            max_context_length=8192,
            steps=10000,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=SelectiveAC.Config(),
        validator=Validator.Config(
            freq=500,
            steps=1200,
        ),
    )


def gemma4_31b_1node_full() -> Trainer.Config:
    config = gemma4_31b()
    config.compile = CompileConfig(enable=True, components=["model"])
    config.parallelism = ParallelismConfig(
        tensor_parallel_degree=1,
    )
    return config


def gemma4_31b_multinode() -> Trainer.Config:
    config = gemma4_31b()
    config.compile = CompileConfig(enable=True, components=["model"])
    config.parallelism = ParallelismConfig(
        tensor_parallel_degree=2,
    )
    return config


def gemma4_31b_long_context() -> Trainer.Config:
    config = gemma4_31b()
    config.training.max_context_length = 16384
    config.training.num_tokens_per_microbatch_per_dp_rank = 1 * 16384
    config.parallelism = ParallelismConfig(
        enable_sequence_parallel=True,
        tensor_parallel_degree=2,
    )
    return config