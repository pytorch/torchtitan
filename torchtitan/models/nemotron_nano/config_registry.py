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
from torchtitan.components.optimizer import default_adamw
from torchtitan.components.validate import Validator
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC, FullAC
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.tools.profiler import Profiler
from torchtitan.trainer import Trainer

from . import model_registry

def nemotron_debugmodel() -> Trainer.Config:
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

def nemotron_31b() -> Trainer.Config:
    model_spec = model_registry("31B")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        profiler=Profiler.Config(
            enable_profiling=True,
            profile_freq=100,
        ),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=True,
        ),
        model_spec=model_spec,
        optimizer=default_adamw(lr=3e-4),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=1 * 128,
            max_context_length=128,
            steps=1000,
            disable_cuda_graphs=True,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        parallelism=ParallelismConfig(
            tensor_parallel_degree=2,
            expert_parallel_degree=4,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=FullAC.Config(),
        validator=Validator.Config(
            freq=500,
            steps=1200,
        ),
    )
