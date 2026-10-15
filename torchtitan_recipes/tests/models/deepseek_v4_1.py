# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek V4.1 model configurations used by tests.

Two recipes:
  - ``deepseek_v4_1_debugmodel``: upstream recipe (GPU-compatible, reference path).
  - ``deepseek_v4_1_reference``: draft-style reference recipe with offline data.
"""

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.observability.profiler import Profiler
from torchtitan.trainer import Trainer

from torchtitan.models.deepseek_v4_1 import build_model_config


def deepseek_v4_1_debugmodel(
    seq_len: int | None = None,
    *,
    parallelism: ParallelismConfig | None = None,
) -> Trainer.Config:
    if parallelism is None:
        parallelism = ParallelismConfig(
            expert_parallel_degree=1,
        )
    model_config = build_model_config(
        "debugmodel", seq_len=seq_len, parallelism=parallelism
    )
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        profiler=Profiler.Config(
            enable_profiling=False,
            profile_freq=10,
            profiler_active=10,
            profiler_warmup=0,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        model=model_config,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
            max_num_documents=64,
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_ratio=0.8,
                decay_type="linear",
                min_lr_factor=0.0,
            ),
        ),
        training=TrainingConfig(
            disable_cuda_graphs=parallelism.expert_parallel_degree > 1,
            num_tokens_per_microbatch_per_dp_rank=8 * model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=10,
        ),
        parallelism=parallelism,
        activation_remat=None,
        checkpointer=None,
    )


def deepseek_v4_1_debugmodel_fsdp2_ep2() -> Trainer.Config:
    config = deepseek_v4_1_debugmodel(
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=2,
            expert_parallel_degree=2,
        )
    )
    config.training.steps = 1
    config.metrics.log_freq = 1
    config.training.disable_cuda_graphs = True
    return config


def deepseek_v4_1_reference() -> Trainer.Config:
    """Draft-style reference recipe with offline data (CPU/single-rank).

    Uses the tiny debugmodel config with reduced seq_len and offline
    data fixture. Suitable for verifying the training chain end-to-end.
    """
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=1,
        expert_parallel_degree=1,
        enable_sequence_parallel=False,
    )
    model_config = build_model_config(
        "debugmodel", seq_len=512, parallelism=parallelism
    )
    config = Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        profiler=Profiler.Config(
            enable_profiling=False,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        model=model_config,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
            max_num_documents=64,
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=1e-3)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=0,
                decay_ratio=1.0,
                decay_type="linear",
                min_lr_factor=0.0,
            ),
        ),
        training=TrainingConfig(
            disable_cuda_graphs=True,
            num_tokens_per_microbatch_per_dp_rank=512,
            max_context_length=model_config.max_context_length,
            steps=3,
        ),
        parallelism=parallelism,
        activation_remat=None,
        checkpointer=None,
    )
    return config
