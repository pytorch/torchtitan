# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek V4.1 Flash configurations for training and distributed tests."""

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import DebugConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.deepseek_v4_1 import build_model_config
from torchtitan.observability.metrics import MetricsProcessor, TensorBoardLogger
from torchtitan.observability.profiler import Profiler
from torchtitan.trainer import Trainer


def deepseek_v4_1_debugmodel(
    seq_len: int = 128,
    *,
    parallelism: ParallelismConfig | None = None,
    attention_backend: str = "fused",
) -> Trainer.Config:
    if parallelism is None:
        parallelism = ParallelismConfig()
    model_config = build_model_config(
        "debugmodel",
        seq_len=seq_len,
        parallelism=parallelism,
        attention_backend=attention_backend,
    )
    model_config.local_compile.regions = []
    return Trainer.Config(
        model=model_config,
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            )
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
            max_num_documents=16,
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
            max_context_length=seq_len,
            num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
            steps=10,
            disable_accelerator_graphs=True,
        ),
        debug=DebugConfig(
            seed=42,
            deterministic=True,
            spmd_typechecking=True,
        ),
        parallelism=parallelism,
        profiler=Profiler.Config(enable_profiling=False),
        metrics=MetricsProcessor.Config(
            loggers=[TensorBoardLogger.Config()],
            log_freq=1,
        ),
        checkpointer=None,
    )


def deepseek_v4_1_debugmodel_seed() -> Trainer.Config:
    config = deepseek_v4_1_debugmodel()
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    return config
