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
            spmd_typechecking=parallelism.pipeline_parallel_degree == 1,
        ),
        parallelism=parallelism,
        profiler=Profiler.Config(enable_profiling=False),
        metrics=MetricsProcessor.Config(
            loggers=[TensorBoardLogger.Config()],
            log_freq=1,
        ),
        checkpointer=None,
    )


def deepseek_v4_1_debugmodel_fsdp2_tp2_pp2_ep2(
    *, attention_backend: str = "reference"
) -> Trainer.Config:
    config = deepseek_v4_1_debugmodel(
        attention_backend=attention_backend,
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=2,
            tensor_parallel_degree=2,
            pipeline_parallel_degree=2,
            pipeline_parallel_first_stage_less_layers=0,
            pipeline_parallel_last_stage_less_layers=0,
            pipeline_parallel_layers_per_stage=3,
            pipeline_parallel_schedule="Interleaved1F1B",
            num_pp_microbatches=4,
            expert_parallel_degree=2,
        ),
    )
    config.training.num_tokens_per_microbatch_per_dp_rank = 64
    return config


def deepseek_v4_1_debugmodel_fsdp2_tp2_pp2_ep2_fused() -> Trainer.Config:
    return deepseek_v4_1_debugmodel_fsdp2_tp2_pp2_ep2(attention_backend="fused")


def deepseek_v4_1_debugmodel_seed() -> Trainer.Config:
    config = deepseek_v4_1_debugmodel()
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    return config


def deepseek_v4_1_debugmodel_tp2_ep2() -> Trainer.Config:
    return deepseek_v4_1_debugmodel(
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=1,
            tensor_parallel_degree=2,
            expert_parallel_degree=2,
        )
    )


def deepseek_v4_1_debugmodel_pp2() -> Trainer.Config:
    config = deepseek_v4_1_debugmodel(
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=1,
            pipeline_parallel_degree=2,
            pipeline_parallel_first_stage_less_layers=0,
            pipeline_parallel_last_stage_less_layers=0,
            pipeline_parallel_layers_per_stage=3,
            pipeline_parallel_schedule="Interleaved1F1B",
            num_pp_microbatches=4,
        )
    )
    config.training.num_tokens_per_microbatch_per_dp_rank = 64
    return config
