# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fault tolerant FSDP Llama 3 recipes.

Parallelism is FSDP only over the active hosts; spares join the lighthouse
and take over a failed host's shards.
"""

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    OptimizersContainer,
)
from torchtitan.config import CommConfig, DebugConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.ft_fsdp.config import FTFSDPConfig
from torchtitan.experiments.ft_fsdp.optim import FTOptim
from torchtitan.experiments.ft_fsdp.trainer import FTFSDPTrainer
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.config_utils import (
    decoder_vocab_size,
    DEFAULT_DEBUG_MODEL_SEQ_LEN,
)
from torchtitan.models.llama3 import build_model_config
from torchtitan.observability.metrics import MetricsProcessor


def llama3_debugmodel_local() -> FTFSDPTrainer.Config:
    """3 active hosts plus 1 spare, one GPU per host."""
    return _debugmodel(FTFSDPConfig(num_active_hosts=3, num_hosts=4, procs_per_host=1))


def llama3_debugmodel_local_2proc() -> FTFSDPTrainer.Config:
    """2 hosts with 2 GPUs each and no spare; a failed host must restart and
    take its slot back."""
    return _debugmodel(FTFSDPConfig(num_active_hosts=2, num_hosts=2, procs_per_host=2))


def _debugmodel(ft_fsdp: FTFSDPConfig) -> FTFSDPTrainer.Config:
    model_config = build_model_config("debugmodel", seq_len=DEFAULT_DEBUG_MODEL_SEQ_LEN)
    return FTFSDPTrainer.Config(
        loss=CrossEntropyLoss.Config(
            global_vocab_size=decoder_vocab_size(model_config),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model=model_config,
        optim=FTOptim.Config(
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
            num_tokens_per_microbatch_per_dp_rank=8 * model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=1000,
            disable_cuda_graphs=True,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
        ),
        checkpointer=None,
        activation_checkpoint=SelectiveAC.Config(),
        parallelism=ParallelismConfig(data_parallel_shard_degree=-1),
        comm=CommConfig(init_timeout_seconds=120, train_timeout_seconds=20),
        debug=DebugConfig(seed=42),
        ft_fsdp=ft_fsdp,
        validator=None,
    )


def llama3_8b_mast() -> FTFSDPTrainer.Config:
    """32 active hosts plus 2 spares, 8 GPUs per host (256 training GPUs)."""
    return _llama3_8b(num_active_hosts=32, num_hosts=34, steps=3000)


def llama3_8b_mast_preflight() -> FTFSDPTrainer.Config:
    """2 active hosts plus 1 spare, 8 GPUs per host."""
    return _llama3_8b(num_active_hosts=2, num_hosts=3, steps=400)


def _llama3_8b(
    *, num_active_hosts: int, num_hosts: int, steps: int
) -> FTFSDPTrainer.Config:
    # The test tokenizer and the C4 fixture keep runs free of downloads.
    model_config = build_model_config("8B", seq_len=8192)
    return FTFSDPTrainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=10),
        model=model_config,
        optim=FTOptim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=3e-4)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=steps,
            disable_cuda_graphs=True,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
        ),
        checkpointer=None,
        activation_checkpoint=SelectiveAC.Config(),
        parallelism=ParallelismConfig(data_parallel_shard_degree=-1),
        comm=CommConfig(init_timeout_seconds=300, train_timeout_seconds=30),
        debug=DebugConfig(seed=42),
        ft_fsdp=FTFSDPConfig(
            num_active_hosts=num_active_hosts,
            num_hosts=num_hosts,
            procs_per_host=8,
            meta_capacity_bytes=16 << 20,
        ),
        validator=None,
    )
