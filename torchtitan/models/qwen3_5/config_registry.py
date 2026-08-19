# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.data import GrainDataLoader, SingleDatasetConfig
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.components.tokenizer import MultiModalTokenizer

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator
from torchtitan.hf_datasets.multimodal.mm_datasets import (
    MM_DATASETS,
    MultiModalProcessor,
)
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.trainer import Trainer

from . import model_registry, QWEN3_5_SPECIAL_TOKENS


def _multimodal_collator_config(
    dataset_config: SingleDatasetConfig,
) -> MultiModalCollator.Config:
    processor_config = dataset_config.processor
    assert isinstance(processor_config, MultiModalProcessor.Config)
    return replace(
        MultiModalCollator.Config(build_mrope_positions=True),
        patch_size=processor_config.patch_size,
        temporal_patch_size=processor_config.temporal_patch_size,
        spatial_merge_size=processor_config.spatial_merge_size,
    )


def qwen35_debugmodel() -> Trainer.Config:
    model_spec = model_registry("debugmodel")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=MM_DATASETS["cc12m-test"],
            collator=_multimodal_collator_config(MM_DATASETS["cc12m-test"]),
            streaming_shuffle_buffer_size=128,
        ),
        optimizer=default_adamw(lr=5e-3),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=512,
            steps=10,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def qwen35_debugmodel_varlen_attn() -> Trainer.Config:
    config = qwen35_debugmodel()
    config.model_spec = model_registry("debugmodel", attn_backend="varlen")
    config.training.disable_cuda_graphs = True
    return config


def qwen35_debugmodel_moe() -> Trainer.Config:
    model_spec = model_registry("debugmodel_moe", moe_comm_backend="standard")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=MM_DATASETS["cc12m-test"],
            collator=_multimodal_collator_config(MM_DATASETS["cc12m-test"]),
            streaming_shuffle_buffer_size=128,
        ),
        optimizer=default_adamw(lr=5e-3),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=512,
            steps=10,
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=2,
            pipeline_parallel_degree=2,
            expert_parallel_degree=4,
            tensor_parallel_degree=2,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def qwen35_0_8b() -> Trainer.Config:
    model_spec = model_registry("0.8B")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3.5-0.8B",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS),
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=MM_DATASETS["cc12m"],
            collator=_multimodal_collator_config(MM_DATASETS["cc12m"]),
            streaming_shuffle_buffer_size=128,
        ),
        optimizer=default_adamw(lr=5e-3),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=1000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def qwen35_2b() -> Trainer.Config:
    model_spec = model_registry("2B")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3.5-2B",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS),
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=MM_DATASETS["cc12m"],
            collator=_multimodal_collator_config(MM_DATASETS["cc12m"]),
            streaming_shuffle_buffer_size=128,
        ),
        optimizer=default_adamw(lr=5e-3),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=1000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def qwen35_4b() -> Trainer.Config:
    model_spec = model_registry("4B")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3.5-4B",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS),
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=MM_DATASETS["cc12m"],
            collator=_multimodal_collator_config(MM_DATASETS["cc12m"]),
            streaming_shuffle_buffer_size=128,
        ),
        optimizer=default_adamw(lr=5e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=1000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
        ),
        activation_checkpoint=FullAC.Config(),
    )


def qwen35_9b() -> Trainer.Config:
    model_spec = model_registry("9B")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3.5-9B",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS),
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=MM_DATASETS["cc12m"],
            collator=_multimodal_collator_config(MM_DATASETS["cc12m"]),
            streaming_shuffle_buffer_size=128,
        ),
        optimizer=default_adamw(lr=5e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=1000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=2,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=FullAC.Config(),
    )


def qwen35_27b() -> Trainer.Config:
    model_spec = model_registry("27B")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3.5-27B",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS),
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=MM_DATASETS["cc12m"],
            collator=_multimodal_collator_config(MM_DATASETS["cc12m"]),
            streaming_shuffle_buffer_size=128,
        ),
        optimizer=default_adamw(lr=5e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=1000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=4,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=FullAC.Config(),
    )


def qwen35_35b_a3b() -> Trainer.Config:
    model_spec = model_registry("35B-A3B", moe_comm_backend="standard")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3.5-35B-A3B",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS),
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=MM_DATASETS["cc12m"],
            collator=_multimodal_collator_config(MM_DATASETS["cc12m"]),
            streaming_shuffle_buffer_size=128,
        ),
        optimizer=default_adamw(lr=5e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=1000,
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=2,
            expert_parallel_degree=8,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=FullAC.Config(),
    )


def qwen35_122b_a10b() -> Trainer.Config:
    model_spec = model_registry("122B-A10B", moe_comm_backend="standard")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3.5-122B-A10B",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS),
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=MM_DATASETS["cc12m"],
            collator=_multimodal_collator_config(MM_DATASETS["cc12m"]),
            streaming_shuffle_buffer_size=128,
        ),
        optimizer=default_adamw(lr=5e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=1000,
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=4,
            expert_parallel_degree=8,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=FullAC.Config(),
    )


def qwen35_397b_a17b() -> Trainer.Config:
    model_spec = model_registry("397B-A17B", moe_comm_backend="standard")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3.5-397B-A17B",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS),
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=MM_DATASETS["cc12m"],
            collator=_multimodal_collator_config(MM_DATASETS["cc12m"]),
            streaming_shuffle_buffer_size=128,
        ),
        optimizer=default_adamw(lr=5e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=1000,
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=8,
            expert_parallel_degree=16,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=FullAC.Config(),
    )
