# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Hunks in this file are copied from upstream open PR 4322/4449/4450 (fegin's CP stack) to unblock running;
# pending rebase and reconcile.

from dataclasses import replace

from torchtitan.components.checkpointer import CheckpointManager
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


def qwen35_debugmodel(seq_len: int | None = None) -> Trainer.Config:
    model_spec = model_registry("debugmodel", seq_len=seq_len)
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
            num_tokens_per_microbatch_per_dp_rank=1 * model_spec.max_context_length,
            max_context_length=model_spec.max_context_length,
            steps=10,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def qwen35_debugmodel_varlen_attn(seq_len: int | None = None) -> Trainer.Config:
    config = qwen35_debugmodel(seq_len=seq_len)
    config.model_spec = model_registry(
        "debugmodel", seq_len=seq_len, attn_backend="varlen"
    )
    config.training.disable_cuda_graphs = True
    return config


def qwen35_debugmodel_moe(seq_len: int | None = None) -> Trainer.Config:
    model_spec = model_registry(
        "debugmodel_moe", seq_len=seq_len, moe_comm_backend="standard"
    )
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
            num_tokens_per_microbatch_per_dp_rank=1 * model_spec.max_context_length,
            max_context_length=model_spec.max_context_length,
            steps=10,
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=2,
            pipeline_parallel_degree=2,
            num_pp_microbatches=2,
            expert_parallel_degree=4,
            tensor_parallel_degree=2,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def qwen35_0_8b(seq_len: int | None = None) -> Trainer.Config:
    model_spec = model_registry("0.8B", seq_len=seq_len)
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
            num_tokens_per_microbatch_per_dp_rank=4 * model_spec.max_context_length,
            max_context_length=model_spec.max_context_length,
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


def qwen35_2b(seq_len: int | None = None) -> Trainer.Config:
    model_spec = model_registry("2B", seq_len=seq_len)
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
            num_tokens_per_microbatch_per_dp_rank=4 * model_spec.max_context_length,
            max_context_length=model_spec.max_context_length,
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


def qwen35_4b(seq_len: int | None = None) -> Trainer.Config:
    model_spec = model_registry("4B", seq_len=seq_len)
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
            num_tokens_per_microbatch_per_dp_rank=4 * model_spec.max_context_length,
            max_context_length=model_spec.max_context_length,
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


def qwen35_9b(seq_len: int | None = None) -> Trainer.Config:
    model_spec = model_registry("9B", seq_len=seq_len)
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
            num_tokens_per_microbatch_per_dp_rank=4 * model_spec.max_context_length,
            max_context_length=model_spec.max_context_length,
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


def qwen35_27b(seq_len: int | None = None) -> Trainer.Config:
    model_spec = model_registry("27B", seq_len=seq_len)
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
            num_tokens_per_microbatch_per_dp_rank=4 * model_spec.max_context_length,
            max_context_length=model_spec.max_context_length,
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


def qwen35_35b_a3b(seq_len: int | None = None) -> Trainer.Config:
    model_spec = model_registry("35B-A3B", seq_len=seq_len, moe_comm_backend="standard")
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
            num_tokens_per_microbatch_per_dp_rank=4 * model_spec.max_context_length,
            max_context_length=model_spec.max_context_length,
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


def qwen35_122b_a10b(seq_len: int | None = None) -> Trainer.Config:
    model_spec = model_registry(
        "122B-A10B", seq_len=seq_len, moe_comm_backend="standard"
    )
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
            num_tokens_per_microbatch_per_dp_rank=4 * model_spec.max_context_length,
            max_context_length=model_spec.max_context_length,
            steps=1000,
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            # n_kv_heads is 2, so the TP degree cannot exceed 2.
            tensor_parallel_degree=2,
            expert_parallel_degree=8,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=FullAC.Config(),
    )


def qwen35_397b_a17b(seq_len: int | None = None) -> Trainer.Config:
    model_spec = model_registry(
        "397B-A17B", seq_len=seq_len, moe_comm_backend="standard"
    )
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
            num_tokens_per_microbatch_per_dp_rank=4 * model_spec.max_context_length,
            max_context_length=model_spec.max_context_length,
            steps=1000,
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            # n_kv_heads is 2, so the TP degree cannot exceed 2.
            tensor_parallel_degree=2,
            expert_parallel_degree=16,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=FullAC.Config(),
    )
