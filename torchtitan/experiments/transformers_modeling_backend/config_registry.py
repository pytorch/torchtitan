# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    FirstFitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.config import DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.hf_datasets.text_datasets import ChatProcessor, DATASETS
from torchtitan.tools.profiler import Profiler
from torchtitan.trainer import Trainer
from . import model_registry
from .tokenizer import HFBackendTokenizer


@dataclass(kw_only=True, slots=True)
class TransformersBackendConfig(Trainer.Config):
    hf_model: str = ""
    """HuggingFace model ID (e.g., 'Qwen/Qwen2.5-7B')"""


def transformers_modeling_backend_debugmodel(
    seq_len: int = 2048,
) -> TransformersBackendConfig:
    model_spec = model_registry("debugmodel", seq_len=seq_len)
    return TransformersBackendConfig(
        loss=CrossEntropyLoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        hf_model="Qwen/Qwen3-4B-Instruct-2507",
        debug=DebugConfig(print_config=True),
        model_spec=model_spec,
        profiler=Profiler.Config(profile_freq=5),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
            max_context_length=seq_len,
            steps=10,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="1F1B",
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def transformers_modeling_backend_debugmodel_moe(
    seq_len: int = 2048,
) -> TransformersBackendConfig:
    return TransformersBackendConfig(
        loss=CrossEntropyLoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        hf_model="Qwen/Qwen3-30B-A3B",
        debug=DebugConfig(print_config=True),
        model_spec=model_registry("debugmodel_moe", seq_len=seq_len),
        profiler=Profiler.Config(profile_freq=5),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
            max_context_length=seq_len,
            steps=10,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="1F1B",
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def transformers_modeling_backend_full_moe(
    seq_len: int = 2048,
) -> TransformersBackendConfig:
    return TransformersBackendConfig(
        hf_model="Qwen/Qwen3-30B-A3B",
        debug=DebugConfig(print_config=True),
        model_spec=model_registry("full_moe", seq_len=seq_len),
        profiler=Profiler.Config(profile_freq=5),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=200,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
            max_context_length=seq_len,
            steps=1000,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        metrics=MetricsProcessor.Config(log_freq=10),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="1F1B",
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def transformers_modeling_backend_full(
    seq_len: int = 2048,
) -> TransformersBackendConfig:
    model_spec = model_registry("full", seq_len=seq_len)
    return TransformersBackendConfig(
        loss=CrossEntropyLoss.Config(),
        hf_model="Qwen/Qwen3-4B-Instruct-2507",
        debug=DebugConfig(print_config=True),
        model_spec=model_spec,
        profiler=Profiler.Config(profile_freq=5),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
            max_context_length=seq_len,
            steps=10,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="1F1B",
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def transformers_modeling_backend_sft_full(
    seq_len: int = 2048,
) -> TransformersBackendConfig:
    """SFT config with real HF pretrained weights loaded via initial_load_in_hf."""

    def process_sample(sample):
        return [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ]

    return TransformersBackendConfig(
        loss=CrossEntropyLoss.Config(),
        hf_assets_path="./tests/assets/qwen3_0.6b",
        hf_model="Qwen/Qwen3-0.6B",
        model_spec=model_registry("sft_full", seq_len=seq_len),
        tokenizer=HFBackendTokenizer.Config(),
        optimizer=default_adamw(lr=2e-5),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
            max_context_length=seq_len,
            steps=10,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=FirstFitPackingConfig(
                dataset=SingleDatasetConfig(
                    source=HuggingFaceRandomAccessSource.Config(
                        path="json",
                        split="train",
                        load_dataset_kwargs={
                            "data_files": "tests/assets/sft_test/data.json",
                        },
                    ),
                    processor=ChatProcessor.Config(messages_fn=process_sample),
                    post_filters=(lambda sample: sample is not None,),
                ),
            ),
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="1F1B",
        ),
        checkpoint=CheckpointManager.Config(
            enable=True,
            initial_load_in_hf=True,
            initial_load_model_only=True,
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def transformers_modeling_backend_sft_debugmodel(
    seq_len: int = 1024,
) -> TransformersBackendConfig:
    """SFT debug config for the transformers backend."""

    def process_sample(sample):
        return [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ]

    return TransformersBackendConfig(
        loss=CrossEntropyLoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        hf_model="Qwen/Qwen3-4B-Instruct-2507",
        model_spec=model_registry("sft_debugmodel", seq_len=seq_len),
        tokenizer=HFBackendTokenizer.Config(),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            # Keep this small: this debug model uses the full Qwen3 vocab
            # (~152k), so cross-entropy materializes a num_tokens * vocab
            # logits tensor. 16384 tokens is ~9GB in fp32 and OOMs the 22GB
            # CI GPUs.
            num_tokens_per_microbatch_per_dp_rank=1 * seq_len,
            max_context_length=seq_len,
            steps=10,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=FirstFitPackingConfig(
                dataset=SingleDatasetConfig(
                    source=HuggingFaceRandomAccessSource.Config(
                        path="json",
                        split="train",
                        load_dataset_kwargs={
                            "data_files": "tests/assets/sft_test/data.json",
                        },
                    ),
                    processor=ChatProcessor.Config(messages_fn=process_sample),
                    post_filters=(lambda sample: sample is not None,),
                ),
            ),
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="1F1B",
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )
