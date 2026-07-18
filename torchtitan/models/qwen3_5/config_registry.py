# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.data.dataset import SingleDatasetConfig
from torchtitan.components.data.loader import GrainDataLoader
from torchtitan.components.data.sources import HuggingFaceStreamingSource
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.components.tokenizer import MultiModalTokenizer

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.hf_datasets.multimodal.mm_collator import QwenMultimodalCollator
from torchtitan.hf_datasets.multimodal.mm_datasets import (
    QwenCC12MProcessor,
    QwenMultimodalPackingConfig,
    QwenObelicsProcessor,
)
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.trainer import Trainer

from . import model_registry, QWEN3_5_SPECIAL_TOKENS


def _dataloader(
    dataset: str,
    *,
    dataset_path: str | None = None,
    dataset_subset: str = "",
    max_images_per_batch: int = 128,
    max_patches_per_batch: int = 8_388_608,
) -> GrainDataLoader.Config:
    dataset = dataset.lower()
    if dataset == "obelics":
        source_path = dataset_path or "HuggingFaceM4/OBELICS"
        source_kwargs = {"split": "train"}
        processor = QwenObelicsProcessor.Config()
    elif dataset in {"cc12m", "cc12m-test"}:
        source_path = dataset_path or (
            "tests/assets/cc12m_test"
            if dataset == "cc12m-test"
            else "pixparse/cc12m-wds"
        )
        source_kwargs = {"split": "train"}
        if dataset == "cc12m-test":
            source_kwargs["data_files"] = {"train": "*.tar"}
        processor = QwenCC12MProcessor.Config()
    else:
        raise ValueError(f"Unsupported Qwen multimodal dataset: {dataset}")
    if dataset_subset:
        source_kwargs["name"] = dataset_subset

    process = processor
    source = HuggingFaceStreamingSource.Config(
        path=source_path,
        load_dataset_kwargs=source_kwargs,
    )
    packing = QwenMultimodalPackingConfig(
        dataset=SingleDatasetConfig(source=source, process=process),
        max_images_per_batch=max_images_per_batch,
        max_patches_per_batch=max_patches_per_batch,
        patch_size=16,
        temporal_patch_size=2,
    )
    return GrainDataLoader.Config(
        dataset=packing,
        collator=QwenMultimodalCollator.Config(
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            build_mrope_positions=True,
        ),
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
        dataloader=_dataloader("cc12m-test"),
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
        dataloader=_dataloader("cc12m-test"),
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
        dataloader=_dataloader("cc12m"),
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
        dataloader=_dataloader("cc12m"),
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
        dataloader=_dataloader("cc12m"),
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
        dataloader=_dataloader("cc12m"),
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
        dataloader=_dataloader("cc12m"),
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
        dataloader=_dataloader("cc12m"),
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
        dataloader=_dataloader("cc12m"),
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
        dataloader=_dataloader("cc12m"),
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
