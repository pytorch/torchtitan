# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.8 model configurations used by tests."""

from dataclasses import replace

from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    SingleDatasetConfig,
)
from torchtitan.components.data.multimodal.mm_collator import MultiModalCollator
from torchtitan.components.data.multimodal.mm_datasets import (
    MM_DATASETS,
    VisionProcessor,
)
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import (
    DefaultActivationRematConfig,
    FullActivationRematConfig,
)
from torchtitan.models.common.config_utils import (
    decoder_vocab_size,
    DEFAULT_DEBUG_MODEL_SEQ_LEN,
)
from torchtitan.models.qwen3_8 import build_model_config, QWEN3_8_SPECIAL_TOKENS
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

from torchtitan_recipes.data import c4_document_capacity


def _multimodal_collator_config(
    dataset_config: SingleDatasetConfig,
) -> MultiModalCollator.Config:
    processor_config = dataset_config.processor
    assert isinstance(processor_config, VisionProcessor.Config)
    return replace(
        MultiModalCollator.Config(build_mrope_positions=True),
        patch_size=processor_config.patch_size,
        temporal_patch_size=processor_config.temporal_patch_size,
        spatial_merge_size=processor_config.spatial_merge_size,
    )


def qwen38_debugmodel(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    parallelism = ParallelismConfig()
    model_config = build_model_config(
        "debugmodel", seq_len=seq_len, attn_backend="flex", parallelism=parallelism
    )
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_8_SPECIAL_TOKENS),
        metrics=MetricsProcessor.Config(log_freq=1),
        model=model_config,
        parallelism=parallelism,
        dataloader=GrainDataLoader.Config(
            dataset=MM_DATASETS["cc12m-test"],
            collator=_multimodal_collator_config(MM_DATASETS["cc12m-test"]),
            streaming_shuffle_buffer_size=128,
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=5e-3)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_ratio=0.8,
                decay_type="linear",
                min_lr_factor=0.0,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=1 * model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=10,
        ),
        checkpointer=None,
        activation_remat=DefaultActivationRematConfig(),
    )


def qwen38_debugmodel_varlen_attn(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    config = qwen38_debugmodel(seq_len=seq_len)
    config.model = build_model_config(
        "debugmodel",
        seq_len=seq_len,
        attn_backend="varlen",
        parallelism=config.parallelism,
    )
    config.training.disable_accelerator_graphs = True
    return config


def qwen38_debugmodel_moe(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=2,
        pipeline_parallel_degree=2,
        num_pp_microbatches=2,
        expert_parallel_degree=4,
        tensor_parallel_degree=2,
    )
    model_config = build_model_config(
        "debugmodel_moe", seq_len=seq_len, attn_backend="flex", parallelism=parallelism
    )
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_8_SPECIAL_TOKENS),
        metrics=MetricsProcessor.Config(log_freq=1),
        model=model_config,
        dataloader=GrainDataLoader.Config(
            dataset=MM_DATASETS["cc12m-test"],
            collator=_multimodal_collator_config(MM_DATASETS["cc12m-test"]),
            streaming_shuffle_buffer_size=128,
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=5e-3)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=1 * model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=10,
            disable_accelerator_graphs=True,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=DefaultActivationRematConfig(),
    )


def qwen38_27b(seq_len: int | None = None) -> Trainer.Config:
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=-1,
        tensor_parallel_degree=4,
    )
    model_config = build_model_config(
        "27B", seq_len=seq_len, attn_backend="flex", parallelism=parallelism
    )
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3.8-27B",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_8_SPECIAL_TOKENS),
        model=model_config,
        dataloader=GrainDataLoader.Config(
            dataset=MM_DATASETS["cc12m"],
            collator=_multimodal_collator_config(MM_DATASETS["cc12m"]),
            streaming_shuffle_buffer_size=128,
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=5e-4)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=4 * model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=1000,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=FullActivationRematConfig(),
    )


def qwen38_2_4t_a95b(seq_len: int | None = None) -> Trainer.Config:
    """Qwen3.8-2.4T-A95B text-only MoE training config."""
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=-1,
        tensor_parallel_degree=4,
        expert_parallel_degree=64,
    )
    model_config = build_model_config(
        "2.4T-A95B", seq_len=seq_len, parallelism=parallelism
    )
    num_tokens_per_microbatch_per_dp_rank = model_config.max_context_length
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3.8-2.4T-A95B",
        model=model_config,
        dataloader=GrainDataLoader.Config(
            max_num_documents=c4_document_capacity(
                num_tokens_per_microbatch_per_dp_rank
            ),
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=5e-4)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=num_tokens_per_microbatch_per_dp_rank,
            max_context_length=model_config.max_context_length,
            steps=1000,
            disable_accelerator_graphs=True,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=FullActivationRematConfig(),
    )
