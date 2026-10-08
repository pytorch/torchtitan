# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5 model configurations used by tests."""

from dataclasses import replace

from torchtitan.components.data import GrainDataLoader, SingleDatasetConfig
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
from torchtitan.config.transform import apply_transforms, LoRATransform
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator
from torchtitan.hf_datasets.multimodal.mm_datasets import MM_DATASETS, VisionProcessor
from torchtitan.models.common.config_utils import (
    decoder_vocab_size,
    DEFAULT_DEBUG_MODEL_SEQ_LEN,
)

from torchtitan.models.qwen3_5 import build_model_config, QWEN3_5_SPECIAL_TOKENS
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

from torchtitan_recipes.tests.datasets.synthetic_multimodal import SYNTHETIC_MM_DATASETS


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


def qwen35_debugmodel(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    model_config = build_model_config("debugmodel", seq_len=seq_len)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS),
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
        activation_checkpoint=SelectiveAC.Config(),
    )


def qwen35_debugmodel_video(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    config = qwen35_debugmodel(seq_len=seq_len)
    dataset = SYNTHETIC_MM_DATASETS["synthetic-video-text"]
    config.dataloader = replace(
        config.dataloader,
        dataset=dataset,
        collator=_multimodal_collator_config(dataset),
    )
    config.training.disable_cuda_graphs = True
    return config


def qwen35_debugmodel_varlen_attn(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    config = qwen35_debugmodel(seq_len=seq_len)
    config.model = build_model_config(
        "debugmodel", seq_len=seq_len, attn_backend="varlen"
    )
    config.training.disable_cuda_graphs = True
    return config


def qwen35_debugmodel_moe(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    model_config = build_model_config("debugmodel_moe", seq_len=seq_len)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS),
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
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=2,
            pipeline_parallel_degree=2,
            num_pp_microbatches=2,
            expert_parallel_degree=4,
            tensor_parallel_degree=2,
        ),
        checkpointer=None,
        activation_checkpoint=SelectiveAC.Config(),
    )


def qwen35_debugmodel_moe_lora(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    config = qwen35_debugmodel_moe(seq_len=seq_len)
    return apply_transforms(
        config,
        [
            LoRATransform(
                rank=8,
                alpha=16.0,
                target_modules=["w13", "w2"],
            )
        ],
    )
