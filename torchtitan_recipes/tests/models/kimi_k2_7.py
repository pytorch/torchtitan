# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K2.5 model configurations used by tests."""

from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    SingleDatasetConfig,
)
from torchtitan.components.data.multimodal.mm_collator import MultiModalCollator
from torchtitan.components.data.multimodal.mm_datasets import MM_DATASETS
from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import LRSchedulersContainer, Optim
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

from torchtitan.models.kimi_k2_7 import (
    build_model_config,
    KIMI_K2_5_SPECIAL_TOKENS,
    KIMI_K2_5_VISION_PROCESSOR_KWARGS,
)
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

from torchtitan_recipes.tests.models.dist_muon import (
    mla_dist_muon_optimizer,
    MLADistMuonTrainerConfig,
)


def _kimi_multimodal_dataloader(
    dataset: SingleDatasetConfig,
) -> GrainDataLoader.Config:
    return GrainDataLoader.Config(
        dataset=dataset,
        collator=MultiModalCollator.Config(
            max_images_per_microbatch=128,
            build_mrope_positions=False,
        ),
    )


def kimi_k2_5_debugmodel(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
    *,
    parallelism: ParallelismConfig | None = None,
) -> Trainer.Config:
    if parallelism is None:
        parallelism = ParallelismConfig()
    model_config = build_model_config(
        "debugmodel", seq_len=seq_len, parallelism=parallelism
    )
    return MLADistMuonTrainerConfig(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=MultiModalTokenizer.Config(**KIMI_K2_5_SPECIAL_TOKENS),
        metrics=MetricsProcessor.Config(log_freq=1),
        model=model_config,
        vision_processor=VisionProcessor.Config(**KIMI_K2_5_VISION_PROCESSOR_KWARGS),
        dataloader=_kimi_multimodal_dataloader(MM_DATASETS["cc12m-test"]),
        optim=Optim.Config(
            optimizer=mla_dist_muon_optimizer(
                model_config,
                muon_lr=8e-4,
                adamw_lr=8e-4,
                parallelism=parallelism,
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
            steps=10,
            disable_accelerator_graphs=True,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=DefaultActivationRematConfig(),
    )


def moonlight_16b_a3b(seq_len: int | None = None) -> Trainer.Config:
    """Moonlight 16B-A3B: the text-only Kimi K2 sibling (no vision tower)."""
    parallelism = ParallelismConfig(
        expert_parallel_degree=8,
    )
    model_config = build_model_config(
        "moonlight-16B-A3B",
        seq_len=seq_len,
        attn_backend="flex",
        parallelism=parallelism,
    )
    return MLADistMuonTrainerConfig(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./assets/hf/Moonlight-16B-A3B",
        model=model_config,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        optim=Optim.Config(
            optimizer=mla_dist_muon_optimizer(
                model_config,
                muon_lr=3e-4,
                adamw_lr=3e-4,
                parallelism=parallelism,
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2000,
                decay_ratio=0.8,
                decay_type="cosine",
                min_lr_factor=0.1,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=4 * model_config.max_context_length,
            steps=10000,
            disable_accelerator_graphs=True,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=FullActivationRematConfig(),
    )


def kimi_vl_a3b(seq_len: int | None = None) -> Trainer.Config:
    """Kimi-VL A3B: Moonlight text tower + 2D MoonViT vision (image-text)."""
    parallelism = ParallelismConfig(
        expert_parallel_degree=8,
    )
    model_config = build_model_config(
        "Kimi-VL-A3B", seq_len=seq_len, attn_backend="flex", parallelism=parallelism
    )
    return MLADistMuonTrainerConfig(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./assets/hf/Kimi-VL-A3B",
        # Kimi-VL-A3B names the vision-start token <|media_start|>, whereas the
        # K2.5 family uses <|media_begin|>; override just that one entry.
        tokenizer=MultiModalTokenizer.Config(
            **{**KIMI_K2_5_SPECIAL_TOKENS, "vision_start_token": "<|media_start|>"}
        ),
        model=model_config,
        # Kimi-VL is a compatibility flavor; resizing intentionally follows
        # Kimi-K2.5 per-side scaling instead of legacy Kimi-VL's side rejection.
        vision_processor=VisionProcessor.Config(**KIMI_K2_5_VISION_PROCESSOR_KWARGS),
        dataloader=_kimi_multimodal_dataloader(MM_DATASETS["cc12m"]),
        optim=Optim.Config(
            optimizer=mla_dist_muon_optimizer(
                model_config,
                muon_lr=3e-4,
                adamw_lr=3e-4,
                parallelism=parallelism,
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2000,
                decay_ratio=0.8,
                decay_type="cosine",
                min_lr_factor=0.1,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=1 * model_config.max_context_length,
            steps=10000,
            disable_accelerator_graphs=True,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=FullActivationRematConfig(),
    )


def kimi_k2_5(seq_len: int | None = None) -> Trainer.Config:
    """Full Kimi K2.5 (~1T-total / ~32B-active)."""
    # The report uses BF16 compute; its FP8 path only compresses saved activations.
    parallelism = ParallelismConfig(
        pipeline_parallel_schedule="Interleaved1F1B",
        expert_parallel_degree=8,
    )
    model_config = build_model_config(
        "Kimi-K2.5", seq_len=seq_len, attn_backend="flex", parallelism=parallelism
    )
    return MLADistMuonTrainerConfig(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./assets/hf/Kimi-K2.5",
        model=model_config,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        optim=Optim.Config(
            optimizer=mla_dist_muon_optimizer(
                model_config,
                muon_lr=2.2e-4,
                adamw_lr=2.2e-4,
                parallelism=parallelism,
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2000,
                decay_ratio=0.8,
                decay_type="cosine",
                min_lr_factor=0.1,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=4 * model_config.max_context_length,
            steps=10000,
            disable_accelerator_graphs=True,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=FullActivationRematConfig(),
    )
