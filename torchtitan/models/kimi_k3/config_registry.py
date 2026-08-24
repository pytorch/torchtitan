# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    SingleDatasetConfig,
)
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.attention import KDABackend, KDAInnerAttention
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.trainer import Trainer

from . import KIMI_K3_SPECIAL_TOKENS, model_registry


def _kimi_k3_multimodal_dataloader(
    dataset: SingleDatasetConfig,
) -> GrainDataLoader.Config:
    from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator
    from torchtitan.hf_datasets.multimodal.mm_datasets import MultiModalProcessor
    from torchtitan.hf_datasets.multimodal.utils.image import resize_to_patch_budget

    processor = dataset.processor
    if not isinstance(processor, MultiModalProcessor.Config):
        raise ValueError("Kimi K3 multimodal data requires MultiModalProcessor.Config")

    processor = MultiModalProcessor.Config(
        sample_processor=processor.sample_processor,
        patch_size=14,
        temporal_patch_size=1,
        spatial_merge_size=2,
        resize_fn=resize_to_patch_budget,
        min_pixels=56 * 56,
        max_pixels=224 * 224,
        max_patches=256,
        max_patches_per_side=16,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )
    return GrainDataLoader.Config(
        dataset=replace(dataset, processor=processor),
        collator=MultiModalCollator.Config(
            max_images_per_batch=8,
            patch_size=processor.patch_size,
            temporal_patch_size=processor.temporal_patch_size,
            spatial_merge_size=processor.spatial_merge_size,
            patch_order="raster",
            build_mrope_positions=False,
        ),
    )


def _kimi_k3_debugmodel_config(
    dataloader: GrainDataLoader.Config,
    *,
    kda_backend: KDABackend = "auto",
) -> Trainer.Config:
    model_spec = model_registry("debugmodel")
    for _fqn, inner_attention, _parent, _attr in model_spec.traverse(
        KDAInnerAttention.Config
    ):
        assert isinstance(inner_attention, KDAInnerAttention.Config)
        inner_attention.backend = kda_backend

    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=MultiModalTokenizer.Config(**KIMI_K3_SPECIAL_TOKENS),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=dataloader,
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        # TODO: Kimi K3 has no spmd_types annotations yet.
        parallelism=ParallelismConfig(spmd_backend="partial_dtensor"),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=256,
            max_context_length=256,
            steps=10,
            dtype="bfloat16",
            disable_cuda_graphs=True,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def kimi_k3_debugmodel() -> Trainer.Config:
    from torchtitan.hf_datasets.multimodal.mm_datasets import MM_DATASETS

    return _kimi_k3_debugmodel_config(
        _kimi_k3_multimodal_dataloader(MM_DATASETS["cc12m-test"])
    )


def _kimi_k3_c4_dataloader() -> GrainDataLoader.Config:
    return GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        shuffle=False,
    )


def kimi_k3_debugmodel_c4() -> Trainer.Config:
    return _kimi_k3_debugmodel_config(_kimi_k3_c4_dataloader())


def kimi_k3_debugmodel_c4_fused() -> Trainer.Config:
    return _kimi_k3_debugmodel_config(
        _kimi_k3_c4_dataloader(),
        kda_backend="fused",
    )


def kimi_k3_debugmodel_c4_fla() -> Trainer.Config:
    return _kimi_k3_debugmodel_config(
        _kimi_k3_c4_dataloader(),
        kda_backend="fla",
    )
