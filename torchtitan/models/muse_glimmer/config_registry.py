# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer

from . import model_registry
from .model import MuseGlimmerModel


# Multimodal special tokens for the Muse Glimmer debug flavor. The debug tokenizer asset
# (``./tests/assets/tokenizer``) already defines these (IDs 2004-2008, all <
# debugmodel_mm's vocab_size=2048), so no asset change is needed.
MUSE_GLIMMER_SPECIAL_TOKENS = {
    "image_token": "<|image_pad|>",
    "video_token": "<|video_pad|>",
    "vision_start_token": "<|vision_start|>",
    "vision_end_token": "<|vision_end|>",
    "pad_token": "<|endoftext|>",
}


def _muse_glimmer_mm_dataloader(model_spec: ModelSpec, dataset: str):
    """Build the shared multimodal dataloader config, taking the vision-patch
    geometry from the model's own vision encoder.

    ``patch_size``/``temporal_patch_size``/``spatial_merge_size`` must match the
    encoder exactly: they drive the image-placeholder token count the shared
    dataset inserts, which must equal the encoder's downsampled output token
    count. Deriving them from ``model_spec`` keeps loader and encoder aligned.
    ``patch_order="raster"`` matches the encoder's raster patch layout (row-major
    grid); ``build_mrope_positions=False`` since Muse Glimmer uses 1D ComplexRoPE
    on the LLM side, not MRoPE.

    NOTE: the multimodal dataloader imports (``MMDataLoader`` /
    ``resize_to_pixel_budget``) are done lazily here because they pull in
    ``torchvision`` (via ``torchtitan.hf_datasets.multimodal.utils.image``).
    ``torchvision`` is an optional dependency (not in requirements.txt /
    pyproject.toml), so importing it at module top level breaks the text-only
    configs (``muse_glimmer_debugmodel`` / ``muse_glimmer_30b``) for users who
    have not installed it. Keeping these imports inside the multimodal-only path
    lets the text configs load without torchvision.
    """
    from torchtitan.hf_datasets.multimodal.mm_datasets import MMDataLoader
    from torchtitan.hf_datasets.multimodal.utils.image import resize_to_pixel_budget

    model_config = model_spec.model
    assert isinstance(model_config, MuseGlimmerModel.Config)
    encoder = model_config.vision_encoder
    assert encoder is not None, "multimodal flavor must own a vision_encoder"
    return MMDataLoader.Config(
        dataset=dataset,
        max_images_per_batch=8,
        patch_size=encoder.patch_size,
        temporal_patch_size=encoder.patch_temporal,
        spatial_merge_size=encoder.downsample_factor,
        patch_order="raster",
        resize_fn=resize_to_pixel_budget,
        min_pixels=784,
        max_pixels=3136,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        max_patches=4096,
        max_patches_per_side=512,
        build_mrope_positions=False,
    )


def muse_glimmer_debugmodel() -> Trainer.Config:
    model_spec = model_registry("debugmodel", attn_backend="flex")
    # The output soft-cap lives in the SoftCappedLinear lm_head, so it is applied
    # per-chunk inside ChunkedLossWrapper just as it would be in the full model
    # forward.
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        parallelism=ParallelismConfig(spmd_backend="spmd_types"),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def muse_glimmer_debugmodel_mm() -> Trainer.Config:
    """Multimodal debug training config.

    Trains the ``debugmodel_mm`` flavor (debug text decoder that owns a
    scaled-down vision encoder + adapter) end-to-end on the ``cc12m-test`` local
    tar fixture. The shared :class:`MMDataLoader` emits padded ``pixel_values`` +
    ``grid_thw`` + ``special_tokens``; the model derives the vision-placeholder
    mask from ``special_tokens``. Vision-placeholder positions are already
    ``IGNORE_INDEX`` in the labels, so a standard ``CrossEntropyLoss`` (wrapped in
    ``ChunkedLossWrapper``) is used.

    The parallelism smoke suite covers FSDP and FSDP+TP+SP (see
    ``build_muse_glimmer_mm_test_list``); PP and CP are multimodal follow-ups.
    """
    mm_model_spec = model_registry("debugmodel_mm", attn_backend="flex")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(mm_model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=MultiModalTokenizer.Config(**MUSE_GLIMMER_SPECIAL_TOKENS),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=mm_model_spec,
        dataloader=_muse_glimmer_mm_dataloader(mm_model_spec, "cc12m-test"),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=512,
            steps=10,
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(spmd_backend="spmd_types"),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def muse_glimmer_30b() -> Trainer.Config:
    model_spec = model_registry("30B", attn_backend="flex")
    return Trainer.Config(
        # ChunkedLossWrapper avoids materializing the full [B, L, vocab] logits;
        # the soft-cap is in the SoftCappedLinear lm_head, so it is still applied
        # per-chunk.
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Muse-Glimmer-30B",
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        optimizer=default_adamw(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=200),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=8192,
            steps=1000,
        ),
        parallelism=ParallelismConfig(
            spmd_backend="spmd_types",
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=FullAC.Config(),
    )


def muse_glimmer_30b_mm() -> Trainer.Config:
    model_spec = model_registry("30B_mm", attn_backend="flex")
    return Trainer.Config(
        # ChunkedLossWrapper avoids materializing the full [B, L, vocab] logits;
        # the soft-cap is in the SoftCappedLinear lm_head, so it is still applied
        # per-chunk.
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Muse-Glimmer-30B",
        tokenizer=MultiModalTokenizer.Config(**MUSE_GLIMMER_SPECIAL_TOKENS),
        model_spec=model_spec,
        dataloader=_muse_glimmer_mm_dataloader(model_spec, "cc12m"),
        optimizer=default_adamw(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=200),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=8192,
            steps=1000,
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            spmd_backend="spmd_types",
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=FullAC.Config(),
    )
