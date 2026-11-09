# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muse Glimmer model configurations used by tests."""

from typing import Any

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.multimodal.mm_collator import MultiModalCollator
from torchtitan.components.data.multimodal.mm_datasets import MM_DATASETS
from torchtitan.components.data.multimodal.vision_processor import (
    PixelBudgetResize,
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

from torchtitan.models.muse_glimmer import build_model_config
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

from torchtitan_recipes.data import c4_document_capacity
from torchtitan_recipes.models.muse_spark import MuseGlimmerPatchGridResize


# Multimodal special tokens for the Muse Glimmer debug flavor. The debug tokenizer asset
# (``./tests/assets/tokenizer``) already defines these (IDs 2004-2008, all <
# debugmodel_mm's vocab_size=2048), so no asset change is needed.
MUSE_GLIMMER_DEBUG_SPECIAL_TOKENS = {
    "image_token": "<|image_pad|>",
    "video_token": "<|video_pad|>",
    "vision_start_token": "<|vision_start|>",
    "vision_end_token": "<|vision_end|>",
    "pad_token": "<|endoftext|>",
}

# ``VisionProcessor.Config`` fields, used like the special tokens:
# ``VisionProcessor.Config(**MUSE_GLIMMER_DEBUG_VISION_PROCESSOR_KWARGS)``. The patch
# sizes match the NaViT encoder (``patch_size``, ``patch_temporal``,
# ``downsample_factor``), and raster order matches its row-major patch layout.
MUSE_GLIMMER_DEBUG_VISION_PROCESSOR_KWARGS: dict[str, Any] = {
    "patch_size": 14,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "image_mean": (0.5, 0.5, 0.5),
    "image_std": (0.5, 0.5, 0.5),
    "resize": PixelBudgetResize.Config(min_pixels=784, max_pixels=3136),
    "patch_order": "raster",
}

# Multimodal special tokens of the released Muse Glimmer 30B tokenizer.
MUSE_GLIMMER_30B_SPECIAL_TOKENS = {
    "image_token": "<|patch|>",
    "video_token": "<|video|>",
    "vision_start_token": "<|image_start|>",
    "vision_end_token": "<|image_end|>",
    "pad_token": "<|finetune_right_pad|>",
}

# The released Muse Glimmer 30B image processor: closest-aspect patch grid up to
# 4096 merged tokens, LANCZOS. Images only.
MUSE_GLIMMER_30B_VISION_PROCESSOR_KWARGS: dict[str, Any] = {
    "patch_size": 14,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "image_mean": (0.5, 0.5, 0.5),
    "image_std": (0.5, 0.5, 0.5),
    "resize": MuseGlimmerPatchGridResize.Config(max_image_tokens=4096),
    "patch_order": "raster",
    "image_interpolation_mode": "lanczos",
}


def _muse_glimmer_mm_dataloader(dataset_name: str) -> GrainDataLoader.Config:
    """Build the shared multimodal dataloader config. The trainer config also
    needs a ``vision_processor``, e.g. built from
    ``MUSE_GLIMMER_DEBUG_VISION_PROCESSOR_KWARGS``.

    ``build_mrope_positions=False`` since Muse Glimmer uses 1D ComplexRoPE on
    the LLM side, not MRoPE.
    """
    return GrainDataLoader.Config(
        dataset=MM_DATASETS[dataset_name],
        collator=MultiModalCollator.Config(
            max_images_per_microbatch=8,
            build_mrope_positions=False,
        ),
    )


def muse_glimmer_debugmodel(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
    *,
    attn_backend: str = "varlen",
) -> Trainer.Config:
    parallelism = ParallelismConfig()
    model_config = build_model_config(
        "debugmodel",
        seq_len=seq_len,
        attn_backend=attn_backend,
        parallelism=parallelism,
    )
    num_tokens_per_microbatch_per_dp_rank = 8 * model_config.max_context_length
    # The output soft-cap lives in the SoftCappedLinear lm_head, so it is applied
    # per-chunk inside ChunkedLossWrapper just as it would be in the full model
    # forward.
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model=model_config,
        dataloader=GrainDataLoader.Config(
            max_num_documents=(
                c4_document_capacity(num_tokens_per_microbatch_per_dp_rank)
                if attn_backend == "varlen"
                else None
            ),
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
            shuffle=False,
        ),
        optim=Optim.Config(
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
            num_tokens_per_microbatch_per_dp_rank=num_tokens_per_microbatch_per_dp_rank,
            steps=10,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=DefaultActivationRematConfig(),
    )


def muse_glimmer_debugmodel_mm(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
    *,
    attn_backend: str = "varlen",
) -> Trainer.Config:
    """Multimodal debug training config.

    Trains the ``debugmodel_mm`` flavor (debug text decoder that owns a
    scaled-down vision encoder + adapter) end-to-end on the ``cc12m-test`` local
    tar fixture. The shared Grain data pipeline emits packed ``pixel_values`` +
    ``grid_thw`` + ``special_tokens``; preprocessing builds packed-bank
    indices from the image placeholder token. Vision-placeholder positions are
    already ``IGNORE_INDEX`` in the labels, so a standard ``CrossEntropyLoss``
    (wrapped in ``ChunkedLossWrapper``) is used.

    The integration smoke suite covers the TP+CP+PP+SP path.
    """
    parallelism = ParallelismConfig()
    mm_model_spec = build_model_config(
        "debugmodel_mm",
        seq_len=seq_len,
        attn_backend=attn_backend,
        parallelism=parallelism,
    )
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(mm_model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=MultiModalTokenizer.Config(**MUSE_GLIMMER_DEBUG_SPECIAL_TOKENS),
        metrics=MetricsProcessor.Config(log_freq=1),
        model=mm_model_spec,
        vision_processor=VisionProcessor.Config(
            **MUSE_GLIMMER_DEBUG_VISION_PROCESSOR_KWARGS
        ),
        dataloader=_muse_glimmer_mm_dataloader("cc12m-test"),
        optim=Optim.Config(
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
            num_tokens_per_microbatch_per_dp_rank=4 * mm_model_spec.max_context_length,
            steps=10,
            disable_accelerator_graphs=True,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=DefaultActivationRematConfig(),
    )


def muse_glimmer_30b(seq_len: int | None = None) -> Trainer.Config:
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=-1,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
    )
    model_config = build_model_config("30B", seq_len=seq_len, parallelism=parallelism)
    num_tokens_per_microbatch_per_dp_rank = model_config.max_context_length
    return Trainer.Config(
        # ChunkedLossWrapper avoids materializing the full [T, vocab] logits;
        # the soft-cap is in the SoftCappedLinear lm_head, so it is still applied
        # per-chunk.
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./assets/hf/Muse-Glimmer-30B",
        model=model_config,
        dataloader=GrainDataLoader.Config(
            max_num_documents=c4_document_capacity(
                num_tokens_per_microbatch_per_dp_rank
            ),
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=3e-4)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(warmup_steps=200),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=num_tokens_per_microbatch_per_dp_rank,
            steps=1000,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=FullActivationRematConfig(),
    )


def muse_glimmer_30b_mm(seq_len: int | None = None) -> Trainer.Config:
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=-1,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
    )
    model_config = build_model_config(
        "30B_mm", seq_len=seq_len, parallelism=parallelism
    )
    return Trainer.Config(
        # ChunkedLossWrapper avoids materializing the full [T, vocab] logits;
        # the soft-cap is in the SoftCappedLinear lm_head, so it is still applied
        # per-chunk.
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./assets/hf/Muse-Glimmer-30B",
        tokenizer=MultiModalTokenizer.Config(**MUSE_GLIMMER_30B_SPECIAL_TOKENS),
        model=model_config,
        # The released image processor, not the debug pixel budget.
        vision_processor=VisionProcessor.Config(
            **MUSE_GLIMMER_30B_VISION_PROCESSOR_KWARGS
        ),
        dataloader=_muse_glimmer_mm_dataloader("cc12m"),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=3e-4)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(warmup_steps=200),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=1 * model_config.max_context_length,
            steps=1000,
            disable_accelerator_graphs=True,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=FullActivationRematConfig(),
    )
