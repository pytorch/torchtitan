# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer (run) configs for Kimi K2.5.

Each function returns a ``Trainer.Config`` that selects a model flavor (via
``model_registry``) and bolts on data, optimizer, parallelism, and checkpoint
settings. ``kimi_k2_5_debugmodel`` / ``_debugmodel_mm`` are small debug configs
(text / image-text); ``kimi_k2_5_1t_a32b`` is a template for the full ~1T model.
"""

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedCELoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.quantization import (
    Float8GroupedExpertsConverter,
    Float8LinearConverter,
)
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.hf_datasets.multimodal.mm_datasets import MMDataLoader
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.trainer import Trainer

from . import KIMI_K2_5_SPECIAL_TOKENS, model_registry


def _mm_dataloader(dataset: str, **kwargs) -> MMDataLoader.Config:
    """Image-text dataloader matching MoonViT3d's patch geometry.

    ``patch_size=14`` and ``temporal_patch_size=1`` (MoonViT uses a 2D Conv/
    Linear patch embed, not a temporal-stacked one) give ``patch_dim = 3*14*14
    = 588``, matching the encoder. ``spatial_merge_size=2`` matches
    ``merge_kernel_size=(2, 2)``.

    ``image_mean``/``image_std`` = 0.5/0.5/0.5 are Kimi K2.5's real values
    (from the released ``preprocessor_config.json``). The collator emits
    block-order patches — the ``debugmodel_mm`` flavor's encoder reorders them
    to raster (the real Kimi processor emits raster directly).

    TODO: image resize policy does not match Kimi K2.5's real processor. The
    shared MMDataLoader uses ``smart_resize(min_pixels, max_pixels)``, whereas
    Kimi's ``navit_resize`` (media_utils) caps by ``in_patch_limit=16384`` and
    ``patch_limit_on_one_side=512`` (``in_patch_limit_each_frame=4096`` for
    video). These yield different resolutions / patch counts, so this dataloader
    is only correct for from-scratch training — matching real Kimi behavior (or
    loading its checkpoint) needs Kimi's navit resize.
    """
    return MMDataLoader.Config(
        dataset=dataset,
        max_images_per_batch=128,
        patch_size=14,
        temporal_patch_size=1,
        spatial_merge_size=2,
        # TODO: smart_resize(min/max_pixels) here is NOT Kimi's resize policy
        # (navit_resize: in_patch_limit=16384, patch_limit_on_one_side=512).
        min_pixels=65536,
        max_pixels=16777216,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        **kwargs,
    )


def kimi_k2_5_debugmodel() -> Trainer.Config:
    """Small text-only debug config.

    Exercises the DeepSeekV3 MLA + MoE decoder; the vision encoder is built but
    unused (no pixels in the text dataset).
    """
    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("debugmodel"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
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
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )


def kimi_k2_5_debugmodel_mm() -> Trainer.Config:
    """Image-text (multimodal) debug training, mirroring qwen3_5's MM configs.

    Uses the shared ``MMDataLoader`` + ``MultiModalTokenizer`` and the
    ``debugmodel_mm`` flavor (block→raster patch reorder). Image-only (cc12m);
    Kimi's video-chunk pipeline is not wired. Uses the real
    ``KIMI_K2_5_SPECIAL_TOKENS`` (the media tokens are present in the shared
    test tokenizer).
    """
    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=MultiModalTokenizer.Config(**KIMI_K2_5_SPECIAL_TOKENS),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("debugmodel_mm"),
        dataloader=_mm_dataloader("cc12m-test"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
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
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )


def kimi_k2_5_1t_a32b() -> Trainer.Config:
    """Full Kimi K2.5 (~1T-total / ~32B-active).

    A starting template for a real run: ``hf_assets_path`` (the Kimi K2.5
    tokenizer), dataset, and the parallelism degrees are placeholders to set
    for the target cluster. This model needs many GPUs — typically FSDP + EP
    (+ PP), not a single host.
    """
    compile_config = CompileConfig(enable=True, components=["loss"])
    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )
    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./assets/hf/Kimi-K2.5",
        model_spec=model_registry(
            "1T-A32B",
            attn_backend="flex",
            converters=[
                Float8LinearConverter.Config(
                    filter_fqns=["output", "router.gate"],
                    model_compile_enabled=model_compile_enabled,
                ),
                Float8GroupedExpertsConverter.Config(
                    model_compile_enabled=model_compile_enabled
                ),
            ],
        ),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        optimizer=OptimizersContainer.Config(lr=2.2e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=10000,
        ),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
            expert_parallel_degree=8,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="full",
        ),
        compile=compile_config,
    )
