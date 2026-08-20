# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multimodal dataset processing for VLM training.

Workflow overview::

    HuggingFace Dataset (streaming)
            │
            ▼
    ┌───────────────────────────────────────────────────────┐
    │  Sample Processor  (MultiModalProcessor)              │
    │                                                       │
    │  1. Parse raw sample (dataset-specific format)        │
    │     e.g. OBELICS interleaved text/images,             │
    │          CC12M text-image pairs                       │
    │                                                       │
    │  2. Process vision: decode image/video bytes,         │
    │     resize to multiples of (patch_size * merge_size), │
    │     normalize with image_mean/std                     │
    │     → pixel_values: list[Tensor(T,H,W,C)]            │
    │                                                       │
    │  3. Process text: insert vision placeholder tokens    │
    │     <|vision_start|><|image_pad|>...<|vision_end|>    │
    │     into text, then tokenize                          │
    │     → input_ids: Tensor(num_tokens,)                  │
    │     → labels: next-token targets, with vision tokens  │
    │       masked to ignore_id (-100)                      │
    └───────────────────────────────────────────────────────┘
            │
            ▼  (optional, if MMSamplePackingConfig is configured)
    ┌───────────────────────────────────────────────────────┐
    │  Sample Packer                                        │
    │  Bin-pack short samples into seq_len-token sequences  │
    │  to reduce padding waste                              │
    └───────────────────────────────────────────────────────┘
            │
            ▼  GrainDataLoader batches samples (batch_size)
    ┌───────────────────────────────────────────────────────┐
    │  Collator  (MultiModalCollator)                    │
    │                                                       │
    │  1. collate_images: for each image Tensor(T,H,W,C),  │
    │     reshape into patches (num_patches, patch_dim),    │
    │     pad all images to same num_patches                │
    │     → pixel_values: (N, max_patches, patch_dim)       │
    │     → grid_thw: (N, 3) per-image [T, H', W'] dims    │
    │     (same for videos)                                 │
    │                                                       │
    │  2. collate_text: pad text fields to seq_len and      │
    │     pad to the target batch size                      │
    │     → input_ids: (batch_size, seq_len)                │
    │     → labels: (batch_size, seq_len)                   │
    └───────────────────────────────────────────────────────┘
            │
            ▼
    Model receives: {input_ids, pixel_values, grid_thw,
                     pixel_values_videos, grid_thw_videos,
                     special_tokens: dict[str, int]}, labels
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any

import grain.python as grain
import numpy as np
import torch
import tyro

from torchtitan.components.data.dataset import (
    DatasetConfig as GrainDatasetConfig,
    SampleProcessor,
    SingleDatasetConfig,
)
from torchtitan.components.data.sources import HuggingFaceStreamingSource
from torchtitan.components.data.types import DatasetBuildContext, DatasetIterationPolicy
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import MultiModalTokenizer

from torchtitan.tools.logging import logger
from .utils.image import calculate_vision_tokens, process_image, resize_to_pixel_budget
from .utils.text import insert_vision_placeholders


def _process_mm_sample(
    texts: list[str | None],
    images: list[bytes | None],
    tokenizer: MultiModalTokenizer,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
    min_pixels: int,
    max_pixels: int,
    image_mean: tuple[float, ...],
    image_std: tuple[float, ...],
    resize_fn: Callable[..., tuple[int, int, int, int]],
    max_patches: int,
    max_patches_per_side: int,
    **kwargs,
) -> dict[str, Any] | None:
    """Common processing logic for multimodal samples.

    Args:
        texts: List of strings with None indicating image positions
        images: List of image bytes with None for text positions
        tokenizer: Tokenizer for text processing
        patch_size: Size of image patches
        spatial_merge_size: merge 2D image patches to reduce LLM's sequence length.
            - if 1 (default): no merge, effectively NoOp
            - if 2: 2x2=4 image patches will be reduced to 1 LLM visual token

    Returns:
        Dict with:
            - input_ids: Tensor of token IDs
            - labels: Tensor of label IDs
            - pixel_values: List of processed image tensors

    Example:
        Interleaved format:
        texts = [text1, None, text2, None, text3]
        images = [None, img1, None, img2, None]

        Image-text pair format as a special case of interleaved:
        texts = [None, text]
        images = [image, None]
    """
    if not texts or len(texts) != len(images):
        return None

    processed_images = []
    num_image_tokens = []

    for idx, img in enumerate(images):
        if img is not None:
            # Resize (to multiples of patch_size x merge_size) and normalize images
            processed_img = process_image(
                img,
                patch_size=patch_size,
                merge_size=spatial_merge_size,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                image_mean=image_mean,
                image_std=image_std,
                resize_fn=resize_fn,
                max_patches=max_patches,
                max_patches_per_side=max_patches_per_side,
            )
            if processed_img is not None:
                num_tokens, _, _ = calculate_vision_tokens(
                    num_frames=1,
                    height=processed_img.shape[1],
                    width=processed_img.shape[2],
                    patch_size=patch_size,
                    spatial_merge_size=spatial_merge_size,
                    # TODO(data-mm-temporal-patches): Unify image/video token counting;
                    # the configured temporal patch size is unused by this image path.
                    temporal_patch_size=1,
                )
                processed_images.append(processed_img)
                num_image_tokens.append(num_tokens)
                # Keep the accepted image at this aligned position as a placeholder.
                texts[idx] = None

    if len(processed_images) != len([_ for _ in images if _ is not None]):
        logger.warning("Cannot process all images for sample. Dropping")
        return None

    # Replace image placeholders (None) with image token sequences
    processed_text = insert_vision_placeholders(
        texts,
        num_image_tokens,
        # pyrefly: ignore [missing-attribute]
        vision_start_token=tokenizer.vision_start_token,
        # pyrefly: ignore [missing-attribute]
        vision_token=tokenizer.image_token,
        # pyrefly: ignore [missing-attribute]
        vision_end_token=tokenizer.vision_end_token,
        # pyrefly: ignore [bad-argument-type]
        eos_token=tokenizer.eos_token,
    )

    tokens = tokenizer.encode(processed_text)
    if len(tokens) < 2:
        return None

    input_ids = torch.tensor(tokens[:-1])
    labels = torch.tensor(tokens[1:])

    special_token_ids = torch.tensor(
        [
            # pyrefly: ignore [missing-attribute]
            tokenizer.vision_start_id,
            # pyrefly: ignore [missing-attribute]
            tokenizer.vision_end_id,
            # pyrefly: ignore [missing-attribute]
            tokenizer.image_id,
            # pyrefly: ignore [missing-attribute]
            tokenizer.video_id,
        ]
    )
    labels = torch.where(torch.isin(labels, special_token_ids), IGNORE_INDEX, labels)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "positions": torch.arange(len(input_ids)),
        "pixel_values": processed_images,
    }


def _process_obelics_sample(
    sample: dict[str, Any],
    tokenizer: MultiModalTokenizer,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
    min_pixels: int,
    max_pixels: int,
    image_mean: tuple[float, ...],
    image_std: tuple[float, ...],
    **kwargs,
) -> dict[str, Any] | None:
    """Process a sample from the OBELICS dataset (interleaved text and images)."""
    return _process_mm_sample(
        texts=sample.get("texts", []),
        images=sample.get("images", []),
        tokenizer=tokenizer,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=spatial_merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        image_mean=image_mean,
        image_std=image_std,
        **kwargs,
    )


def _process_cc12_wd_sample(
    sample: dict[str, Any],
    tokenizer: MultiModalTokenizer,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
    min_pixels: int,
    max_pixels: int,
    image_mean: tuple[float, ...],
    image_std: tuple[float, ...],
    **kwargs,
) -> dict[str, Any] | None:
    """Process a sample from the CC12-WD dataset (text-image pairs)."""
    text = sample.get("txt", "")
    image = sample.get("jpg", None)

    texts = [None, text]
    images = [image, None]

    return _process_mm_sample(
        texts=texts,
        images=images,
        tokenizer=tokenizer,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=spatial_merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        image_mean=image_mean,
        image_std=image_std,
        **kwargs,
    )


class MultiModalProcessor(SampleProcessor):
    """Adapts a multimodal processor to Grain's map contract."""

    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        sample_processor: Annotated[Callable, tyro.conf.Suppress]
        patch_size: int = 16
        temporal_patch_size: int = 2
        spatial_merge_size: int = 2
        min_pixels: int = 65_536
        max_pixels: int = 16_777_216
        image_mean: tuple[float, ...] = (0.5, 0.5, 0.5)
        image_std: tuple[float, ...] = (0.5, 0.5, 0.5)
        resize_fn: Annotated[
            Callable[..., tuple[int, int, int, int]], tyro.conf.Suppress
        ] = resize_to_pixel_budget
        max_patches: int = 4096
        max_patches_per_side: int = 512
        video_dir: str = ""
        video_fps: float = 2.0
        video_min_frames: int = 4
        video_max_frames: int = 768

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        self._config = config
        self._tokenizer = context.tokenizer
        self._seq_len = context.seq_len

    def __call__(
        self,
        sample: dict[str, Any],
        rng: np.random.Generator,
    ) -> dict[str, Any] | None:
        del rng
        processed = self._config.sample_processor(
            sample=sample,
            tokenizer=self._tokenizer,
            patch_size=self._config.patch_size,
            temporal_patch_size=self._config.temporal_patch_size,
            spatial_merge_size=self._config.spatial_merge_size,
            min_pixels=self._config.min_pixels,
            max_pixels=self._config.max_pixels,
            image_mean=self._config.image_mean,
            image_std=self._config.image_std,
            resize_fn=self._config.resize_fn,
            max_patches=self._config.max_patches,
            max_patches_per_side=self._config.max_patches_per_side,
            video_dir=self._config.video_dir,
            video_fps=self._config.video_fps,
            video_min_frames=self._config.video_min_frames,
            video_max_frames=self._config.video_max_frames,
        )
        if processed is not None and processed["input_ids"].shape[0] > self._seq_len:
            logger.warning(
                f"Sample length {processed['input_ids'].shape[0]} > training "
                f"seq_len={self._seq_len}. Skip"
            )
            return None
        return processed


MM_DATASETS: dict[str, SingleDatasetConfig] = {
    "obelics": SingleDatasetConfig(
        source=HuggingFaceStreamingSource.Config(
            path="HuggingFaceM4/OBELICS",
            split="train",
        ),
        processor=MultiModalProcessor.Config(
            sample_processor=_process_obelics_sample,
        ),
        post_filters=(lambda sample: sample is not None,),
    ),
    "cc12m": SingleDatasetConfig(
        source=HuggingFaceStreamingSource.Config(
            path="pixparse/cc12m-wds",
            split="train",
        ),
        processor=MultiModalProcessor.Config(
            sample_processor=_process_cc12_wd_sample,
        ),
        post_filters=(lambda sample: sample is not None,),
    ),
    "cc12m-test": SingleDatasetConfig(
        source=HuggingFaceStreamingSource.Config(
            path="tests/assets/cc12m_test",
            split="train",
            load_dataset_kwargs={
                "data_files": {"train": "*.tar"},
            },
        ),
        processor=MultiModalProcessor.Config(
            sample_processor=_process_cc12_wd_sample,
        ),
        post_filters=(lambda sample: sample is not None,),
    ),
}


@dataclass(frozen=True, kw_only=True, slots=True)
class MMSamplePackingConfig:
    """Packs whole multimodal documents into fixed-length rows."""

    dataset: GrainDatasetConfig
    num_packing_bins: int = 8
    """Candidate rows kept open; more bins can reduce padding but retain more media."""

    def __post_init__(self) -> None:
        if self.num_packing_bins <= 0:
            raise ValueError("num_packing_bins must be positive")

    def build(
        self,
        *,
        context: DatasetBuildContext,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> grain.IterDataset[dict[str, Any]]:
        dataset = self.dataset.build(
            context=context,
            dataset_iteration_policy=dataset_iteration_policy,
        )
        dataset = dataset.filter(
            lambda sample: len(sample["input_ids"]) <= context.seq_len
        )
        dataset = dataset.map(_mm_sample_to_packing_input)
        if isinstance(dataset, grain.MapDataset):
            dataset = dataset.to_iter_dataset(read_options=context.read_options)
        # TODO(data-global-pack-plan): Consider packing before DP sharding so
        # ranks receive similar text and media work.
        dataset = grain.experimental.FirstFitPackIterDataset(
            dataset,
            length_struct={
                "input_ids": context.seq_len,
                "labels": context.seq_len,
                "positions": context.seq_len,
            },
            padding_struct={
                # pyrefly: ignore [missing-attribute]
                "input_ids": context.tokenizer.pad_id,
                "labels": IGNORE_INDEX,
                "positions": 0,
            },
            num_packing_bins=self.num_packing_bins,
            meta_features=(
                "labels",
                "positions",
                "pixel_values",
                "pixel_values_videos",
            ),
            seed=dataset_iteration_policy.seed,
            shuffle_bins=dataset_iteration_policy.shuffle,
        )
        return dataset.map(_packing_output_to_mm_sample)


def _mm_sample_to_packing_input(sample: dict[str, Any]) -> dict[str, Any]:
    """Convert Torch token fields to the arrays expected by Grain packing."""
    return {
        "input_ids": np.asarray(sample["input_ids"]),
        "labels": np.asarray(sample["labels"]),
        "positions": np.asarray(sample["positions"]),
        "pixel_values": sample.get("pixel_values", []),
        "pixel_values_videos": sample.get("pixel_values_videos", []),
    }


def _packing_output_to_mm_sample(
    packing_output: dict[str, Any],
) -> dict[str, Any]:
    """Restore Torch token fields and flatten per-document media lists."""
    return {
        "input_ids": torch.from_numpy(packing_output["input_ids"]),
        "labels": torch.from_numpy(packing_output["labels"]),
        "positions": torch.from_numpy(packing_output["positions"]),
        "pixel_values": [
            image
            for document_images in packing_output["pixel_values"]
            for image in document_images
        ],
        "pixel_values_videos": [
            video
            for document_videos in packing_output["pixel_values_videos"]
            for video in document_videos
        ],
    }
