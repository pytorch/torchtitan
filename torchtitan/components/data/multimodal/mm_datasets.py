# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multimodal dataset processing for image, video, and audio training.

Data flow::

    +------------------------------------------------------------+
    | Dataset source                                             |
    | Hugging Face streaming or indexed JSONL -> raw row dict    |
    +------------------------------------------------------------+
                              |
                              v
    +------------------------------------------------------------+
    | Sample processor                                           |
    |                                                            |
    | VisionProcessor -> configured callback                     |
    |   decode, resize, and normalize images / video frames with |
    |   the trainer's ImageProcessor / VideoProcessor            |
    |   (resize_and_normalize / resize_and_normalize_video)      |
    |   -> ordered pixel_values / pixel_values_videos lists      |
    |                                                            |
    | AudioProcessor -> configured callback                      |
    |   decode, resample, and truncate channels-last waveforms   |
    |   -> ordered waveforms list                                |
    |                                                            |
    | Insert media placeholders and tokenize text                |
    |   -> 1-D input_ids, labels, and positions                  |
    |   -> targets in LOSS_MASK_TOKEN_FIELDS use -100            |
    +------------------------------------------------------------+
                              |
                              v  optional MMSamplePackingConfig
    +------------------------------------------------------------+
    | Pack whole samples to the token-microbatch size while      |
    | preserving document and media-list order                   |
    +------------------------------------------------------------+
                              |
                              v
    +------------------------------------------------------------+
    | MultiModalCollator                                         |
    |                                                            |
    | Concatenate and pad text to one token microbatch           |
    |   -> input_ids, labels, positions, and padding_mask        |
    | Patchify images and videos (ImageProcessor.patchify)       |
    |   -> pixel_values / pixel_values_videos patches and        |
    |      grid_thw / grid_thw_videos                            |
    | Independently pad audio clips, then concatenate them       |
    |   -> waveforms, waveform_lengths, waveform_padded_lengths  |
    +------------------------------------------------------------+
                              |
                              v
                  TokenizedTrainingMicrobatch

TODO: reorganize modality specific functions and non-hf specific functions
into separate files and directories.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import grain.python as grain
import numpy as np
import torch

from torchtitan.components.data.dataset import (
    DatasetConfig as GrainDatasetConfig,
    SampleProcessor,
    SingleDatasetConfig,
)
from torchtitan.components.data.sources import HuggingFaceStreamingSource
from torchtitan.components.data.types import DatasetBuildContext, DatasetIterationPolicy
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import MultiModalTokenizer

from .image_processor import ImageProcessor
from .utils.audio import resample_waveform
from .utils.text import insert_vision_placeholders
from .video_processor import VideoProcessor


logger = logging.getLogger(__name__)


def _tokenize_mm_text(
    processed_text: str,
    tokenizer: MultiModalTokenizer,
) -> dict[str, torch.Tensor] | None:
    """Tokenize text and mask targets named by ``LOSS_MASK_TOKEN_FIELDS``."""
    tokens = tokenizer.encode(processed_text)
    if len(tokens) < 2:
        return None

    input_ids = torch.tensor(tokens[:-1])
    labels = torch.tensor(tokens[1:])
    special_token_ids = torch.tensor(
        [
            getattr(tokenizer, f"{field}_id")
            for field in tokenizer.LOSS_MASK_TOKEN_FIELDS
        ]
    )
    labels = torch.where(torch.isin(labels, special_token_ids), IGNORE_INDEX, labels)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "positions": torch.arange(len(input_ids)),
    }


def _process_audio_text_sample(
    text: str,
    waveform: torch.Tensor,
    source_sample_rate: int,
    tokenizer: MultiModalTokenizer,
    *,
    target_sample_rate: int,
    max_audio_samples: int | None,
    audio_token_count_fn: Callable[[int, int], int],
) -> dict[str, Any] | None:
    """Process one paired waveform and caption.

    The processor owns resampling and placeholder geometry. It returns one
    channels-last float32 waveform in a list aligned with the audio placeholders.
    """
    waveform = resample_waveform(
        waveform,
        source_sample_rate=source_sample_rate,
        target_sample_rate=target_sample_rate,
    )
    if max_audio_samples is not None:
        waveform = waveform[:max_audio_samples]
    waveform = waveform.to(dtype=torch.float32).contiguous()

    num_audio_tokens = audio_token_count_fn(waveform.shape[0], target_sample_rate)
    if num_audio_tokens <= 0:
        raise ValueError("audio_token_count_fn must return a positive integer")

    audio_start_token = cast(
        str,
        tokenizer.audio_start_token,  # pyrefly: ignore [missing-attribute]
    )
    audio_token = cast(
        str,
        tokenizer.audio_token,  # pyrefly: ignore [missing-attribute]
    )
    audio_end_token = cast(
        str,
        tokenizer.audio_end_token,  # pyrefly: ignore [missing-attribute]
    )
    eos_token = cast(str, tokenizer.eos_token)
    processed_text = "".join(
        [
            audio_start_token,
            *([audio_token] * num_audio_tokens),
            audio_end_token,
            text,
            eos_token,
        ]
    )
    text_fields = _tokenize_mm_text(processed_text, tokenizer)
    if text_fields is None:
        return None
    return {
        **text_fields,
        "waveforms": [waveform],
    }


def _process_video_text_sample(
    text: str,
    frames: torch.Tensor,
    tokenizer: MultiModalTokenizer,
    video_processor: VideoProcessor | None,
) -> dict[str, Any] | None:
    """Process one paired frame sequence and caption."""
    if video_processor is None:
        raise ValueError("Video samples require Trainer.Config.video_processor.")
    processed_video, grid_thw_video = video_processor.resize_and_normalize_video(frames)
    num_video_tokens = video_processor.num_placeholder_tokens(grid_thw_video)
    if num_video_tokens <= 0:
        raise ValueError("video preprocessing must produce a positive token count")

    processed_text = insert_vision_placeholders(
        [None, text],
        [num_video_tokens],
        vision_start_token=cast(
            str,
            tokenizer.vision_start_token,  # pyrefly: ignore [missing-attribute]
        ),
        vision_token=cast(
            str,
            tokenizer.video_token,  # pyrefly: ignore [missing-attribute]
        ),
        vision_end_token=cast(
            str,
            tokenizer.vision_end_token,  # pyrefly: ignore [missing-attribute]
        ),
        eos_token=cast(str, tokenizer.eos_token),
    )
    text_fields = _tokenize_mm_text(processed_text, tokenizer)
    if text_fields is None:
        return None
    return {**text_fields, "pixel_values_videos": [processed_video]}


def _process_mm_sample(
    texts: list[str | None],
    images: list[bytes | None],
    tokenizer: MultiModalTokenizer,
    image_processor: ImageProcessor | None,
) -> dict[str, Any] | None:
    """Common processing logic for multimodal samples.

    Args:
        texts: List of strings with None indicating image positions
        images: List of image bytes with None for text positions
        tokenizer: Tokenizer for text processing
        image_processor: The model's image preprocessing; required when the
            sample has images

    Returns:
        Dict with:
            - input_ids: Tensor of token IDs
            - labels: Tensor of label IDs
            - pixel_values: List of each image's resized, normalized float32
              (1, H, W, C) pixels; the collator patchifies them

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
        if img is None:
            continue
        if image_processor is None:
            raise ValueError("Image samples require Trainer.Config.image_processor.")
        try:
            processed_image, grid_thw = image_processor.resize_and_normalize(img)
        except Exception as e:
            # Pretraining data skips a sample with an unreadable image.
            logger.warning(f"Dropping sample, error processing image: {e}")
            return None
        processed_images.append(processed_image)
        num_image_tokens.append(image_processor.num_placeholder_tokens(grid_thw))
        # Keep the accepted image at this aligned position as a placeholder.
        texts[idx] = None

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

    text_fields = _tokenize_mm_text(processed_text, tokenizer)
    if text_fields is None:
        return None
    return {**text_fields, "pixel_values": processed_images}


def _process_obelics_sample(
    sample: dict[str, Any],
    tokenizer: MultiModalTokenizer,
    image_processor: ImageProcessor | None,
    **kwargs,
) -> dict[str, Any] | None:
    """Process a sample from the OBELICS dataset (interleaved text and images)."""
    return _process_mm_sample(
        texts=sample.get("texts", []),
        images=sample.get("images", []),
        tokenizer=tokenizer,
        image_processor=image_processor,
    )


def _process_cc12_wd_sample(
    sample: dict[str, Any],
    tokenizer: MultiModalTokenizer,
    image_processor: ImageProcessor | None,
    **kwargs,
) -> dict[str, Any] | None:
    """Process a sample from the CC12-WD dataset (text-image pairs)."""
    text = sample.get("txt", "")
    image = sample.get("jpg", None)

    if image is None:
        texts: list[str | None] = [text]
        images: list[bytes | None] = [None]
    else:
        texts = [None, text]
        images = [image, None]

    return _process_mm_sample(
        texts=texts,
        images=images,
        tokenizer=tokenizer,
        image_processor=image_processor,
    )


def _filter_overlong_sample(
    processed: dict[str, Any] | None,
    *,
    max_context_length: int,
) -> dict[str, Any] | None:
    if processed is not None and processed["input_ids"].shape[0] > max_context_length:
        logger.warning(
            f"Sample length {processed['input_ids'].shape[0]} > training "
            f"max_context_length={max_context_length}. Skip"
        )
        return None
    return processed


class VisionProcessor(SampleProcessor):
    """Processes image-text and video-text rows for Grain.

    ``sample_processor`` gets the row, the tokenizer, and the trainer's
    ``image_processor`` and ``video_processor`` (None when not configured).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        sample_processor: Callable

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        if context.image_processor is None and context.video_processor is None:
            raise ValueError(
                "VisionProcessor requires Trainer.Config.image_processor or "
                "Trainer.Config.video_processor."
            )
        self._config = config
        self._tokenizer = context.tokenizer
        self._image_processor = context.image_processor
        self._video_processor = context.video_processor
        self._max_context_length = context.max_context_length

    def __call__(
        self,
        sample: dict[str, Any],
        rng: np.random.Generator,
    ) -> dict[str, Any] | None:
        del rng
        processed = self._config.sample_processor(
            sample=sample,
            tokenizer=self._tokenizer,
            image_processor=self._image_processor,
            video_processor=self._video_processor,
        )
        return _filter_overlong_sample(
            processed,
            max_context_length=self._max_context_length,
        )


class AudioProcessor(SampleProcessor):
    """Processes audio-text rows for Grain."""

    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        sample_processor: Callable
        expected_num_channels: int
        target_sample_rate: int
        audio_token_count_fn: Callable[[int, int], int]
        max_audio_samples: int | None = None

        def __post_init__(self) -> None:
            if self.expected_num_channels <= 0:
                raise ValueError("expected_num_channels must be positive")
            if self.target_sample_rate <= 0:
                raise ValueError("target_sample_rate must be positive")
            if self.max_audio_samples is not None and self.max_audio_samples <= 0:
                raise ValueError("max_audio_samples must be positive when configured")

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        self._config = config
        self._tokenizer = context.tokenizer
        self._max_context_length = context.max_context_length

    def __call__(
        self,
        sample: dict[str, Any],
        rng: np.random.Generator,
    ) -> dict[str, Any] | None:
        del rng
        processed = self._config.sample_processor(
            sample=sample,
            tokenizer=self._tokenizer,
            expected_num_channels=self._config.expected_num_channels,
            target_sample_rate=self._config.target_sample_rate,
            max_audio_samples=self._config.max_audio_samples,
            audio_token_count_fn=self._config.audio_token_count_fn,
        )
        return _filter_overlong_sample(
            processed,
            max_context_length=self._max_context_length,
        )


MM_DATASETS: dict[str, SingleDatasetConfig] = {
    "obelics": SingleDatasetConfig(
        source=HuggingFaceStreamingSource.Config(
            path="HuggingFaceM4/OBELICS",
            split="train",
        ),
        processor=VisionProcessor.Config(
            sample_processor=_process_obelics_sample,
        ),
        post_filters=(lambda sample: sample is not None,),
    ),
    "cc12m": SingleDatasetConfig(
        source=HuggingFaceStreamingSource.Config(
            path="pixparse/cc12m-wds",
            split="train",
        ),
        processor=VisionProcessor.Config(
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
        processor=VisionProcessor.Config(
            sample_processor=_process_cc12_wd_sample,
        ),
        post_filters=(lambda sample: sample is not None,),
    ),
}


# Per-sample media lists, in document order; packing concatenates them.
_MEDIA_FIELDS = ("pixel_values", "pixel_values_videos", "waveforms")


@dataclass(frozen=True, kw_only=True, slots=True)
class MMSamplePackingConfig:
    """Pack whole documents while preserving document-then-media order.

    Media lists are flattened by one document level in each packed row.
    """

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
            lambda sample: len(sample["input_ids"]) <= context.max_context_length
        )
        dataset = dataset.map(_mm_sample_to_packing_input)
        if isinstance(dataset, grain.MapDataset):
            dataset = dataset.to_iter_dataset(read_options=context.read_options)
        # TODO(data-global-pack-plan): Consider packing before DP sharding so
        # ranks receive similar text and media work.
        dataset = grain.experimental.FirstFitPackIterDataset(
            dataset,
            length_struct={
                "input_ids": context.num_tokens_per_microbatch,
                "labels": context.num_tokens_per_microbatch,
                "positions": context.num_tokens_per_microbatch,
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
                *_MEDIA_FIELDS,
            ),
            seed=dataset_iteration_policy.seed,
            shuffle_bins=dataset_iteration_policy.shuffle,
        )
        return dataset.map(
            partial(
                _packing_output_to_mm_sample,
                max_context_length=context.max_context_length,
            )
        )


def _mm_sample_to_packing_input(sample: dict[str, Any]) -> dict[str, Any]:
    """Convert Torch token fields to the arrays expected by Grain packing."""
    return {
        "input_ids": np.asarray(sample["input_ids"]),
        "labels": np.asarray(sample["labels"]),
        "positions": np.asarray(sample["positions"]),
        **{key: sample.get(key, []) for key in _MEDIA_FIELDS},
    }


def _packing_output_to_mm_sample(
    packing_output: dict[str, Any],
    *,
    max_context_length: int,
) -> dict[str, Any]:
    """Restore Torch token fields and flatten per-document media lists."""
    padding_mask = np.asarray(packing_output["input_ids_segment_ids"]) == 0
    positions = np.asarray(packing_output["positions"]).copy()
    if np.any(padding_mask):
        first_padding_token = int(np.flatnonzero(padding_mask)[0])
        positions[first_padding_token:] = (
            np.arange(len(positions) - first_padding_token) % max_context_length
        )
    return {
        "input_ids": torch.from_numpy(packing_output["input_ids"]),
        "labels": torch.from_numpy(packing_output["labels"]),
        "positions": torch.from_numpy(positions),
        "padding_mask": torch.from_numpy(padding_mask),
        **{
            key: [
                item
                for document_items in packing_output[key]
                for item in document_items
            ]
            for key in _MEDIA_FIELDS
        },
    }
