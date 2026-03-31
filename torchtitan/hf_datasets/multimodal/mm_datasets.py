# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multimodal dataset and dataloader for VLM training.

Workflow overview::

    HuggingFace Dataset (streaming)
            │
            ▼
    ┌───────────────────────────────────────────────────────┐
    │  Sample Processor  (per-sample, in Dataset.__iter__)  │
    │                                                       │
    │  1. Parse raw sample (dataset-specific format)        │
    │     e.g. OBELICS interleaved text/images,             │
    │          CC12M text-image pairs,                      │
    │          Nemotron video QA messages                   │
    │                                                       │
    │  2. Process vision: decode image/video bytes,         │
    │     resize to multiples of (patch_size * merge_size), │
    │     normalize with image_mean/std                     │
    │     → pixel_values: list[Tensor(T,H,W,C)]            │
    │                                                       │
    │  3. Process text: insert vision placeholder tokens    │
    │     <|vision_start|><|image_pad|>...<|vision_end|>    │
    │     into text, then tokenize                          │
    │     → input_ids: Tensor(seq_len,)                     │
    │     → labels: same as input_ids, with vision tokens   │
    │       masked to ignore_id (-100)                      │
    └───────────────────────────────────────────────────────┘
            │
            ▼  (optional, if packing_buffer_size > 0)
    ┌───────────────────────────────────────────────────────┐
    │  Sample Packer                                        │
    │  Bin-pack short samples into seq_len-length sequences │
    │  to reduce padding waste                              │
    └───────────────────────────────────────────────────────┘
            │
            ▼  DataLoader batches samples (batch_size)
    ┌───────────────────────────────────────────────────────┐
    │  Collator  (MultiModalCollatorNLD)                    │
    │                                                       │
    │  1. collate_images: for each image Tensor(T,H,W,C),  │
    │     reshape into patches (num_patches, patch_dim),    │
    │     pad all images to same num_patches                │
    │     → pixel_values: (N, max_patches, patch_dim)       │
    │     → grid_thw: (N, 3) per-image [T, H', W'] dims    │
    │     (same for videos)                                 │
    │                                                       │
    │  2. collate_text: pad input_ids/labels across batch   │
    │     to seq_len, pad batch to target batch_size        │
    │     → input_ids: (batch_size, seq_len)                │
    │     → labels: (batch_size, seq_len)                   │
    └───────────────────────────────────────────────────────┘
            │
            ▼
    Model receives: {input_ids, pixel_values, grid_thw,
                     pixel_values_videos, grid_thw_videos,
                     special_tokens}, labels
"""

import inspect
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer, HuggingFaceTokenizer

from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets.multimodal import MMSpecialTokens
from torchtitan.tools.logging import logger
from .mm_collator_nld import MultiModalCollatorNLD
from .utils.image import calculate_vision_tokens, process_image, smart_resize
from .utils.packing import MMSamplePacker
from .utils.text import process_text_with_images, process_text_with_videos
from .utils.video import load_video, process_video


def _process_mm_sample(
    texts: list[str] | str,
    images: list[bytes] | bytes,
    tokenizer: BaseTokenizer,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
    min_pixels: int,
    max_pixels: int,
    image_mean: tuple[float, ...],
    image_std: tuple[float, ...],
    special_tokens: MMSpecialTokens,
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
    try:
        texts = [texts] if isinstance(texts, str) else texts
        images = [images] if isinstance(images, bytes) else images

        if not texts or len(texts) != len(images):
            return None

        processed_images = []
        image_dimensions = []

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
                )
                if processed_img is not None:
                    # Each (patch_size x temporal_patch_size) x (patch_size x temporal_patch_size)
                    # square block of pixels is mapped to one image token
                    num_tokens, tokens_per_row, num_rows = calculate_vision_tokens(
                        num_frames=1,
                        height=processed_img.shape[1],
                        width=processed_img.shape[2],
                        patch_size=patch_size,
                        spatial_merge_size=spatial_merge_size,
                        temporal_patch_size=1,
                    )
                    processed_images.append(processed_img)
                    image_dimensions.append((num_tokens, tokens_per_row, num_rows))
                    # pyrefly: ignore [unsupported-operation]
                    texts[idx] = None
                else:
                    texts[idx] = ""

        if len(processed_images) != len([_ for _ in images if _ is not None]):
            logger.warning("Cannot process all images for sample. Dropping")
            return None

        # Replace an image placeholder, i.e., None, by a sequence of image token placeholders
        processed_text = process_text_with_images(
            texts, image_dimensions, tokenizer, special_tokens, add_eos=True
        )

        tokens = tokenizer.encode(processed_text)
        input_ids = torch.tensor(tokens)
        labels = torch.tensor(tokens)

        special_token_ids = torch.tensor(
            [
                special_tokens.vision_start_id,
                special_tokens.vision_end_id,
                special_tokens.img_id,
            ]
        )
        labels = torch.where(
            torch.isin(labels, special_token_ids), special_tokens.ignore_id, labels
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "positions": torch.arange(len(input_ids)),
            "pixel_values": processed_images,
        }

    except Exception as e:
        logger.warning(f"Error processing sample: {e}")
        return None


def _process_obelics_sample(
    sample: dict[str, Any],
    tokenizer: HuggingFaceTokenizer,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
    min_pixels: int,
    max_pixels: int,
    image_mean: tuple[float, ...],
    image_std: tuple[float, ...],
    special_tokens: MMSpecialTokens,
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
        special_tokens=special_tokens,
    )


def _process_cc12_wd_sample(
    sample: dict[str, Any],
    tokenizer: BaseTokenizer,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
    min_pixels: int,
    max_pixels: int,
    image_mean: tuple[float, ...],
    image_std: tuple[float, ...],
    special_tokens: MMSpecialTokens,
    **kwargs,
) -> dict[str, Any] | None:
    """Process a sample from the CC12-WD dataset (text-image pairs)."""
    text = sample.get("txt", "")
    image = sample.get("jpg", None)

    texts = [None, text]
    images = [image, None]

    return _process_mm_sample(
        # pyrefly: ignore [bad-argument-type]
        texts=texts,
        # pyrefly: ignore [bad-argument-type]
        images=images,
        tokenizer=tokenizer,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=spatial_merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        image_mean=image_mean,
        image_std=image_std,
        special_tokens=special_tokens,
    )


def _process_nemotron_video_sample(
    sample: dict[str, Any],
    tokenizer: BaseTokenizer,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
    min_pixels: int,
    max_pixels: int,
    image_mean: tuple[float, ...],
    image_std: tuple[float, ...],
    special_tokens: MMSpecialTokens,
    video_dir: str = "",
    video_fps: float = 2.0,
    video_min_frames: int = 4,
    video_max_frames: int = 768,
    **kwargs,
) -> dict[str, Any] | None:
    """Process a sample from the Nemotron video dataset.

    Expected format::

        {
            "messages": [
                {"role": "user", "content": [
                    {"type": "video", "video": "NExTVideo/1122/xxx.mp4", ...},
                    {"type": "text", "text": "question text"},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "answer"},
                ]},
            ]
        }
    """
    try:
        messages = sample.get("messages", [])
        if len(messages) < 2:
            return None

        # Extract video filename and user text from the first turn
        user_turn = messages[0]
        user_content = user_turn.get("content", [])
        video_filename = None
        user_text = ""
        for item in user_content:
            if item.get("type") == "video" and item.get("video"):
                video_filename = item["video"]
            if item.get("type") == "text" and item.get("text"):
                user_text = item["text"]

        if video_filename is None:
            return None

        # Pre-filter: estimate token count from metadata to avoid decoding
        # videos that will exceed seq_len
        video_item = next(
            (item for item in user_content if item.get("type") == "video"), None
        )
        metadata = video_item.get("metadata", {}) if video_item else {}
        if metadata and "video_duration" in metadata:
            duration = metadata["video_duration"]
            src_frames = metadata.get("video_num_frames") or int(
                duration * (metadata.get("video_fps") or 24.0)
            )
            est_frames = int(duration * video_fps)
            est_frames = max(video_min_frames, min(est_frames, video_max_frames))
            est_frames = min(est_frames, src_frames)

            factor = patch_size * spatial_merge_size
            est_h, est_w = smart_resize(
                metadata.get("video_height", 360),
                metadata.get("video_width", 640),
                factor=factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                num_frames=est_frames,
                temporal_factor=temporal_patch_size,
            )
            est_tokens, _, _ = calculate_vision_tokens(
                num_frames=est_frames,
                height=est_h,
                width=est_w,
                patch_size=patch_size,
                spatial_merge_size=spatial_merge_size,
                temporal_patch_size=temporal_patch_size,
            )

            if est_tokens > kwargs.get("seq_len", float("inf")):
                logger.debug(
                    f"Pre-filter skip {video_filename}: ~{est_tokens} video tokens > seq_len"
                )
                return None

        # Extract assistant response
        assistant_turn = messages[1]
        assistant_content = assistant_turn.get("content", [])
        assistant_text = ""
        for item in assistant_content:
            if item.get("type") == "text" and item.get("text"):
                assistant_text = item["text"]

        # Load and process video
        video_path = os.path.join(video_dir, video_filename)
        raw_video = load_video(
            video_path,
            fps=video_fps,
            min_frames=video_min_frames,
            max_frames=video_max_frames,
        )
        if raw_video is None:
            return None

        processed_video = process_video(
            raw_video,
            patch_size=patch_size,
            merge_size=spatial_merge_size,
            temporal_patch_size=temporal_patch_size,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            image_mean=image_mean,
            image_std=image_std,
        )
        if processed_video is None:
            return None

        # Calculate video token count
        T, H, W, _C = processed_video.shape
        num_tokens, tokens_per_row, num_rows = calculate_vision_tokens(
            num_frames=T,
            height=H,
            width=W,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            temporal_patch_size=temporal_patch_size,
        )

        # Build text: [video_placeholder, user_text + assistant_text]
        full_text = user_text + assistant_text
        texts = [None, full_text]
        video_dimensions = [(num_tokens, tokens_per_row, num_rows)]

        processed_text = process_text_with_videos(
            # pyrefly: ignore [bad-argument-type]
            texts,
            video_dimensions,
            tokenizer,
            special_tokens,
            add_eos=True,
        )

        tokens = tokenizer.encode(processed_text)
        input_ids = torch.tensor(tokens)
        labels = torch.tensor(tokens)

        # Mask vision token IDs in labels
        special_token_ids = torch.tensor(
            [
                special_tokens.vision_start_id,
                special_tokens.vision_end_id,
                special_tokens.vid_id,
            ]
        )
        labels = torch.where(
            torch.isin(labels, special_token_ids), special_tokens.ignore_id, labels
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "positions": torch.arange(len(input_ids)),
            "pixel_values_videos": [processed_video],
        }

    except Exception as e:
        logger.warning(f"Error processing Nemotron video sample: {e}")
        return None


MM_DATASETS = {
    "obelics": DatasetConfig(
        path="HuggingFaceM4/OBELICS",
        loader=lambda path: load_dataset(path, split="train", streaming=True),
        sample_processor=_process_obelics_sample,
    ),
    "cc12m": DatasetConfig(
        path="pixparse/cc12m-wds",
        loader=lambda path: load_dataset(path, split="train", streaming=True),
        sample_processor=_process_cc12_wd_sample,
    ),
    "cc12m-test": DatasetConfig(
        path="tests/assets/cc12m_test",
        loader=lambda path: load_dataset(
            path, split="train", data_files={"train": "*.tar"}, streaming=True
        ),
        sample_processor=_process_cc12_wd_sample,
    ),
    "nemotron-video": DatasetConfig(
        path="nvidia/Nemotron-VLM-Dataset-V2",
        loader=lambda path, subset="nextqa": load_dataset(
            path, subset or "nextqa", split="train", streaming=True
        ),
        sample_processor=_process_nemotron_video_sample,
    ),
}


def _validate_mm_dataset(
    dataset_name: str, dataset_path: str | None = None
) -> tuple[str, Callable, Callable]:
    """Validate dataset name and path, returning (path, loader, sample_processor)."""
    if dataset_name not in MM_DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(MM_DATASETS.keys())}"
        )

    config = MM_DATASETS[dataset_name]
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.sample_processor


class HuggingFaceMultiModalDataset(IterableDataset, Stateful):
    """HuggingFace MultiModal Dataset for Qwen3-VL with support for sample packing."""

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        batch_size: int,
        seq_len: int,
        patch_size: int,
        temporal_patch_size: int,
        spatial_merge_size: int,
        max_images_per_batch: int,
        min_pixels: int,
        max_pixels: int,
        image_mean: tuple[float, ...],
        image_std: tuple[float, ...],
        packing_buffer_size: int,
        special_tokens: MMSpecialTokens,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        video_dir: str = "",
        video_fps: float = 2.0,
        video_min_frames: int = 4,
        video_max_frames: int = 768,
        dataset_subset: str = "",
    ) -> None:
        dataset_name = dataset_name.lower()

        path, dataset_loader, self.sample_processor = _validate_mm_dataset(
            dataset_name, dataset_path
        )

        # Pass subset to loaders that accept it (e.g. nemotron-video)
        sig = inspect.signature(dataset_loader)
        if "subset" in sig.parameters and dataset_subset:
            ds = dataset_loader(path, subset=dataset_subset)
        else:
            ds = dataset_loader(path)
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)

        self._tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.max_images_per_batch = max_images_per_batch
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_mean = image_mean
        self.image_std = image_std
        self.special_tokens = special_tokens
        self.video_dir = video_dir
        self.video_fps = video_fps
        self.video_min_frames = video_min_frames
        self.video_max_frames = video_max_frames
        self.enable_packing = packing_buffer_size > 0
        if self.enable_packing:
            self.packer = MMSamplePacker(
                max_seq_length=seq_len,
                buffer_size=packing_buffer_size,
                batch_size=batch_size,
            )
        self.infinite = infinite
        self._sample_idx = 0
        self._hf_state_restored = False

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                try:
                    self._sample_idx += 1

                    processed = self.sample_processor(
                        sample=sample,
                        tokenizer=self._tokenizer,
                        patch_size=self.patch_size,
                        temporal_patch_size=self.temporal_patch_size,
                        spatial_merge_size=self.spatial_merge_size,
                        min_pixels=self.min_pixels,
                        max_pixels=self.max_pixels,
                        image_mean=self.image_mean,
                        image_std=self.image_std,
                        special_tokens=self.special_tokens,
                        video_dir=self.video_dir,
                        video_fps=self.video_fps,
                        video_min_frames=self.video_min_frames,
                        video_max_frames=self.video_max_frames,
                        seq_len=self.seq_len,
                    )
                    if processed is None:
                        continue

                    if processed["input_ids"].shape[0] > self.seq_len:
                        logger.warning(
                            f"Sample length {processed['input_ids'].shape[0]} > training {self.seq_len=}. Skip"
                        )
                        continue

                    if self.enable_packing:
                        self.packer.add_sample(processed)

                        if self.packer.has_batch_ready():
                            batch = self.packer.get_next_batch()
                            if batch:
                                yield from batch
                    else:
                        yield processed

                except Exception as e:
                    logger.warning(f"Error in iteration: {e}")
                    continue

            # Flush leftovers in packer when raw samples are exhausted
            if self.enable_packing:
                self.packer.flush()
                while self.packer.has_batch_ready():
                    # pyrefly: ignore [invalid-yield]
                    yield from self.packer.get_next_batch()
                # Drain any remainder that doesn't fill a full batch
                while self.packer.packed_samples:
                    yield self.packer.packed_samples.popleft()

            if not self.infinite:
                break
            else:
                self._sample_idx = 0

    def _get_data_iter(self):
        try:
            # If HF dataset state was restored, iterator already starts
            # at the right position — no need to skip.
            if self._hf_state_restored:
                self._hf_state_restored = False
                return iter(self._data)

            # Map-style dataset: use random access to skip directly
            if isinstance(self._data, Dataset):
                if self._sample_idx >= len(self._data):
                    return iter([])
                return iter(self._data.select(range(self._sample_idx, len(self._data))))

            # Streaming dataset without restored state: brute-force skip
            it = iter(self._data)
            if self._sample_idx > 0:
                logger.info(
                    f"Skipping {self._sample_idx} samples to resume from checkpoint"
                )
                for _ in range(self._sample_idx):
                    next(it)

            return it
        except Exception as e:
            logger.error(f"Error in _get_data_iter: {e}")
            return iter([])

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]

        # Restore HF dataset state if available, enabling fast resume
        if "hf_dataset_state" in state_dict and hasattr(self._data, "load_state_dict"):
            self._data.load_state_dict(state_dict["hf_dataset_state"])
            self._hf_state_restored = True

        if (
            self.enable_packing
            and hasattr(self, "packer")
            and "packer_state" in state_dict
        ):
            packer_state = state_dict["packer_state"]
            self.packer.sample_buffer.clear()
            self.packer.packed_samples.clear()
            self.packer.sample_buffer.extend(packer_state["sample_buffer"])
            self.packer.packed_samples.extend(packer_state["packed_samples"])

    def state_dict(self):
        state = {"sample_idx": self._sample_idx}

        # Save HF dataset state for fast resume if supported
        if hasattr(self._data, "state_dict"):
            state["hf_dataset_state"] = self._data.state_dict()

        if self.enable_packing and hasattr(self, "packer"):
            # pyrefly: ignore [bad-typed-dict-key]
            state["packer_state"] = {
                "sample_buffer": list(self.packer.sample_buffer),
                "packed_samples": list(self.packer.packed_samples),
            }

        return state


class MMDataLoader(ParallelAwareDataloader):
    """Configurable multimodal dataloader for Qwen3-VL."""

    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        dataset: str = "cc12m-test"
        """Dataset to use"""

        dataset_subset: str = ""
        """Dataset subset/config name (e.g. 'nextqa' for nemotron-video)."""

        infinite: bool = True
        """Whether to loop the dataset infinitely"""

        # Batching configs
        max_images_per_batch: int = 10
        """Vision encoder batch size (N)"""

        max_patches_per_image: int = 256
        """Vision encoder sequence length (L)"""

        packing_buffer_size: int = 0
        """Set to a value >0 to enable sample packing."""

        # Preprocessing configs
        patch_size: int
        """Patch size of the vision encoder."""

        temporal_patch_size: int
        """Temporal patch size for video processing."""

        spatial_merge_size: int
        """Spatially merge visual tokens after encoder. e.g. 2 means 2x2=4 patches merged."""

        min_pixels: int = 65536
        """Minimum number of pixels for image resizing."""

        max_pixels: int = 16777216
        """Maximum number of pixels for image resizing."""

        image_mean: tuple[float, ...] = (0.5, 0.5, 0.5)
        """Per-channel mean for image normalization."""

        image_std: tuple[float, ...] = (0.5, 0.5, 0.5)
        """Per-channel std for image normalization."""

        video_dir: str = ""
        """Base directory for video files (for datasets with video filename references)."""

        video_fps: float = 2.0
        """Target frames per second for video sampling."""

        video_min_frames: int = 4
        """Minimum number of frames to sample from a video."""

        video_max_frames: int = 768
        """Maximum number of frames to sample from a video."""

    # Subclasses must set this to their model's special tokens class.
    special_tokens_cls: type[MMSpecialTokens]

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: HuggingFaceTokenizer,
        seq_len: int,
        local_batch_size: int,
        **kwargs,
    ):
        special_tokens = self.special_tokens_cls.from_tokenizer(tokenizer)

        dataset = HuggingFaceMultiModalDataset(
            dataset_name=config.dataset,
            dataset_path=config.dataset_path,
            tokenizer=tokenizer,
            batch_size=local_batch_size,
            seq_len=seq_len,
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            spatial_merge_size=config.spatial_merge_size,
            max_images_per_batch=config.max_images_per_batch,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            image_mean=config.image_mean,
            image_std=config.image_std,
            packing_buffer_size=config.packing_buffer_size,
            special_tokens=special_tokens,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=config.infinite,
            video_dir=config.video_dir,
            video_fps=config.video_fps,
            video_min_frames=config.video_min_frames,
            video_max_frames=config.video_max_frames,
            dataset_subset=config.dataset_subset,
        )

        collate_fn = MultiModalCollatorNLD(
            batch_size=local_batch_size,
            seq_len=seq_len,
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            spatial_merge_size=config.spatial_merge_size,
            max_images_per_batch=config.max_images_per_batch,
            max_patches_per_image=config.max_patches_per_image,
            special_tokens=special_tokens,
        )

        dataloader_kwargs = {
            "num_workers": config.num_workers,
            "persistent_workers": config.persistent_workers,
            "pin_memory": config.pin_memory,
            "prefetch_factor": config.prefetch_factor,
            "batch_size": local_batch_size,
            "collate_fn": collate_fn,
        }

        super().__init__(
            dataset,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            **dataloader_kwargs,
        )
