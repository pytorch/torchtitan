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
            ▼  Dataset groups whole samples by token budget
    ┌───────────────────────────────────────────────────────┐
    │  Collator  (MultiModalCollator)                    │
    │                                                       │
    │  1. collate_images: for each image Tensor(T,H,W,C),  │
    │     reshape into patches (num_patches, patch_dim),    │
    |     concatenate all valid patches                     |
    |     -> pixel_values: (total_patches, patch_dim)       |
    |     -> grid_thw: (N, 3) per-image [T, H', W'] dims   |
    │     (same for videos)                                 │
    │                                                       │
    │  2. collate_text: concatenate whole samples and pad   │
    │     only the unused token-batch tail                  │
    │     -> input_ids: (num_tokens_per_batch,)             │
    │     -> labels: (num_tokens_per_batch,)                │
    └───────────────────────────────────────────────────────┘
            │
            ▼
    Model receives: {input_ids, pixel_values, grid_thw,
                     pixel_values_videos, grid_thw_videos,
                     special_tokens: dict[str, int]}, labels
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import torch
import tyro
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import MultiModalTokenizer

from torchtitan.hf_datasets import DatasetConfig
from torchtitan.tools.logging import logger
from .mm_collator import MultiModalCollator
from .utils.image import calculate_vision_tokens, process_image, resize_to_pixel_budget
from .utils.packing import MMSamplePacker
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
                    temporal_patch_size=1,
                )
                processed_images.append(processed_img)
                num_image_tokens.append(num_tokens)
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
    input_ids = torch.tensor(tokens)
    labels = torch.tensor(tokens)

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
    """HuggingFace multimodal dataset with support for sample packing."""

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: MultiModalTokenizer,
        max_context_length: int,
        num_tokens_per_batch: int,
        patch_size: int,
        temporal_patch_size: int,
        spatial_merge_size: int,
        min_pixels: int,
        max_pixels: int,
        image_mean: tuple[float, ...],
        image_std: tuple[float, ...],
        packing_buffer_size: int,
        resize_fn: Callable[..., tuple[int, int, int, int]],
        max_patches: int,
        max_patches_per_side: int,
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

        # Pass subset to loaders that accept it
        sig = inspect.signature(dataset_loader)
        if "subset" in sig.parameters and dataset_subset:
            ds = dataset_loader(path, subset=dataset_subset)
        else:
            ds = dataset_loader(path)
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)

        self._tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.num_tokens_per_batch = num_tokens_per_batch
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_mean = image_mean
        self.image_std = image_std
        self.resize_fn = resize_fn
        self.max_patches = max_patches
        self.max_patches_per_side = max_patches_per_side
        self.video_dir = video_dir
        self.video_fps = video_fps
        self.video_min_frames = video_min_frames
        self.video_max_frames = video_max_frames
        self.enable_packing = packing_buffer_size > 0
        if self.enable_packing:
            self.packer = MMSamplePacker(
                max_seq_length=max_context_length,
                buffer_size=packing_buffer_size,
            )
        self.infinite = infinite
        self._sample_idx = 0
        self._hf_state_restored = False
        self._batch_samples: list[dict[str, Any]] = []
        self._num_batch_tokens = 0

    def _append_to_batch(self, sample: dict[str, Any]):
        """Add one whole sample and yield completed token-budget batches."""
        num_sample_tokens = sample["input_ids"].shape[0] - 1
        if num_sample_tokens <= 0:
            return
        if num_sample_tokens > self.max_context_length:
            logger.warning(
                "Sample has %d token slots, exceeding max_context_length=%d. Skipping.",
                num_sample_tokens,
                self.max_context_length,
            )
            return

        if (
            self._batch_samples
            and self._num_batch_tokens + num_sample_tokens > self.num_tokens_per_batch
        ):
            batch = self._batch_samples
            # Record the already-consumed sample as the next batch state before
            # yielding so a checkpoint taken at the yield boundary cannot lose it.
            self._batch_samples = [sample]
            self._num_batch_tokens = num_sample_tokens
            yield batch
            if self._num_batch_tokens == self.num_tokens_per_batch:
                batch = self._batch_samples
                self._batch_samples = []
                self._num_batch_tokens = 0
                yield batch
            return

        self._batch_samples.append(sample)
        self._num_batch_tokens += num_sample_tokens
        if self._num_batch_tokens == self.num_tokens_per_batch:
            batch = self._batch_samples
            self._batch_samples = []
            self._num_batch_tokens = 0
            yield batch

    def _drain_packed_samples(self):
        while self.packer.packed_samples:
            yield from self._append_to_batch(self.packer.packed_samples.popleft())

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
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
                    resize_fn=self.resize_fn,
                    max_patches=self.max_patches,
                    max_patches_per_side=self.max_patches_per_side,
                    video_dir=self.video_dir,
                    video_fps=self.video_fps,
                    video_min_frames=self.video_min_frames,
                    video_max_frames=self.video_max_frames,
                    seq_len=self.max_context_length,
                )
                if processed is None:
                    continue

                if processed["input_ids"].shape[0] - 1 > self.max_context_length:
                    logger.warning(
                        "Sample has %d token slots, exceeding max_context_length=%d. "
                        "Skipping.",
                        processed["input_ids"].shape[0] - 1,
                        self.max_context_length,
                    )
                    continue

                if self.enable_packing:
                    self.packer.add_sample(processed)
                    yield from self._drain_packed_samples()
                else:
                    yield from self._append_to_batch(processed)

            # Flush leftovers in packer when raw samples are exhausted
            if self.enable_packing:
                self.packer.flush()
                yield from self._drain_packed_samples()

            if not self.infinite:
                if self._batch_samples:
                    batch = self._batch_samples
                    self._batch_samples = []
                    self._num_batch_tokens = 0
                    yield batch
                break
            else:
                self._sample_idx = 0
                if hasattr(self._data, "set_epoch"):
                    self._data.set_epoch(self._data.epoch + 1)

    def _get_data_iter(self):
        # TODO: add epoch counter and per-epoch reshuffling (see text_datasets.py)

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

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        self._batch_samples = state_dict.get("batch_samples", [])
        self._num_batch_tokens = state_dict.get(
            "num_batch_tokens",
            sum(sample["input_ids"].shape[0] - 1 for sample in self._batch_samples),
        )

        # Restore HF dataset state if available, enabling fast resume
        if "hf_dataset_state" in state_dict and hasattr(self._data, "load_state_dict"):
            data_state = state_dict["hf_dataset_state"]
            if hasattr(self._data, "set_epoch"):
                self._data.set_epoch(data_state.get("epoch", 0))
            self._data.load_state_dict(data_state)
            self._hf_state_restored = True

        if self.enable_packing and "packer_state" in state_dict:
            packer_state = state_dict["packer_state"]
            self.packer._sample_buffer = dict(enumerate(packer_state["sample_buffer"]))
            self.packer._next_id = len(packer_state["sample_buffer"])
            self.packer.packed_samples.clear()
            self.packer.packed_samples.extend(packer_state["packed_samples"])

    def state_dict(self):
        state = {
            "sample_idx": self._sample_idx,
            "batch_samples": self._batch_samples,
            "num_batch_tokens": self._num_batch_tokens,
        }

        # Save HF dataset state for fast resume if supported
        if hasattr(self._data, "state_dict"):
            state["hf_dataset_state"] = self._data.state_dict()

        if self.enable_packing:
            # pyrefly: ignore [bad-typed-dict-key]
            state["packer_state"] = {
                "sample_buffer": list(self.packer._sample_buffer.values()),
                "packed_samples": list(self.packer.packed_samples),
            }

        return state


class MMDataLoader(ParallelAwareDataloader):
    """Configurable multimodal dataloader for VLM training."""

    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        dataset: str = "cc12m-test"
        """Dataset to use"""

        dataset_subset: str = ""
        """Dataset subset/config name."""

        infinite: bool = True
        """Whether to loop the dataset infinitely"""

        # Batching configs
        packing_buffer_size: int = 0
        """Set to a value >0 to enable sample packing."""

        max_images_per_batch: int
        """Max images per batch to bound vision encoder memory."""

        # Preprocessing configs
        patch_size: int
        """Patch size of the vision encoder."""

        temporal_patch_size: int
        """Temporal patch size for video processing."""

        spatial_merge_size: int
        """Spatially merge visual tokens after encoder. e.g. 2 means 2x2=4 patches merged."""

        patch_order: Literal["block", "raster"] = "block"
        """Patch sequence layout the collator emits: ``"block"`` (each
        ``spatial_merge_size**2`` group contiguous, or ``"raster"`` (row-major).
        Must be ``"block"`` when ``build_mrope_positions`` is set."""

        resize_fn: Annotated[
            Callable[..., tuple[int, int, int, int]], tyro.conf.Suppress
        ] = resize_to_pixel_budget
        """Image-resize strategy (a callable, like ``sample_processor``):
        ``resize_to_pixel_budget`` or ``resize_to_patch_budget`` (cap patches at
        ``max_patches``, pad to a ``patch_size * spatial_merge_size`` multiple).
        Both share the signature ``(h, w, *, patch_size, merge_size,
        **budget) -> (resize_h, resize_w, pad_h, pad_w)``."""

        min_pixels: int
        """Minimum number of pixels for image resizing (pixel-budget strategy)."""

        max_pixels: int
        """Maximum number of pixels for image resizing (pixel-budget strategy)."""

        max_patches: int = 4096
        """Max raw patches per image."""

        max_patches_per_side: int = 512
        """Per-side patch cap for the vision position-embedding grid (``navit``)."""

        image_mean: tuple[float, ...]
        """Per-channel mean for image normalization."""

        image_std: tuple[float, ...]
        """Per-channel std for image normalization."""

        video_dir: str = ""
        """Base directory for video files (for datasets with video filename references)."""

        video_fps: float = 2.0
        """Target frames per second for video sampling."""

        video_min_frames: int = 4
        """Minimum number of frames to sample from a video."""

        video_max_frames: int = 768
        """Maximum number of frames to sample from a video."""

        # Other loading configs
        build_mrope_positions: bool = False
        """Build 3D MRoPE position IDs (``mrope_positions``) for models that use
        multi-dimensional RoPE"""

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: MultiModalTokenizer,
        max_context_length: int,
        num_tokens_per_batch: int,
        **kwargs,
    ):
        if num_tokens_per_batch < max_context_length:
            raise ValueError(
                "num_tokens_per_batch must be greater than or equal to "
                "max_context_length so an accepted multimodal sample fits in one batch."
            )
        dataset = HuggingFaceMultiModalDataset(
            dataset_name=config.dataset,
            dataset_path=config.dataset_path,
            tokenizer=tokenizer,
            max_context_length=max_context_length,
            num_tokens_per_batch=num_tokens_per_batch,
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            spatial_merge_size=config.spatial_merge_size,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            image_mean=config.image_mean,
            image_std=config.image_std,
            packing_buffer_size=config.packing_buffer_size,
            resize_fn=config.resize_fn,
            max_patches=config.max_patches,
            max_patches_per_side=config.max_patches_per_side,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=config.infinite,
            video_dir=config.video_dir,
            video_fps=config.video_fps,
            video_min_frames=config.video_min_frames,
            video_max_frames=config.video_max_frames,
            dataset_subset=config.dataset_subset,
        )

        collate_fn = MultiModalCollator(
            num_tokens_per_batch=num_tokens_per_batch,
            max_images_per_batch=config.max_images_per_batch,
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            spatial_merge_size=config.spatial_merge_size,
            tokenizer=tokenizer,
            build_mrope_positions=config.build_mrope_positions,
            patch_order=config.patch_order,
        )

        dataloader_kwargs = {
            "num_workers": config.num_workers,
            "persistent_workers": config.persistent_workers,
            "pin_memory": config.pin_memory,
            "prefetch_factor": config.prefetch_factor,
            "batch_size": None,
            "collate_fn": collate_fn,
        }

        super().__init__(
            dataset,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            **dataloader_kwargs,
        )
