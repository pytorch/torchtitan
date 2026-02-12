# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multimodal dataset implementation for Qwen3-VL training.

This module provides dataset classes for handling multimodal data
including images, videos, and text for Qwen3-VL model training.
"""

from dataclasses import asdict
from typing import Any, Callable

import torch
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer, HuggingFaceTokenizer
from torchtitan.config import JobConfig
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.tools.logging import logger

from ..model.args import SpecialTokens
from .mm_collator_nld import MultiModalCollatorNLD
from .utils.image import calculate_image_tokens, process_image
from .utils.packing import SamplePacker
from .utils.text import process_text_with_images


def _process_mm_sample(
    texts: list[str] | str,
    images: list[bytes] | bytes,
    tokenizer: BaseTokenizer,
    patch_size: int,
    temporal_patch_size: int,
    max_patch_per_image: int,
    spatial_merge_size: int,
    special_tokens: SpecialTokens,
) -> dict[str, Any] | None:
    """Common processing logic for multimodal samples.

    Args:
        texts: List of strings with None indicating image positions
        images: List of image bytes with None for text positions
        tokenizer: Tokenizer for text processing
        patch_size: Size of image patches
        max_patch_per_image: Maximum patches per image
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
                    temporal_patch_size=temporal_patch_size,
                    max_patch_per_image=max_patch_per_image,
                )
                if processed_img is not None:
                    # Each (patch_size x temporal_patch_size) x (patch_size x temporal_patch_size)
                    # square block of pixels is mapped to one image token
                    num_tokens, tokens_per_row, num_rows = calculate_image_tokens(
                        processed_img,
                        patch_size=patch_size,
                        spatial_merge_size=spatial_merge_size,
                    )
                    processed_images.append(processed_img)
                    image_dimensions.append((num_tokens, tokens_per_row, num_rows))
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
    max_patch_per_image: int,
    special_tokens: SpecialTokens,
) -> dict[str, Any] | None:
    """Process a sample from the OBELICS dataset (interleaved text and images)."""
    return _process_mm_sample(
        texts=sample.get("texts", []),
        images=sample.get("images", []),
        tokenizer=tokenizer,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=spatial_merge_size,
        max_patch_per_image=max_patch_per_image,
        special_tokens=special_tokens,
    )


def _process_cc12_wd_sample(
    sample: dict[str, Any],
    tokenizer: BaseTokenizer,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
    max_patch_per_image: int,
    special_tokens: SpecialTokens,
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
        max_patch_per_image=max_patch_per_image,
        special_tokens=special_tokens,
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
    """Validate dataset name and path."""
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
        max_patches_per_image: int,
        max_images_per_batch: int,
        packing_buffer_size: int,
        special_tokens: SpecialTokens,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        dataset_name = dataset_name.lower()

        path, dataset_loader, self.sample_processor = _validate_mm_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path)
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)

        self._tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.max_patches_per_image = max_patches_per_image
        self.max_images_per_batch = max_images_per_batch
        self.special_tokens = special_tokens
        self.enable_packing = packing_buffer_size > 0
        if self.enable_packing:
            self.packer = SamplePacker(
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
                        max_patch_per_image=self.max_patches_per_image,
                        special_tokens=self.special_tokens,
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
            state["packer_state"] = {
                "sample_buffer": list(self.packer.sample_buffer),
                "packed_samples": list(self.packer.packed_samples),
            }

        return state


def build_mm_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: HuggingFaceTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for Qwen3-VL multimodal datasets.

    Args:
        dp_world_size: Data parallel world size.
        dp_rank: Data parallel rank.
        tokenizer: Tokenizer for text processing.
        job_config: Job configuration containing dataset and DataLoader settings.
        infinite: Whether to loop infinitely.

    Returns:
        DataLoader with appropriate parallelism handling.
    """
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len

    max_images_per_batch = job_config.data.max_images_per_batch
    max_patches_per_image = job_config.data.max_patches_per_image
    patch_size = job_config.data.patch_size
    temporal_patch_size = getattr(job_config.data, "temporal_patch_size", 2)
    spatial_merge_size = job_config.data.spatial_merge_size
    packing_buffer_size = job_config.data.packing_buffer_size
    special_tokens = SpecialTokens.from_tokenizer(tokenizer)

    dataset = HuggingFaceMultiModalDataset(
        dataset_name=job_config.training.dataset,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        seq_len=seq_len,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=spatial_merge_size,
        max_patches_per_image=max_patches_per_image,
        max_images_per_batch=max_images_per_batch,
        packing_buffer_size=packing_buffer_size,
        special_tokens=special_tokens,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    collate_fn = MultiModalCollatorNLD(
        batch_size=batch_size,
        seq_len=job_config.training.seq_len,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=spatial_merge_size,
        max_images_per_batch=max_images_per_batch,
        max_patches_per_image=max_patches_per_image,
        special_tokens=special_tokens,
    )

    dataloader_kwargs = {
        **asdict(job_config.training.dataloader),
        "batch_size": batch_size,
        "collate_fn": collate_fn,
    }

    base_dataloader = ParallelAwareDataloader(
        dataset=dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        **dataloader_kwargs,
    )

    return base_dataloader
