# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from io import BytesIO
from typing import Any

import numpy as np
import requests
import torch
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node

from PIL import Image
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger

from .mm_collator_nld import MultiModalCollatorNLD


IGNORE_INDEX = -100
# TODO: should add this to the tokenizer
BEGIN_OF_IMAGE_TOKEN = "<|begin_of_image|>"
END_OF_IMAGE_TOKEN = "<|end_of_image|>"
IMAGE_TOKEN = "<|image|>"


def smart_resize(
    height: int,
    width: int,
    factor: int = 16,
    min_pixels: int = 16 * 16 * 16,
    max_pixels: int = 16 * 16 * 4 * 1280,
):
    if height < factor or width < factor:
        raise ValueError(
            f"height:{height} or width:{width} must be larger than factor:{factor}"
        )
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = np.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = np.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def resize_image_by_patch_count(
    image,
    max_patch_per_image,
    patch_size=16,
    merge_size=1,
    min_dimension=56,
):
    """Resize image while maintaining aspect ratio and ensuring patch count <= max_patch_per_image.

    Args:
        image: PIL Image or image bytes
        max_patch_per_image: Maximum number of patches (L) allowed per image
        patch_size: Size of each patch (default: 16)
        merge_size: Spatial Merge size factor (default: 1)
        min_dimension: Minimum dimension for width/height (default: 56)

    Returns:
        Resized PIL Image with dimensions divisible by factor and patches <= max_patch_per_image
    """
    if not isinstance(image, Image.Image):
        image = Image.open(BytesIO(image))

    original_width, original_height = image.size
    factor = patch_size * merge_size

    # Calculate current number of patches
    current_patches = (original_height * original_width) // (factor * factor)

    # If already within limits and divisible, return as-is after smart_resize
    if current_patches <= max_patch_per_image:
        try:
            resized_height, resized_width = smart_resize(
                original_height, original_width, factor=factor
            )
            return image.resize((resized_width, resized_height))
        except ValueError:
            # If smart_resize fails, continue with scaling
            pass

    # Calculate maximum area that gives us max_patch_per_image patches
    max_area = max_patch_per_image * (factor * factor)

    # Calculate scaling factor to fit within max_area while maintaining aspect ratio
    current_area = original_width * original_height
    scale_factor = math.sqrt(max_area / current_area)

    # Scale dimensions
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Ensure minimum dimensions
    if new_width < min_dimension:
        new_width = min_dimension
        new_height = int(new_width * original_height / original_width)
    if new_height < min_dimension:
        new_height = min_dimension
        new_width = int(new_height * original_width / original_height)

    # Use smart_resize to ensure divisibility and handle constraints
    try:
        resized_height, resized_width = smart_resize(
            new_height, new_width, factor=factor
        )
    except ValueError:
        # If smart_resize fails, fall back to manual rounding
        resized_height = (new_height // factor) * factor
        resized_width = (new_width // factor) * factor
        resized_height = max(factor, resized_height)  # Ensure at least one patch
        resized_width = max(factor, resized_width)

    # Final verification: ensure patch count is within limit
    final_patches = (resized_height * resized_width) // (factor * factor)
    if final_patches > max_patch_per_image:
        # Reduce dimensions proportionally
        reduction_factor = math.sqrt(max_patch_per_image / final_patches)
        resized_height = int(resized_height * reduction_factor)
        resized_width = int(resized_width * reduction_factor)

        # Round down to nearest factor multiple
        resized_height = (resized_height // factor) * factor
        resized_width = (resized_width // factor) * factor
        resized_height = max(factor, resized_height)
        resized_width = max(factor, resized_width)

    resized_image = image.resize((resized_width, resized_height))
    return resized_image


def calculate_image_tokens(image, patch_size=16, merge_size=1):
    """Calculate tokens for an image based on patch size."""
    if isinstance(image, torch.Tensor):
        height, width = image.shape[-2:]
    else:
        width, height = image.size
    return (
        int((height * width) / (patch_size * patch_size * merge_size * merge_size)),
        int(width / (patch_size * merge_size)),
        int(height / (patch_size * merge_size)),
    )


class MultiModalDataset(IterableDataset, Stateful):
    """PyTorch MultiModal Dataset."""

    def __init__(
        self,
        dataset_path: str,
        tokenizer: BaseTokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        patch_size: int = 16,
        merge_size: int = 1,
        max_patch_per_image: int = 256,
        max_images_per_batch: int = 4,
    ) -> None:
        ds = load_dataset(dataset_path, split="train", streaming=True)
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self._sample_idx = 0
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.max_patch_per_image = max_patch_per_image
        self.max_images_per_batch = max_images_per_batch

    def _process_sample(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        """Process a single sample into the required format."""
        try:
            # Get texts, images and metadata
            texts = sample.get("texts", [])
            images = sample.get("images", [])
            metadata = sample.get("metadata", [])

            if not texts or len(texts) != len(images):
                logger.warning(
                    f"Invalid sample: texts={len(texts)}, images={len(images)}"
                )
                return None

            # Process images and build interleaved text
            processed_images = []
            processed_text = ""

            for i, (img, txt) in enumerate(zip(images, texts)):
                # Add text if it exists
                if txt is not None:
                    processed_text += txt

                # Try to get image if it exists
                if img is not None:
                    try:
                        # Handle online case (image URLs)
                        if isinstance(img, str) and img.startswith("http"):
                            response = requests.get(img)
                            img = Image.open(BytesIO(response.content))
                        # Handle offline/cached case
                        elif isinstance(img, bytes):
                            img = Image.open(BytesIO(img))
                        elif isinstance(img, str):
                            img = Image.open(img)

                        if img.mode != "RGB":
                            img = img.convert("RGB")

                        # Resize maintaining aspect ratio
                        img = resize_image_by_patch_count(
                            img,
                            max_patch_per_image=self.max_patch_per_image,
                            patch_size=self.patch_size,
                            merge_size=self.merge_size,
                        )

                        # Convert to numpy array and rescale to [0, 1]
                        img_array = np.array(img) / 255.0

                        # Normalize with OpenAI CLIP mean/std
                        mean = np.array([0.48145466, 0.4578275, 0.40821073])
                        std = np.array([0.26862954, 0.26130258, 0.27577711])
                        img_array = (img_array - mean) / std

                        # Convert to tensor in NTHWC format (1, H, W, 3)
                        img_tensor = torch.from_numpy(img_array).float()
                        img_tensor = img_tensor.unsqueeze(0)  # Add time dimension

                        # Calculate number of image tokens needed
                        (
                            num_tokens,
                            add_row_image_token_after,
                            _,
                        ) = calculate_image_tokens(
                            img, patch_size=self.patch_size, merge_size=self.merge_size
                        )

                        processed_images.append(img_tensor)
                        processed_text += BEGIN_OF_IMAGE_TOKEN

                        # Add image tokens with row separators following dataset_utils pattern
                        image_tokens = []
                        for token_idx in range(num_tokens):
                            image_tokens.append(IMAGE_TOKEN)

                        processed_text += "".join(image_tokens)
                        processed_text += END_OF_IMAGE_TOKEN

                    except Exception as e:
                        logger.warning(f"Error processing image {i}: {e}")

            if not processed_images:
                return None

            # Add EOS token and tokenize
            processed_text = processed_text + END_OF_IMAGE_TOKEN
            tokens = self._tokenizer.encode(processed_text)

            input_ids = torch.tensor(tokens)
            labels = torch.tensor(tokens)

            if len(input_ids) > self.seq_len:
                logger.warning(
                    f"Skipping sample with length {len(input_ids)} greater than max_seq_len {self.seq_len}"
                )
                return None

            # Get special token IDs just like in dataset_utils.py
            def _get_special_token_id(token):
                token_id = self._tokenizer.encode(token)
                assert (
                    len(token_id) == 1
                ), f"{token} is not a special token of the tokenizer"
                return token_id[0]

            special_tokens = [
                _get_special_token_id(token)
                for token in (
                    IMAGE_TOKEN,
                    BEGIN_OF_IMAGE_TOKEN,
                    END_OF_IMAGE_TOKEN,
                )
            ]

            labels = torch.where(
                torch.isin(labels, torch.tensor(special_tokens)), IGNORE_INDEX, labels
            )

            # No truncation here - let collator handle it

            # Keep images as list
            pixel_values = processed_images  # List of tensors

            return {
                "input_ids": input_ids,
                "labels": labels,
                "pixel_values": pixel_values,
            }

        except Exception as e:
            logger.warning(f"Error processing sample: {e}")
            return None

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                try:
                    processed = self._process_sample(sample)
                    if processed is None:
                        continue

                    # Simply yield individual samples - DataLoader will handle batching
                    self._sample_idx += 1
                    yield processed

                except Exception as e:
                    logger.warning(f"Error in iteration: {e}")
                    continue

            if not self.infinite:
                break
            else:
                self._sample_idx = 0

    def _get_data_iter(self):
        try:
            # For streaming datasets, we don't need to check length
            if not hasattr(self._data, "iterable_dataset"):
                if isinstance(self._data, Dataset) and self._sample_idx == len(
                    self._data
                ):
                    return iter([])

            it = iter(self._data)

            # Skip samples if needed
            if self._sample_idx > 0:
                for _ in range(self._sample_idx):
                    next(it)

            return it
        except Exception as e:
            logger.error(f"Error in _get_data_iter: {e}")
            return iter([])

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]

    def state_dict(self):
        return {"sample_idx": self._sample_idx}


def build_mm_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets."""
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len
    # TODO: config
    max_images_per_batch = batch_size * 2
    max_patch_per_image = 256
    patch_size = 16

    hf_ds = MultiModalDataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        patch_size=patch_size,
        merge_size=1,
        max_patch_per_image=max_patch_per_image,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
        max_images_per_batch=max_images_per_batch,
    )

    collate_fn = MultiModalCollatorNLD(
        padding_idx=0,
        max_images_per_batch=max_images_per_batch,
        max_patch_per_image=max_patch_per_image,
        patch_size=patch_size,
        merge_size=1,
        seq_len=seq_len,
    )

    base_dataloader = ParallelAwareDataloader(
        dataset=hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,  # Use micro_batch_size for initial batching
        collate_fn=collate_fn,
    )

    return base_dataloader
