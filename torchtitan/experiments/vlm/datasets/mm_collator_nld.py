# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from torchtitan.tools.logging import logger

from ..model.args import SpecialTokens

from .utils.image import (
    convert_to_patches,
    pad_empty_images_to_target_batch_size,
    pad_patches,
)
from .utils.text import pad_input_ids_and_labels_to_target_batch_size, pad_text_batch


@dataclass
class MultiModalCollatorNLD:
    """Multimodal collator that works with image patches in NLD format.
    N: Number of images (vision encoder's batch size)
    L: Length of patches (vision encoder's sequence length)
    D: Dimension of a patch (3 * spatial_patch_size**2 * temporal patch_size)

    This module provides a collator class that handles both image and text data,
    converting images to patches and preparing text for model input.

    Example:
        >>> # Initialize collator
        >>> collator = MultiModalCollatorNLD(
        ...     batch_size=2,
        ...     seq_len=32,
        ...     max_images_per_batch=4,
        ...     max_patch_per_image=6,
        ...     patch_size=16,
        ...     padding_idx=0,
        ... )
        >>>
        >>> # Create sample batch
        >>> batch = [
        ...     {
        ...         "input_ids": torch.tensor([1, 2, 3]),
        ...         "labels": torch.tensor([2, 3, 4]),
        ...         "pixel_values": [
        ...             torch.randn(1, 32, 32, 3),
        ...             torch.randn(1, 32, 48, 3)
        ...         ]
        ...     },
        ...     {
        ...         "input_ids": torch.tensor([5, 6]),
        ...         "labels": torch.tensor([6, 7]),
        ...         "pixel_values": [
        ...             torch.randn(1, 32, 32, 3)   # One image
        ...         ]
        ...     }
        ... ]
        >>>
        >>> # Collate batch
        >>> outputs = collator(batch)
        >>>
        >>> # Examine outputs
        >>> print(outputs["input_ids"].shape)     # (2, 32)     - Padded to seq_len
        >>> print(outputs["labels"].shape)        # (2, 32)     - Padded to seq_len
        >>> print(outputs["pixel_values"].shape)  # (4, 6, 768) - (N=4 images, L=6 patches, D=16*16*3)
        >>> print(outputs["grid_thw"].shape)      # (4, 6, 3)   - Coordinates for each patch
        >>>
        >>> # The collated batch has:
        >>> # 1. Text tensors padded to max length
        >>> # 2. Images converted to patches in NLD format
        >>> # 3. Grid coordinates for each patch
        >>> # 4. All tensors properly batched and padded
    """

    batch_size: int  # LLM's batch size
    seq_len: int  # LLM's maximum sequence length

    patch_size: int  # Patch size for converting images to patches
    max_images_per_batch: int  # Vision Encoder's batch size
    max_patches_per_image: int  # Vision Encoder's sequence length

    special_tokens: SpecialTokens

    def collate_images(
        self, all_images: list[torch.Tensor]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Process a list of image tensors into patches with coordinate grids.

        Args:
            all_images: list of image tensors, each of shape (T, H, W, 3)

        Returns:
            patches: Tensor of shape (N, L, D) or None if no images
            grids: Tensor of shape (N, L, 3) or None if no images
        """
        if not all_images:
            return None, None

        patch_list, grid_list = [], []
        for img in all_images:
            # Convert single image to patches
            patches, grids = convert_to_patches(img, patch_size=self.patch_size)

            # Pad/truncate to max patches
            patches, grids = pad_patches(patches, grids, self.max_patches_per_image)

            patch_list.append(patches)
            grid_list.append(grids)

        # Stack all images
        patches = torch.stack(patch_list)
        grids = torch.stack(grid_list)

        # Pad to max_images_per_batch with empty images
        patches, grids = pad_empty_images_to_target_batch_size(
            patches, grids, self.max_images_per_batch
        )

        return patches, grids

    def collate_text(
        self,
        batch: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process text inputs and labels from batch.

        Args:
            batch: list of dictionaries containing "input_ids" and "labels"

        Returns:
            input_ids: Tensor of shape (B, L)
            labels: Tensor of shape (B, L)

        Note:
            B = batch size (padded if needed)
            L = sequence length (padded/truncated to seq_len)
        """
        # Pad sequences in batch
        input_ids = pad_sequence(
            [s["input_ids"] for s in batch],
            batch_first=True,
            padding_value=self.special_tokens.pad_id,
        )
        labels = pad_sequence(
            [s["labels"] for s in batch],
            batch_first=True,
            padding_value=self.special_tokens.pad_id,
        )

        # Handle sequence length
        input_ids, labels = pad_text_batch(
            input_ids,
            labels,
            self.seq_len + 1,  # Extra token for label shifting
            padding_idx=self.special_tokens.pad_id,
            ignore_idx=self.special_tokens.ignore_id,
        )
        input_ids, labels = pad_input_ids_and_labels_to_target_batch_size(
            input_ids,
            labels,
            self.batch_size,
            padding_idx=self.special_tokens.pad_id,
            ignore_idx=self.special_tokens.ignore_id,
        )

        return input_ids[:, :-1], labels[:, 1:]  # Shift for next token prediction

    def __call__(
        self, batch: list[dict[str, Any]]
    ) -> tuple[dict[str, torch.Tensor | None], torch.Tensor]:
        """Encode batch with patch-based approach.

        Args:
            batch: list of dictionaries containing:
                - input_ids: Tensor of shape (S)
                - labels: Tensor of shape (L)
                - pixel_values: list of tensors, each (1, H, W, 3)

        Returns:
            Dictionary containing:
                - input_ids: Tensor of shape (B, L)
                - labels: Tensor of shape (B, L)
                - pixel_values: Tensor of shape (N, L, D)
                - grid_thw: Tensor of shape (N, L, 3)
        """
        # Count images per sample and total images
        images_per_sample = []
        for sample in batch:
            num_images = len(sample.get("pixel_values", []))
            images_per_sample.append(num_images)

        # Remove samples from end until total images <= max_images_per_batch
        total_images = sum(images_per_sample)
        while total_images > self.max_images_per_batch and batch:
            removed_images = images_per_sample.pop()
            total_images -= removed_images
            batch.pop()
            logger.warning(
                f"Removed sample with {removed_images} images to keep "
                f"total images <= {self.max_images_per_batch}"
            )

        # Process all images in batch
        all_images = [
            img
            for sample in batch
            if "pixel_values" in sample
            for img in sample["pixel_values"]
        ]
        patches, grids = self.collate_images(all_images)

        # Process text and pad to batch size
        input_ids, labels = self.collate_text(batch)
        input_dict = {
            "input": input_ids,
            "pixel_values": patches,
            "grid_thw": grids,
            "special_tokens": self.special_tokens,
        }

        return input_dict, labels
