# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multimodal collator for Qwen3-VL datasets."""

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from torchtitan.tools.logging import logger

from ..model.args import SpecialTokens

from .utils.image import image_to_patches
from .utils.text import pad_input_ids_and_labels_to_target_batch_size, pad_text_batch


@dataclass
class MultiModalCollatorNLD:
    """Multimodal collator for Qwen3-VL that works with image patches in NLD format.

    N: Number of images (vision encoder's batch size)
    L: Length of patches (vision encoder's sequence length)
    D: Dimension of a patch

    Handles both image and text data, converting images to patches
    and preparing text for model input.
    """

    batch_size: int
    seq_len: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    max_images_per_batch: int
    max_patches_per_image: int  # This is merged patches count
    special_tokens: SpecialTokens

    def __post_init__(self):
        # Calculate raw patches limit (before spatial merging)
        self.max_raw_patches_per_image = self.max_patches_per_image * (self.spatial_merge_size ** 2)

    def collate_images(
        self, all_images: list[torch.Tensor]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Process a list of image tensors into padded patches with grid dimensions.

        Args:
            all_images: list of image tensors, each of shape (T, H, W, C)

        Returns:
            pixel_values: Padded patches (num_images, max_num_patch, patch_dim) or None
            grid_thw: Grid dimensions (num_images, 3) with [T, H_patches, W_patches] or None
        """
        if not all_images:
            return None, None

        results = [
            image_to_patches(img, self.patch_size, self.temporal_patch_size, self.spatial_merge_size)
            for img in all_images
        ]
        all_patches = [r[0] for r in results]
        grid_thw_list = [r[1] for r in results]

        # Pad to same length for batched processing
        # Ensure max_num_patch is divisible by spatial_merge_size^2 for merger
        merge_unit = self.spatial_merge_size ** 2
        max_num_patch = max(p.shape[0] for p in all_patches)
        if max_num_patch % merge_unit != 0:
            max_num_patch = ((max_num_patch // merge_unit) + 1) * merge_unit

        patch_dim = all_patches[0].shape[1]

        padded_patches = torch.zeros(len(all_patches), max_num_patch, patch_dim)
        for i, patches in enumerate(all_patches):
            padded_patches[i, :patches.shape[0]] = patches

        grid_thw = torch.stack(grid_thw_list, dim=0)  # (num_images, 3)

        return padded_patches, grid_thw

    def collate_text(
        self,
        batch: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process text inputs and labels from batch."""
        # Pad sequences to the longest in the batch
        input_ids = pad_sequence(
            [s["input_ids"] for s in batch],
            batch_first=True,
            padding_value=self.special_tokens.pad_id,
        )
        labels = pad_sequence(
            [s["labels"] for s in batch],
            batch_first=True,
            padding_value=self.special_tokens.ignore_id,
        )
        # Pad or truncate to seq_len + 1
        input_ids, labels = pad_text_batch(
            input_ids,
            labels,
            self.seq_len + 1,
            padding_idx=self.special_tokens.pad_id,
            ignore_idx=self.special_tokens.ignore_id,
        )
        # Pad dummy rows to reach target batch size
        input_ids, labels = pad_input_ids_and_labels_to_target_batch_size(
            input_ids,
            labels,
            self.batch_size,
            padding_idx=self.special_tokens.pad_id,
            ignore_idx=self.special_tokens.ignore_id,
        )

        return input_ids[:, :-1], labels[:, 1:]

    def __call__(
        self, batch: list[dict[str, Any]]
    ) -> tuple[dict[str, torch.Tensor | None], torch.Tensor]:
        """Collate batch with patch-based approach."""
        images_per_sample = []
        for sample in batch:
            num_images = len(sample.get("pixel_values", []))
            images_per_sample.append(num_images)

        total_images = sum(images_per_sample)
        while total_images > self.max_images_per_batch and batch:
            removed_images = images_per_sample.pop()
            total_images -= removed_images
            batch.pop()
            logger.warning(
                f"Removed sample with {removed_images} images to keep "
                f"total images <= {self.max_images_per_batch}"
            )

        all_images = [
            img
            for sample in batch
            if "pixel_values" in sample
            for img in sample["pixel_values"]
        ]
        patches, grids = self.collate_images(all_images)

        input_ids, labels = self.collate_text(batch)
        input_dict = {
            "input": input_ids,
            "pixel_values": patches,
            "grid_thw": grids,
            "special_tokens": self.special_tokens,
        }

        return input_dict, labels
