# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multimodal collator for VLM datasets."""

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.tools.logging import logger
from .utils.image import vision_to_patches
from .utils.text import pad_batch_dim, pad_seq_len


@dataclass
class MultiModalCollator:
    """Multimodal collator for VLM training.

    Handles both image and text data, converting images to patches
    and preparing text for model input.
    """

    batch_size: int
    seq_len: int
    max_images_per_batch: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    tokenizer: MultiModalTokenizer

    def collate_images(
        self, all_images: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a list of image/video tensors into padded patches with grid dimensions.

        Args:
            all_images: Non-empty list of image/video tensors, each of shape (T, H, W, C)

        Returns:
            pixel_values: Padded patches (num_images, max_num_patch, patch_dim)
            grid_thw: Grid dimensions (num_images, 3) with [T, H_patches, W_patches]

        NOTE: Both num_images and max_num_patch vary per batch.
        """
        results = [
            vision_to_patches(
                img, self.patch_size, self.temporal_patch_size, self.spatial_merge_size
            )
            for img in all_images
        ]
        all_patches = [r[0] for r in results]
        grid_thw_list = [r[1] for r in results]

        # Pad to same length for batched processing
        # Ensure max_num_patch is divisible by spatial_merge_size^2 for merger
        merge_unit = self.spatial_merge_size**2
        max_num_patch = max(p.shape[0] for p in all_patches)
        if max_num_patch % merge_unit != 0:
            max_num_patch = ((max_num_patch // merge_unit) + 1) * merge_unit

        patch_dim = all_patches[0].shape[1]

        padded_patches = torch.zeros(len(all_patches), max_num_patch, patch_dim)
        for i, patches in enumerate(all_patches):
            padded_patches[i, : patches.shape[0]] = patches

        grid_thw = torch.stack(grid_thw_list, dim=0)  # (num_images, 3)

        return padded_patches, grid_thw

    def collate_text(
        self,
        batch: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process text inputs, labels, and positions from batch."""
        # Pad sequences to the longest in the batch
        input_ids = pad_sequence(
            [s["input_ids"] for s in batch],
            batch_first=True,
            # pyrefly: ignore [missing-attribute]
            padding_value=self.tokenizer.pad_id,
        )
        labels = pad_sequence(
            [s["labels"] for s in batch],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        positions = pad_sequence(
            [s["positions"] for s in batch],
            batch_first=True,
            padding_value=0,
        )
        # Pad or truncate to seq_len + 1
        input_ids, labels = pad_seq_len(
            input_ids,
            labels,
            self.seq_len + 1,
            # pyrefly: ignore [missing-attribute]
            padding_idx=self.tokenizer.pad_id,
            ignore_idx=IGNORE_INDEX,
        )
        # Pad or truncate positions to seq_len + 1
        if positions.shape[1] < self.seq_len + 1:
            positions = torch.nn.functional.pad(
                positions,
                (0, self.seq_len + 1 - positions.shape[1]),
                value=0,
            )
        else:
            positions = positions[:, : self.seq_len + 1]
        # Pad dummy rows to reach target batch size
        input_ids, labels = pad_batch_dim(
            input_ids,
            labels,
            self.batch_size,
            # pyrefly: ignore [missing-attribute]
            padding_idx=self.tokenizer.pad_id,
            ignore_idx=IGNORE_INDEX,
        )
        if positions.shape[0] < self.batch_size:
            positions = torch.nn.functional.pad(
                positions,
                (0, 0, 0, self.batch_size - positions.shape[0]),
                value=0,
            )

        return input_ids[:, :-1], labels[:, 1:], positions[:, :-1]

    def __call__(
        self, batch: list[dict[str, Any]]
    ) -> tuple[dict[str, torch.Tensor | None], torch.Tensor]:
        """Collate batch with patch-based approach."""
        images_per_sample: list[int] = []
        for sample in batch:
            num_images = len(sample.get("pixel_values", []))
            for vid in sample.get("pixel_values_videos", []):
                num_images += vid.shape[0] // self.temporal_patch_size
            images_per_sample.append(num_images)

        total_images = sum(images_per_sample)
        while total_images > self.max_images_per_batch and batch:
            removed_images = images_per_sample.pop()
            total_images -= removed_images
            batch.pop()
            logger.warning(
                f"Removed sample with {removed_images} vision entries to keep "
                f"total <= {self.max_images_per_batch}"
            )

        all_images = [
            img
            for sample in batch
            if "pixel_values" in sample
            for img in sample["pixel_values"]
        ]
        patches, grids = self.collate_images(all_images) if all_images else (None, None)

        all_videos = [
            vid
            for sample in batch
            if "pixel_values_videos" in sample
            for vid in sample["pixel_values_videos"]
        ]
        video_patches, video_grids = (
            self.collate_images(all_videos) if all_videos else (None, None)
        )

        input_ids, labels, positions = self.collate_text(batch)
        input_dict = {
            "input": input_ids,
            "positions": positions,
            "pixel_values": patches,
            "grid_thw": grids,
            "pixel_values_videos": video_patches,
            "grid_thw_videos": video_grids,
            "special_tokens": {
                f"{name}_id": getattr(self.tokenizer, f"{name}_id")
                for name in self.tokenizer.TOKEN_FIELDS
            },
        }

        # pyrefly: ignore [bad-return]
        return input_dict, labels
