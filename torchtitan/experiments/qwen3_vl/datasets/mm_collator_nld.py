# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multimodal collator for Qwen3-VL datasets."""

from dataclasses import dataclass
from typing import Any

import einops as E
import torch
from torch.nn.utils.rnn import pad_sequence

from torchtitan.tools.logging import logger

from ..model.args import SpecialTokens

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
            pixel_values: Padded patches (num_images, max_seq_len, patch_dim) or None
            grid_thw: Grid dimensions (num_images, 3) with [T, H_patches, W_patches] or None
        """
        if not all_images:
            return None, None

        all_patches = []
        grid_thw_list = []
        merge_size = self.spatial_merge_size

        for img in all_images:
            T, H, W, C = img.shape
            ps = self.patch_size
            ts = self.temporal_patch_size

            # Pad temporal dimension if needed
            if T % ts != 0:
                pad_t = ts - (T % ts)
                img = torch.nn.functional.pad(img, (0, 0, 0, 0, 0, 0, 0, pad_t))
                T = img.shape[0]

            # Calculate grid dimensions (in patches, before merging)
            T_patches = T // ts
            H_patches = H // ps
            W_patches = W // ps

            # Convert to patches in block-order (matching position embedding order)
            patches = E.rearrange(
                img,
                "(t pt) (bh m ph) (bw n pw) c -> (t bh bw m n) (pt ph pw c)",
                pt=ts,
                ph=ps,
                pw=ps,
                m=merge_size,
                n=merge_size,
            )

            all_patches.append(patches)
            grid_thw_list.append(torch.tensor([T_patches, H_patches, W_patches]))

        if not all_patches:
            return None, None

        # Pad to same length for batched processing
        # Ensure max_seq_len is divisible by spatial_merge_size^2 for merger
        merge_unit = merge_size ** 2
        max_seq_len = max(p.shape[0] for p in all_patches)
        if max_seq_len % merge_unit != 0:
            max_seq_len = ((max_seq_len // merge_unit) + 1) * merge_unit

        patch_dim = all_patches[0].shape[1]

        padded_patches = torch.zeros(len(all_patches), max_seq_len, patch_dim)
        for i, patches in enumerate(all_patches):
            padded_patches[i, :patches.shape[0]] = patches

        grid_thw = torch.stack(grid_thw_list, dim=0)  # (num_images, 3)

        return padded_patches, grid_thw

    def collate_text(
        self,
        batch: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process text inputs and labels from batch."""
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

        input_ids, labels = pad_text_batch(
            input_ids,
            labels,
            self.seq_len + 1,
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
