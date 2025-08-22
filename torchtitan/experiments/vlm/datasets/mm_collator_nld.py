# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Dict, List

import einops as E
import torch
from torch.nn.utils.rnn import pad_sequence

from torchtitan.tools.logging import logger


IGNORE_INDEX = -100


@dataclass
class MultiModalCollatorNLD:
    """Collator that works with patches in NLD format (N=batch, L=patches, D=patch_features)"""

    padding_idx: int = 0
    ignore_idx: int = IGNORE_INDEX
    max_images_per_batch: int = 5
    max_patch_per_image: int = 256  # Maximum patches per image
    patch_size: int = 16  # Patch size for converting images to patches
    merge_size: int = 1  # Merge size for converting spatial patches to channel dim
    seq_len: int = 2048

    def convert_to_patches(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Direct NTHWC -> NLD conversion using einops."""
        N, T, H, W, C = pixel_values.shape
        ps = self.patch_size
        device = pixel_values.device
        patches = E.rearrange(
            pixel_values, "n t (h p1) (w p2) c -> n (t h w) (p1 p2 c)", p1=ps, p2=ps
        )

        coords = torch.meshgrid(
            torch.arange(T, device=device),
            torch.arange(H // ps, device=device),
            torch.arange(W // ps, device=device),
            indexing="ij",
        )
        grid = E.rearrange(torch.stack(coords), "coords t h w -> (t h w) coords")
        grid = grid.unsqueeze(0).expand(N, -1, -1)  # (N, t*h*w, 3)

        # All patches are valid since we resize images to be divisible by patch_size
        return patches, grid

    def _pad_to_max(self, patches, grids):
        """Pad or truncate to max_patch_per_image."""
        N, L, D = patches.shape
        if L == self.max_patch_per_image:
            return patches, grids
        elif L < self.max_patch_per_image:
            # Pad
            pad_len = self.max_patch_per_image - L
            zero_patches = torch.zeros(N, pad_len, D, device=patches.device)
            invalid_grids = torch.full(
                (grids.shape[0], pad_len, 3), -1, device=grids.device
            )
            return torch.cat([patches, zero_patches], 1), torch.cat(
                [grids, invalid_grids], 1
            )
        else:
            # Truncate
            return (
                patches[:, : self.max_patch_per_image],
                grids[:, : self.max_patch_per_image],
            )

    def __call__(
        self, batch: List[Dict[str, Any]]
    ) -> tuple[Dict[str, torch.Tensor | None], torch.Tensor]:
        """Encode batch with patch-based approach."""
        if not batch:
            return None

        # Count images per sample and total images
        images_per_sample = []
        for sample in batch:
            num_images = (
                len(sample.get("pixel_values", [])) if "pixel_values" in sample else 0
            )
            images_per_sample.append(num_images)

        # Remove samples from end until total images <= max_images_per_batch
        total_images = sum(images_per_sample)
        while total_images > self.max_images_per_batch and batch:
            removed_images = images_per_sample.pop()
            total_images -= removed_images
            batch.pop()
            logger.warning(f"Removed sample with {removed_images} images to keep total images <= {self.max_images_per_batch}")

        all_images = [
            img
            for sample in batch
            if "pixel_values" in sample
            for img in sample["pixel_values"]
        ]

        if all_images:
            patch_list, grid_list = [], []
            for img in all_images:
                p, g = self.convert_to_patches(img.unsqueeze(0))
                p, g = self._pad_to_max(p, g)
                patch_list.append(p[0])
                grid_list.append(g[0])
            patches = torch.stack(patch_list)
            grids = torch.stack(grid_list)

            if len(all_images) < self.max_images_per_batch:
                blank_count = self.max_images_per_batch - len(all_images)
                blank_patches = torch.zeros(
                    blank_count,
                    self.max_patch_per_image,
                    patches.shape[2],
                    device=patches.device,
                )
                blank_grids = torch.full(
                    (blank_count, self.max_patch_per_image, 3), -1, device=grids.device
                )
                patches = torch.cat([patches, blank_patches], dim=0)
                grids = torch.cat([grids, blank_grids], dim=0)
        else:
            patches = grids = None

        # Text processing
        input_ids = pad_sequence(
            [s["input_ids"] for s in batch],
            batch_first=True,
            padding_value=self.padding_idx,
        )
        labels = pad_sequence(
            [s["labels"] for s in batch],
            batch_first=True,
            padding_value=self.padding_idx,
        )

        # Pad along batch dimension if needed
        batch_size = len(batch)
        if input_ids.size(0) < batch_size:
            padding_needed = batch_size - input_ids.size(0)
            padding_input = (
                torch.ones(padding_needed, input_ids.size(1), dtype=torch.long)
                * self.padding_idx
            )
            padding_labels = (
                torch.ones(padding_needed, labels.size(1), dtype=torch.long)
                * self.padding_idx
            )
            input_ids = torch.cat([input_ids, padding_input], dim=0)
            labels = torch.cat([labels, padding_labels], dim=0)

        # Handle sequence length
        current_length = input_ids.size(1)
        desired_length = self.seq_len + 1  # Extra token for label shift and cut
        if current_length < desired_length:
            padding_length = desired_length - current_length
            padding_input = (
                torch.ones(batch_size, padding_length, dtype=torch.long)
                * self.padding_idx
            )
            padding_labels = (
                torch.ones(batch_size, padding_length, dtype=torch.long)
                * self.padding_idx
            )
            input_ids = torch.cat([input_ids, padding_input], dim=1)
            labels = torch.cat([labels, padding_labels], dim=1)
        elif current_length > self.seq_len:
            input_ids = input_ids[:, :desired_length]
            labels = labels[:, :desired_length]

        labels[labels == self.padding_idx] = self.ignore_idx
        # Cut and shift
        input_ids = input_ids[:, :-1]
        labels = labels[:, 1:]

        return {
            "input": input_ids,
            "pixel_values": patches,
            "grid_thw": grids,
        }, labels
