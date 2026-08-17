# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multimodal collator for VLM datasets.

Shape conventions:
- T = packed tokens (text tokens or vision patches, depending on the tensor)
- Each ``grid_thw`` row is ``[t, h, w]``, where lowercase ``t`` is the
  temporal patch count for one visual item
"""

from dataclasses import dataclass
from typing import Any

import torch

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.tools.logging import logger
from .utils.image import vision_to_patches


@dataclass
class MultiModalCollator:
    """Multimodal collator for VLM training.

    Handles both image and text data, converting images to patches
    and preparing text for model input.
    """

    num_tokens_per_batch: int
    max_images_per_batch: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    tokenizer: MultiModalTokenizer
    build_mrope_positions: bool
    patch_order: str = "block"

    def collate_images(
        self, all_images: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a list of image/video tensors into packed patches with grid dimensions.

        Args:
            all_images: Non-empty list of image/video tensors, each of shape (T, H, W, C)

        Returns:
            pixel_values: Packed patches (num_patches, patch_dim).
            grid_thw: Grid dimensions (num_images, 3) with [T, H_patches, W_patches]

        ``grid_thw.prod(-1)`` gives each visual item's segment length in the
        packed patch sequence.
        """
        results = [
            vision_to_patches(
                img,
                self.patch_size,
                self.temporal_patch_size,
                self.spatial_merge_size,
                patch_order=self.patch_order,
            )
            for img in all_images
        ]
        all_patches = [r[0] for r in results]
        grid_thw_list = [r[1] for r in results]

        packed_patches = torch.cat(all_patches, dim=0)
        grid_thw = torch.stack(grid_thw_list, dim=0)  # (num_images, 3)

        return packed_patches, grid_thw

    def collate_text(
        self,
        batch: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Concatenate whole samples and pad only the token-batch tail."""
        inputs: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        positions: list[torch.Tensor] = []
        for sample in batch:
            inputs.append(sample["input_ids"][:-1])
            labels.append(sample["labels"][1:])
            positions.append(sample["positions"][:-1])

        input_ids = torch.cat(inputs)
        label_ids = torch.cat(labels)
        position_ids = torch.cat(positions)
        pad_len = self.num_tokens_per_batch - input_ids.shape[0]
        if pad_len > 0:
            input_ids = torch.nn.functional.pad(
                input_ids,
                (0, pad_len),
                # pyrefly: ignore [missing-attribute]
                value=self.tokenizer.pad_id,
            )
            label_ids = torch.nn.functional.pad(
                label_ids, (0, pad_len), value=IGNORE_INDEX
            )
            position_ids = torch.cat([position_ids, torch.arange(pad_len)])

        return input_ids, label_ids, position_ids

    def _build_mrope_positions(
        self,
        tokens: torch.Tensor,
        grid_thw: torch.Tensor | None,
        grid_thw_videos: torch.Tensor | None,
        positions: torch.Tensor | None,
        *,
        image_token_id: int,
        video_token_id: int,
    ) -> torch.Tensor:
        """Build 3D (temporal, height, width) MRoPE position IDs per token.

        Returns ``(num_tokens, 3)`` with temporal/height/width coordinates in
        the final dimension. Runs here on CPU data workers, off the GPU
        training path.

        Args:
            tokens: ``(num_tokens,)`` token IDs.
            grid_thw: (num_images, 3) image grid dims, or None.
            grid_thw_videos: (num_videos, 3) video grid dims, or None.
            positions: ``(num_tokens,)`` per-token positions; document
                boundaries are detected where positions reset.
            image_token_id: Placeholder token ID marking image positions.
            video_token_id: Placeholder token ID marking video positions.

        Returns:
            ``(num_tokens, 3)`` MRoPE position IDs.
        """
        # MRoPE position IDs are laid out in block order; a raster patch order
        # would desync them from the patch sequence.
        if self.patch_order != "block":
            raise ValueError(
                f"MRoPE requires patch_order='block', got {self.patch_order!r}."
            )

        # Expand each video [T, H, W] into T rows of [1, H, W] so each frame is
        # treated like an image; temporal position comes from frame ordering.
        if grid_thw_videos is not None:
            grid_thw_videos = torch.repeat_interleave(
                grid_thw_videos, grid_thw_videos[:, 0], dim=0
            )
            grid_thw_videos[:, 0] = 1

        spatial_merge_size = self.spatial_merge_size

        num_tokens = tokens.shape[0]
        if positions is not None:
            reset_indices = torch.where(positions[1:] < positions[:-1])[0] + 1
            doc_starts = [0] + reset_indices.tolist()
            doc_ranges = [
                (
                    doc_starts[d],
                    doc_starts[d + 1] if d + 1 < len(doc_starts) else num_tokens,
                )
                for d in range(len(doc_starts))
            ]
        else:
            doc_ranges = [(0, num_tokens)]

        # First token of each consecutive vision region (image or video).
        vision_mask = (tokens == image_token_id) | (tokens == video_token_id)
        prev_vision = torch.cat(
            [torch.zeros_like(vision_mask[:1]), vision_mask[:-1]], dim=0
        )
        vision_starts = torch.where(vision_mask & ~prev_vision)[0].tolist()
        grid_cache: dict[tuple[int, int, int], torch.Tensor] = {}

        image_index, video_index = 0, 0
        vision_start_index = 0
        llm_pos_ids_list: list[torch.Tensor] = []
        for doc_start, doc_end in doc_ranges:
            doc_pos_ids_list: list[torch.Tensor] = []
            doc_vision_starts: list[int] = []
            while (
                vision_start_index < len(vision_starts)
                and vision_starts[vision_start_index] < doc_end
            ):
                doc_vision_starts.append(vision_starts[vision_start_index])
                vision_start_index += 1

            pair_cursor = doc_start
            for vision_start in doc_vision_starts:
                if tokens[vision_start] == image_token_id:
                    # pyrefly: ignore [unsupported-operation]
                    t, h, w = grid_thw[image_index]
                    image_index += 1
                else:
                    # pyrefly: ignore [unsupported-operation]
                    t, h, w = grid_thw_videos[video_index]
                    video_index += 1

                llm_grid_t, llm_grid_h, llm_grid_w = (
                    int(t.item()),
                    int(h.item()) // spatial_merge_size,
                    int(w.item()) // spatial_merge_size,
                )
                text_len = vision_start - pair_cursor
                pos_id_offset = (
                    doc_pos_ids_list[-1].max() + 1 if doc_pos_ids_list else 0
                )
                # Text positions are sequential and identical on all three axes.
                doc_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + pos_id_offset
                )
                grid_key = (llm_grid_t, llm_grid_h, llm_grid_w)
                if grid_key not in grid_cache:
                    hw = llm_grid_h * llm_grid_w
                    t_index = (
                        torch.arange(llm_grid_t).view(-1, 1).expand(-1, hw).flatten()
                    )
                    h_index = (
                        torch.arange(llm_grid_h)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    grid_cache[grid_key] = torch.stack([t_index, h_index, w_index])
                doc_pos_ids_list.append(grid_cache[grid_key] + text_len + pos_id_offset)
                pair_cursor = vision_start + llm_grid_t * llm_grid_h * llm_grid_w

            if pair_cursor < doc_end:
                pos_id_offset = (
                    doc_pos_ids_list[-1].max() + 1 if doc_pos_ids_list else 0
                )
                text_len = doc_end - pair_cursor
                doc_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + pos_id_offset
                )
            llm_pos_ids_list.extend(doc_pos_ids_list)

        return torch.cat(llm_pos_ids_list, dim=1).T

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

        if self.build_mrope_positions and (
            grids is not None or video_grids is not None
        ):
            special_tokens = input_dict["special_tokens"]
            input_dict["mrope_positions"] = self._build_mrope_positions(
                input_ids,
                grids,
                video_grids,
                positions,
                image_token_id=special_tokens["image_id"],
                video_token_id=special_tokens["video_id"],
            )

        # pyrefly: ignore [bad-return]
        return input_dict, labels
