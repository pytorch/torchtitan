# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multimodal collator for text and media datasets."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch

from torchtitan.components.data.collators import Collator
from torchtitan.components.data.multimodal.image_processor import ImageProcessor
from torchtitan.components.data.types import (
    DatasetBuildContext,
    TokenizedTrainingMicrobatch,
)
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.models.deepseek_v3.mtp import get_mtp_token_counts

from .mrope import get_mrope_positions


class MultiModalCollator(Collator):
    """Prepare text and optional image, video, or audio model inputs.

    Images and videos arrive resized and normalized by the sample processor
    (``ImageProcessor.resize_and_normalize`` /
    ``VideoProcessor.resize_and_normalize_video``); the collator patchifies
    them (``ImageProcessor.patchify``) and reads the processors' spatial merge
    size for M-RoPE positions.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Collator.Config):
        """Configure media collation.

        Audio transport is enabled only when ``waveform_pad_multiple`` and
        ``audio_input_channels`` are both set.
        """

        max_images_per_microbatch: int = 128
        build_mrope_positions: bool = False
        waveform_pad_multiple: int | None = None
        audio_input_channels: int | None = None

        def __post_init__(self) -> None:
            if (self.waveform_pad_multiple is None) != (
                self.audio_input_channels is None
            ):
                raise ValueError(
                    "waveform_pad_multiple and audio_input_channels must be "
                    "provided together"
                )
            for name in ("waveform_pad_multiple", "audio_input_channels"):
                value = getattr(self, name)
                if value is not None and value <= 0:
                    raise ValueError(f"{name} must be positive")

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        self._num_tokens_per_microbatch = context.num_tokens_per_microbatch
        self._max_context_length = context.max_context_length
        self._num_mtp_layers = context.num_mtp_layers
        self.max_images_per_microbatch = config.max_images_per_microbatch
        self.image_processor = context.image_processor
        self.video_processor = context.video_processor
        if config.build_mrope_positions:
            # MRoPE position IDs are laid out in block order; a raster patch
            # order would desync them from the patch sequence.
            for processor in (self.image_processor, self.video_processor):
                if processor is not None and processor.config.patch_order != "block":
                    raise ValueError(
                        "MRoPE requires patch_order='block', got "
                        f"{processor.config.patch_order!r}."
                    )
        self.tokenizer = cast(MultiModalTokenizer, context.tokenizer)
        self.build_mrope_positions = config.build_mrope_positions
        self.waveform_pad_multiple = config.waveform_pad_multiple
        self.audio_input_channels = config.audio_input_channels

    def collate_images(
        self, all_images: list[torch.Tensor], processor: ImageProcessor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process image/video tensors into packed patches and grid dimensions.

        Args:
            all_images: Non-empty list of resized, normalized image/video tensors,
                each of shape (T, H, W, C)
            processor: The image or video processor that resized them; its
                ``patchify`` cuts them into patches.

        Returns:
            pixel_values: Packed patches (num_patches, patch_dim)
            grid_thw: Grid dimensions (num_images, 3) with [T, H_patches, W_patches]

        ``grid_thw.prod(-1)`` gives each item's length in the patch sequence.
        """
        # The sample processor needs the processor to produce the items.
        assert processor is not None
        # TODO: normalize here too, so the sample processor only decodes and
        # resizes and packing buffers hold uint8 images (4x fewer bytes than
        # float32). That needs ImageProcessor to expose resize and normalize as
        # separate steps.
        results = [processor.patchify(img) for img in all_images]
        all_patches = [r[0] for r in results]
        grid_thw_list = [r[1] for r in results]

        packed_patches = torch.cat(all_patches, dim=0)
        grid_thw = torch.stack(grid_thw_list, dim=0)  # (num_images, 3)

        return packed_patches, grid_thw

    def collate_text(
        self,
        rows: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Concatenate whole samples and pad only the token-microbatch tail."""
        input_ids = torch.cat([sample["input_ids"] for sample in rows])
        labels = torch.cat([sample["labels"] for sample in rows])
        positions = torch.cat([sample["positions"] for sample in rows])
        padding_mask = torch.cat(
            [
                (
                    sample["padding_mask"]
                    if "padding_mask" in sample
                    else torch.zeros(sample["input_ids"].shape[0], dtype=torch.bool)
                )
                for sample in rows
            ]
        )
        pad_len = self._num_tokens_per_microbatch - input_ids.shape[0]
        if pad_len < 0:
            raise ValueError("multimodal rows exceed the configured token microbatch")
        if pad_len:
            input_ids = torch.nn.functional.pad(
                input_ids,
                (0, pad_len),
                # pyrefly: ignore [missing-attribute]
                value=self.tokenizer.pad_id,
            )
            labels = torch.nn.functional.pad(labels, (0, pad_len), value=IGNORE_INDEX)
            padding_positions = (
                torch.arange(pad_len, dtype=positions.dtype) % self._max_context_length
            )
            positions = torch.cat([positions, padding_positions])
            padding_mask = torch.nn.functional.pad(
                padding_mask, (0, pad_len), value=True
            )

        return input_ids, labels, positions, padding_mask

    def collate_audio(
        self, all_waveforms: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Independently pad and concatenate audio clips.

        Returns concatenated samples, true clip lengths, and padded clip
        lengths. Prefix sums of the padded lengths delimit storage segments;
        each true length gives the valid sample count within its segment.
        """
        assert self.waveform_pad_multiple is not None
        assert self.audio_input_channels is not None

        padded_waveforms = []
        waveform_lengths = []
        waveform_padded_lengths = []
        for index, waveform in enumerate(all_waveforms):
            if not isinstance(waveform, torch.Tensor) or waveform.ndim != 2:
                raise ValueError(f"waveform {index} must be a rank-2 tensor")
            if waveform.shape[0] == 0:
                raise ValueError(f"waveform {index} must be nonempty")
            if waveform.shape[1] != self.audio_input_channels:
                raise ValueError(
                    f"waveform {index} has {waveform.shape[1]} channels, expected "
                    f"{self.audio_input_channels}"
                )

            length = waveform.shape[0]
            padded_length = (
                (length + self.waveform_pad_multiple - 1)
                // self.waveform_pad_multiple
                * self.waveform_pad_multiple
            )
            padded_waveforms.append(
                torch.nn.functional.pad(waveform, (0, 0, 0, padded_length - length))
            )
            waveform_lengths.append(length)
            waveform_padded_lengths.append(padded_length)

        return (
            torch.cat(padded_waveforms),
            torch.tensor(waveform_lengths, dtype=torch.int64),
            torch.tensor(waveform_padded_lengths, dtype=torch.int64),
        )

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

        Finds each packed document's image and video placeholder runs and builds the
        document's positions with `get_mrope_positions`, which the RL generator and
        trainer use too. Runs here on CPU data workers, off the GPU path.

        Args:
            tokens: (num_tokens,) token IDs.
            grid_thw: (num_images, 3) image grid dims, or None.
            grid_thw_videos: (num_videos, 3) video grid dims, or None.
            positions: (num_tokens,) per-token positions; document
                boundaries are detected where positions reset.
            image_token_id: Placeholder token ID marking image positions.
            video_token_id: Placeholder token ID marking video positions.

        Returns:
            (num_tokens, 3) MRoPE position IDs.
        """
        # Transformers splits Qwen3.5 video grids into per-frame entries because
        # timestamps split its video tokens into matching modality runs. Our data
        # pipeline emits one contiguous placeholder run per video instead, so each
        # run must consume the original [T, H, W] grid as a single 3D region.

        num_tokens = tokens.shape[0]
        # With sample packing, a sample holds several documents, and each starts at
        # position 0. A decrease check misses 0 -> 0 boundaries after padding or a
        # single-token document.
        doc_starts = [0]
        if positions is not None:
            doc_starts += (torch.where(positions[1:] == 0)[0] + 1).tolist()
        doc_ends = doc_starts[1:] + [num_tokens]

        # First token of each consecutive vision run (image or video); the runs take
        # the image and video grids in token order, merged by their processor.
        vision_mask = (tokens == image_token_id) | (tokens == video_token_id)
        previous_is_vision = torch.cat([vision_mask.new_zeros(1), vision_mask[:-1]])
        vision_starts = torch.where(vision_mask & ~previous_is_vision)[0].tolist()
        image_grids = iter(grid_thw.tolist() if grid_thw is not None else [])
        video_grids = iter(
            grid_thw_videos.tolist() if grid_thw_videos is not None else []
        )

        mrope_positions = torch.empty(
            num_tokens, 3, dtype=tokens.dtype, device=tokens.device
        )
        for doc_start, doc_end in zip(doc_starts, doc_ends):
            vision_items = []
            for start in vision_starts:
                if not doc_start <= start < doc_end:
                    continue
                if tokens[start] == image_token_id:
                    t, h, w = next(image_grids)
                    processor = self.image_processor
                else:
                    t, h, w = next(video_grids)
                    processor = self.video_processor
                assert processor is not None
                merge = processor.config.spatial_merge_size
                vision_items.append((start - doc_start, (t, h // merge, w // merge)))
            mrope_positions[doc_start:doc_end] = get_mrope_positions(
                doc_end - doc_start, vision_items
            )
        return mrope_positions

    def __call__(self, rows: Sequence[dict[str, Any]]) -> TokenizedTrainingMicrobatch:
        """Collate rows into one multimodal training microbatch.

        Audio batches add ``waveforms``, ``waveform_lengths``, and
        ``waveform_padded_lengths`` to model kwargs. Non-audio batches do not.
        """
        rows = list(rows)
        all_waveforms = [
            waveform for sample in rows for waveform in sample.get("waveforms", [])
        ]
        audio_enabled = self.waveform_pad_multiple is not None
        if all_waveforms and not audio_enabled:
            raise ValueError("audio rows cannot be collated when audio is disabled")

        all_images = [
            image for sample in rows for image in sample.get("pixel_values", [])
        ]
        patches, grids = (
            self.collate_images(all_images, self.image_processor)
            if all_images
            else (None, None)
        )
        all_videos = [
            video for sample in rows for video in sample.get("pixel_values_videos", [])
        ]
        video_patches, video_grids = (
            self.collate_images(all_videos, self.video_processor)
            if all_videos
            else (None, None)
        )

        # Each image is one vision entry; each video is one per temporal patch.
        total_images = (len(grids) if grids is not None else 0) + (
            int(video_grids[:, 0].sum()) if video_grids is not None else 0
        )
        if total_images > self.max_images_per_microbatch:
            raise ValueError(
                f"multimodal microbatch has {total_images} vision entries, exceeding "
                f"max_images_per_microbatch={self.max_images_per_microbatch}"
            )

        # Pad text.
        input_ids, labels, positions, padding_mask = self.collate_text(rows)
        model_kwargs = {
            "pixel_values": patches,
            "grid_thw": grids,
            "pixel_values_videos": video_patches,
            "grid_thw_videos": video_grids,
            "special_tokens": {
                f"{name}_id": getattr(self.tokenizer, f"{name}_id")
                for name in self.tokenizer.TOKEN_FIELDS
            },
        }
        if all_waveforms:
            waveforms, waveform_lengths, waveform_padded_lengths = self.collate_audio(
                all_waveforms
            )
            model_kwargs.update(
                {
                    "waveforms": waveforms,
                    "waveform_lengths": waveform_lengths,
                    "waveform_padded_lengths": waveform_padded_lengths,
                }
            )

        # Build multimodal RoPE positions.
        if self.build_mrope_positions and (
            grids is not None or video_grids is not None
        ):
            special_tokens = cast(dict[str, int], model_kwargs["special_tokens"])
            model_kwargs["mrope_positions"] = self._build_mrope_positions(
                input_ids,
                grids,
                video_grids,
                positions,
                image_token_id=special_tokens["image_id"],
                video_token_id=special_tokens["video_id"],
            )

        target_mask = labels != IGNORE_INDEX
        loss_token_counts, routing_token_counts = get_mtp_token_counts(
            target_mask=target_mask,
            positions=positions,
            padding_mask=padding_mask,
            num_mtp_layers=self._num_mtp_layers,
        )
        if self._num_mtp_layers == 0:
            loss_token_counts = loss_token_counts[0]
        return TokenizedTrainingMicrobatch(
            input=input_ids,
            labels=labels,
            positions=positions,
            padding_mask=padding_mask,
            loss_token_counts=loss_token_counts,
            routing_token_counts=routing_token_counts,
            model_kwargs=model_kwargs,
        )
