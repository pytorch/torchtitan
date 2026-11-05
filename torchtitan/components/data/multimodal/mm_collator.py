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
from torchtitan.components.data.types import (
    DatasetBuildContext,
    TokenizedTrainingMicrobatch,
)
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.models.deepseek_v3.mtp import get_mtp_token_counts


class MultiModalCollator(Collator):
    """Prepare text and optional image, video, or audio model inputs.

    Images and videos arrive as uint8 ``(T, H, W, C)`` items resized by the
    sample processor; the trainer's ``ImageProcessor`` / ``VideoProcessor``
    normalizes and patches them here.
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

        Returns ``(num_tokens, 3)`` with temporal/height/width coordinates in
        the final dimension. Runs here on CPU data workers, off the GPU path.

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

        tokens = tokens.unsqueeze(0)
        positions = positions.unsqueeze(0) if positions is not None else None
        batch_size, seq_len = tokens.shape
        mrope_positions = torch.zeros(
            batch_size, seq_len, 3, dtype=tokens.dtype, device=tokens.device
        )

        if positions is not None:
            # Every document starts at 0. A decrease check misses 0 -> 0
            # boundaries after padding or a single-token document.
            resets = positions[:, 1:] == 0  # (batch, seq_len-1)
        # First token of each consecutive vision region (image or video).
        vision_mask = (tokens == image_token_id) | (tokens == video_token_id)
        prev_vision = torch.cat(
            [torch.zeros_like(vision_mask[:, :1]), vision_mask[:, :-1]], dim=1
        )
        batch_vision_starts = vision_mask & ~prev_vision  # (batch, seq_len)
        grid_cache: dict[tuple[int, int, int], torch.Tensor] = {}

        image_index, video_index = 0, 0
        # With sample packing, each sample may contain multiple documents.
        for sample_i in range(batch_size):
            llm_pos_ids_list: list[torch.Tensor] = []

            if positions is not None:
                # pyrefly: ignore [unbound-name]
                reset_indices = torch.where(resets[sample_i])[0] + 1
                doc_starts = [0] + reset_indices.tolist()
                doc_ranges = [
                    (
                        doc_starts[d],
                        doc_starts[d + 1] if d + 1 < len(doc_starts) else seq_len,
                    )
                    for d in range(len(doc_starts))
                ]
            else:
                doc_ranges = [(0, seq_len)]

            sample_tokens = tokens[sample_i]
            sample_vision_starts = torch.where(batch_vision_starts[sample_i])[
                0
            ].tolist()
            vision_start_index = 0

            for doc_start, doc_end in doc_ranges:
                doc_pos_ids_list: list[torch.Tensor] = []

                doc_vision_starts: list[int] = []
                while (
                    vision_start_index < len(sample_vision_starts)
                    and sample_vision_starts[vision_start_index] < doc_end
                ):
                    doc_vision_starts.append(sample_vision_starts[vision_start_index])
                    vision_start_index += 1

                pair_cursor = doc_start
                for vision_start in doc_vision_starts:
                    if sample_tokens[vision_start] == image_token_id:
                        # pyrefly: ignore [unsupported-operation]
                        t, h, w = grid_thw[image_index]
                        image_index += 1
                        processor = self.image_processor
                    else:
                        # pyrefly: ignore [unsupported-operation]
                        t, h, w = grid_thw_videos[video_index]
                        video_index += 1
                        processor = self.video_processor
                    assert processor is not None
                    spatial_merge_size = processor.config.spatial_merge_size

                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        int(t.item()),
                        int(h.item()) // spatial_merge_size,
                        int(w.item()) // spatial_merge_size,
                    )
                    text_len = vision_start - pair_cursor

                    pos_id_offset = (
                        doc_pos_ids_list[-1].max() + 1
                        if len(doc_pos_ids_list) > 0
                        else 0
                    )
                    # [text tokens] — sequential positions, identical on all 3 axes.
                    doc_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + pos_id_offset
                    )
                    # [vision tokens] — 3D grid positions (T, H, W).
                    grid_key = (llm_grid_t, llm_grid_h, llm_grid_w)
                    if grid_key not in grid_cache:
                        hw = llm_grid_h * llm_grid_w
                        t_index = (
                            torch.arange(llm_grid_t)
                            .view(-1, 1)
                            .expand(-1, hw)
                            .flatten()
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
                    doc_pos_ids_list.append(
                        grid_cache[grid_key] + text_len + pos_id_offset
                    )
                    pair_cursor = vision_start + llm_grid_t * llm_grid_h * llm_grid_w

                # Trailing [text tokens] after the last text/vision pair.
                if pair_cursor < doc_end:
                    pos_id_offset = (
                        doc_pos_ids_list[-1].max() + 1
                        if len(doc_pos_ids_list) > 0
                        else 0
                    )
                    text_len = doc_end - pair_cursor
                    doc_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + pos_id_offset
                    )

                llm_pos_ids_list.extend(doc_pos_ids_list)

            # llm_pos_ids_list is (3, segment_len); concat -> (3, seq), then transpose
            mrope_positions[sample_i] = torch.cat(llm_pos_ids_list, dim=1).T

        return mrope_positions.squeeze(0)

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

        # Collate image and video patches.
        all_images = [
            img
            for sample in rows
            if "pixel_values" in sample
            for img in sample["pixel_values"]
        ]
        # The sample processor requires the trainer's processor before it
        # emits media, so it is set whenever images or videos are present.
        patches, grids = None, None
        if all_images:
            assert self.image_processor is not None
            patches, grids = self.image_processor.normalize_and_patchify(all_images)

        all_videos = [
            vid
            for sample in rows
            if "pixel_values_videos" in sample
            for vid in sample["pixel_values_videos"]
        ]
        video_patches, video_grids = None, None
        if all_videos:
            assert self.video_processor is not None
            video_patches, video_grids = self.video_processor.normalize_and_patchify(
                all_videos
            )

        # Each image is one vision entry; each video is one per temporal patch.
        total_images = len(all_images) + (
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
