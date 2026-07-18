# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen multimodal Grain processors and packing recipe."""

from abc import abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any

import grain.python as grain
import numpy as np
import torch

from torchtitan.components.data.dataset import (
    BuildOptions,
    DataRuntime,
    DatasetConfig,
    SampleProcessor,
)
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.tools.logging import logger

from .utils.image import calculate_vision_tokens, process_image
from .utils.text import insert_vision_placeholders


QwenTrainingRow = dict[str, Any]


def _process_qwen_sample(
    *,
    texts: list[str | None],
    images: list[Any],
    tokenizer: MultiModalTokenizer,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
    min_pixels: int,
    max_pixels: int,
    image_mean: tuple[float, ...],
    image_std: tuple[float, ...],
) -> QwenTrainingRow | None:
    """Convert an ordered parallel text/image representation to a Qwen row."""
    if not texts or len(texts) != len(images):
        return None

    processed_images: list[torch.Tensor] = []
    num_image_tokens: list[int] = []
    processed_texts = list(texts)
    expected_images = sum(image is not None for image in images)

    for index, image in enumerate(images):
        if image is None:
            continue
        processed_image = process_image(
            image,
            patch_size=patch_size,
            merge_size=spatial_merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            image_mean=image_mean,
            image_std=image_std,
        )
        if processed_image is None:
            logger.warning("Cannot process all images for sample. Dropping")
            return None
        num_tokens, _, _ = calculate_vision_tokens(
            num_frames=1,
            height=processed_image.shape[1],
            width=processed_image.shape[2],
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            temporal_patch_size=temporal_patch_size,
        )
        processed_images.append(processed_image)
        num_image_tokens.append(num_tokens)
        processed_texts[index] = None

    if len(processed_images) != expected_images:
        logger.warning("Cannot process all images for sample. Dropping")
        return None

    processed_text = insert_vision_placeholders(
        processed_texts,
        num_image_tokens,
        vision_start_token=tokenizer.vision_start_token,
        vision_token=tokenizer.image_token,
        vision_end_token=tokenizer.vision_end_token,
        eos_token=tokenizer.eos_token,
    )
    token_ids = torch.as_tensor(tokenizer.encode(processed_text), dtype=torch.long)
    labels = token_ids.clone()
    special_token_ids = torch.as_tensor(
        [
            tokenizer.vision_start_id,
            tokenizer.vision_end_id,
            tokenizer.image_id,
            tokenizer.video_id,
        ],
        dtype=token_ids.dtype,
    )
    labels = torch.where(
        torch.isin(labels, special_token_ids),
        torch.as_tensor(IGNORE_INDEX, dtype=labels.dtype),
        labels,
    )
    return {
        "input_ids": token_ids,
        "labels": labels,
        "positions": torch.arange(token_ids.shape[0], dtype=torch.long),
        "pixel_values": processed_images,
    }


class _QwenProcessor(SampleProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        patch_size: int = 16
        temporal_patch_size: int = 2
        spatial_merge_size: int = 2
        min_pixels: int = 65_536
        max_pixels: int = 16_777_216
        image_mean: tuple[float, ...] = (0.5, 0.5, 0.5)
        image_std: tuple[float, ...] = (0.5, 0.5, 0.5)

    def __init__(self, config: Config, *, runtime: DataRuntime) -> None:
        self._config = config
        self._tokenizer = runtime.tokenizer

    @abstractmethod
    def _ordered_parts(
        self, sample: dict[str, Any]
    ) -> tuple[list[str | None], list[Any]]:
        ...

    def __call__(
        self,
        sample: dict[str, Any],
        rng: np.random.Generator,
    ) -> QwenTrainingRow | None:
        del rng
        texts, images = self._ordered_parts(sample)
        return _process_qwen_sample(
            texts=texts,
            images=images,
            tokenizer=self._tokenizer,
            patch_size=self._config.patch_size,
            temporal_patch_size=self._config.temporal_patch_size,
            spatial_merge_size=self._config.spatial_merge_size,
            min_pixels=self._config.min_pixels,
            max_pixels=self._config.max_pixels,
            image_mean=self._config.image_mean,
            image_std=self._config.image_std,
        )


class QwenObelicsProcessor(_QwenProcessor):
    """Processes OBELICS' ordered ``texts`` and ``images`` columns."""

    @dataclass(kw_only=True, slots=True)
    class Config(_QwenProcessor.Config):
        pass

    def _ordered_parts(
        self, sample: dict[str, Any]
    ) -> tuple[list[str | None], list[Any]]:
        return list(sample.get("texts", [])), list(sample.get("images", []))


class QwenCC12MProcessor(_QwenProcessor):
    """Processes CC12M-WDS' text/image pair columns."""

    @dataclass(kw_only=True, slots=True)
    class Config(_QwenProcessor.Config):
        text_field: str = "txt"
        image_field: str = "jpg"

    def _ordered_parts(
        self, sample: dict[str, Any]
    ) -> tuple[list[str | None], list[Any]]:
        return [None, sample.get(self._config.text_field, "")], [
            sample.get(self._config.image_field),
            None,
        ]


def _is_qwen_row(row: QwenTrainingRow | None) -> bool:
    return row is not None


def _num_image_patches(
    image: torch.Tensor,
    *,
    patch_size: int,
    temporal_patch_size: int,
) -> int:
    frames, height, width, _ = image.shape
    return (
        (frames + temporal_patch_size - 1)
        // temporal_patch_size
        * (height // patch_size)
        * (width // patch_size)
    )


def _vision_admission(
    row: QwenTrainingRow, *, patch_size: int, temporal_patch_size: int
) -> tuple[int, int]:
    num_images = len(row.get("pixel_values", ()))
    num_patches = sum(
        _num_image_patches(
            image,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
        )
        for image in row.get("pixel_values", ())
    )
    for video in row.get("pixel_values_videos", ()):
        num_images += (video.shape[0] + temporal_patch_size - 1) // temporal_patch_size
        num_patches += _num_image_patches(
            video,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
        )
    return num_images, num_patches


def _merge_qwen_rows(rows: list[QwenTrainingRow]) -> QwenTrainingRow:
    labels = [row["labels"].clone() for row in rows]
    for later_labels in labels[1:]:
        later_labels[0] = IGNORE_INDEX
    return {
        "input_ids": torch.cat([row["input_ids"] for row in rows]),
        "labels": torch.cat(labels),
        "positions": torch.cat([row["positions"] for row in rows]),
        "pixel_values": [
            image for row in rows for image in row.get("pixel_values", ())
        ],
        "pixel_values_videos": [
            video for row in rows for video in row.get("pixel_values_videos", ())
        ],
    }


class _QwenMultimodalPackingDataset(grain.IterDataset[QwenTrainingRow]):
    def __init__(
        self,
        parent: grain.IterDataset[QwenTrainingRow],
        *,
        max_seq_len: int,
        local_batch_size: int,
        max_images_per_batch: int,
        max_patches_per_batch: int,
        patch_size: int,
        temporal_patch_size: int,
    ) -> None:
        super().__init__(parent)
        self._max_seq_len = max_seq_len
        self._local_batch_size = local_batch_size
        self._max_images_per_batch = max_images_per_batch
        self._max_patches_per_batch = max_patches_per_batch
        self._patch_size = patch_size
        self._temporal_patch_size = temporal_patch_size

    def __iter__(self) -> "_QwenMultimodalPackingIterator":
        return _QwenMultimodalPackingIterator(
            self._parent.__iter__(),
            max_seq_len=self._max_seq_len,
            local_batch_size=self._local_batch_size,
            max_images_per_batch=self._max_images_per_batch,
            max_patches_per_batch=self._max_patches_per_batch,
            patch_size=self._patch_size,
            temporal_patch_size=self._temporal_patch_size,
        )


class _QwenMultimodalPackingIterator(grain.DatasetIterator[QwenTrainingRow]):
    def __init__(
        self,
        parent: grain.DatasetIterator[QwenTrainingRow],
        *,
        max_seq_len: int,
        local_batch_size: int,
        max_images_per_batch: int,
        max_patches_per_batch: int,
        patch_size: int,
        temporal_patch_size: int,
    ) -> None:
        super().__init__(parent)
        self._max_seq_len = max_seq_len
        self._local_batch_size = local_batch_size
        self._max_images_per_batch = max_images_per_batch
        self._max_patches_per_batch = max_patches_per_batch
        self._patch_size = patch_size
        self._temporal_patch_size = temporal_patch_size
        self._ready_rows: deque[QwenTrainingRow] = deque()
        self._lookahead: QwenTrainingRow | None = None
        self._exhausted = False

    def _next_candidate(self) -> QwenTrainingRow | None:
        if self._lookahead is not None:
            row = self._lookahead
            self._lookahead = None
            return row
        if self._exhausted:
            return None
        while True:
            try:
                row = next(self._parent)
            except StopIteration:
                self._exhausted = True
                return None
            if _is_qwen_row(row):
                return row

    def _fill_ready_rows(self) -> None:
        bins: list[list[QwenTrainingRow]] = []
        image_count = 0
        patch_count = 0
        budget_blocked = False

        while True:
            row = self._next_candidate()
            if row is None:
                break

            row_length = int(row["input_ids"].shape[0])
            if row_length > self._max_seq_len:
                logger.warning(
                    "Dropping Qwen sample with length %d > max_seq_len %d",
                    row_length,
                    self._max_seq_len,
                )
                continue

            row_images, row_patches = _vision_admission(
                row,
                patch_size=self._patch_size,
                temporal_patch_size=self._temporal_patch_size,
            )
            if (
                row_images > self._max_images_per_batch
                or row_patches > self._max_patches_per_batch
            ):
                logger.warning(
                    "Dropping Qwen sample exceeding vision admission limits: "
                    "images=%d patches=%d",
                    row_images,
                    row_patches,
                )
                continue
            if (
                image_count + row_images > self._max_images_per_batch
                or patch_count + row_patches > self._max_patches_per_batch
            ):
                self._lookahead = row
                budget_blocked = True
                break

            placed = False
            for bin_rows in bins:
                bin_length = sum(
                    int(candidate["input_ids"].shape[0]) for candidate in bin_rows
                )
                if bin_length + row_length <= self._max_seq_len:
                    bin_rows.append(row)
                    placed = True
                    break
            if not placed and len(bins) < self._local_batch_size:
                bins.append([row])
            elif not placed:
                self._lookahead = row
                break
            image_count += row_images
            patch_count += row_patches

        if not bins:
            return

        for bin_rows in bins:
            self._ready_rows.append(_merge_qwen_rows(bin_rows))

        # Admission limits apply to the fixed loader batch. Empty rows retain
        # that batch's shape when a vision budget closes it early.
        if budget_blocked:
            empty = {
                "input_ids": torch.empty(0, dtype=torch.long),
                "labels": torch.empty(0, dtype=torch.long),
                "positions": torch.empty(0, dtype=torch.long),
                "pixel_values": [],
                "pixel_values_videos": [],
            }
            while len(self._ready_rows) < self._local_batch_size:
                self._ready_rows.append(empty)

    def __next__(self) -> QwenTrainingRow:
        if not self._ready_rows:
            self._fill_ready_rows()
        if not self._ready_rows:
            raise StopIteration
        return self._ready_rows.popleft()

    def get_state(self) -> dict[str, Any]:
        return {
            "parent": self._parent.get_state(),
            "ready_rows": list(self._ready_rows),
            "lookahead": self._lookahead,
            "exhausted": self._exhausted,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._parent.set_state(state["parent"])
        self._ready_rows = deque(state["ready_rows"])
        self._lookahead = state["lookahead"]
        self._exhausted = state["exhausted"]


@dataclass(frozen=True, kw_only=True, slots=True)
class QwenMultimodalPackingConfig:
    """Packs Qwen rows while enforcing fixed-batch vision admission limits."""

    dataset: DatasetConfig
    max_images_per_batch: int
    max_patches_per_batch: int
    patch_size: int = 16
    temporal_patch_size: int = 2

    def build(
        self,
        *,
        runtime: DataRuntime,
        options: BuildOptions,
    ) -> grain.IterDataset[QwenTrainingRow]:
        if not options.repeat and options.dp_world_size > 1:
            raise ValueError(
                "finite packed datasets are not supported with data parallelism"
            )
        parent = self.dataset.build(runtime=runtime, options=options)
        if isinstance(parent, grain.MapDataset):
            parent = parent.to_iter_dataset(read_options=runtime.read_options)
        if not isinstance(parent, grain.IterDataset):
            raise TypeError("Qwen multimodal packing requires a Grain dataset")
        # TODO(data-global-pack-plan): Plan packed rows before effective-DP sharding
        # when SFT/pretraining measurements justify shared length metadata and a cached plan.
        return _QwenMultimodalPackingDataset(
            parent,
            max_seq_len=runtime.seq_len,
            local_batch_size=runtime.local_batch_size,
            max_images_per_batch=self.max_images_per_batch,
            max_patches_per_batch=self.max_patches_per_batch,
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
        )
