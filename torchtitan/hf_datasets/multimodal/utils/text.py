# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Text processing utilities for multimodal datasets."""

import torch


def pad_seq_len(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    target_len: int,
    *,
    padding_idx: int,
    ignore_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad input_ids and labels to desired sequence length."""
    B, L = input_ids.shape

    if L < target_len:
        padding_length = target_len - L
        padding_input = torch.full(
            (B, padding_length), padding_idx, dtype=torch.long, device=input_ids.device
        )
        padding_labels = torch.full(
            (B, padding_length), ignore_idx, dtype=torch.long, device=labels.device
        )

        input_ids = torch.cat([input_ids, padding_input], dim=1)
        labels = torch.cat([labels, padding_labels], dim=1)

    elif L > target_len:
        input_ids = input_ids[:, :target_len]
        labels = labels[:, :target_len]

    return input_ids, labels


def pad_batch_dim(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    target_batch_size: int,
    *,
    padding_idx: int,
    ignore_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad batch dimension to target size."""
    B, L = input_ids.shape
    assert B <= target_batch_size, f"Batch size {B} exceeds target {target_batch_size}"
    if B == target_batch_size:
        return input_ids, labels

    padding_needed = target_batch_size - B
    padding_input = torch.full(
        (padding_needed, L), padding_idx, dtype=torch.long, device=input_ids.device
    )
    padding_labels = torch.full(
        (padding_needed, L), ignore_idx, dtype=torch.long, device=labels.device
    )

    input_ids = torch.cat([input_ids, padding_input], dim=0)
    labels = torch.cat([labels, padding_labels], dim=0)

    return input_ids, labels


def insert_vision_placeholders(
    input_parts: list[str | None],
    num_vision_tokens: list[int],
    *,
    vision_start_token: str,
    vision_token: str,
    vision_end_token: str,
    eos_token: str = "",
) -> str:
    """Insert vision placeholder token sequences into text.

    Args:
        input_parts: Mixed list of text strings and ``None`` entries.
            Each ``None`` marks where a vision region (image or video) should
            be inserted; text strings are kept as-is.  Produced by the dataset
            processor which sets ``texts[idx] = None`` for each image/video.
        num_vision_tokens: Number of vision tokens per ``None`` placeholder.
        vision_start_token: Token marking start of a vision region.
        vision_token: Repeated placeholder token (image or video).
        vision_end_token: Token marking end of a vision region.
        eos_token: Appended at the end if non-empty.

    Returns:
        Text with vision placeholders expanded.
    """
    output_parts: list[str] = []
    vision_idx = 0

    for part in input_parts:
        if part is None and vision_idx < len(num_vision_tokens):
            output_parts.extend(
                [
                    vision_start_token,
                    *([vision_token] * num_vision_tokens[vision_idx]),
                    vision_end_token,
                ]
            )
            vision_idx += 1
        else:
            output_parts.append(part)  # pyrefly: ignore [bad-argument-type]

    result = "".join(output_parts).strip()
    if eos_token and not result.endswith(eos_token):
        result += eos_token
    return result
