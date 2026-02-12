# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Text processing utilities for Qwen3-VL datasets."""

import torch


def pad_text_batch(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    target_len: int,
    padding_idx: int = 0,
    ignore_idx: int = -100,
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


def pad_input_ids_and_labels_to_target_batch_size(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    target_batch_size: int,
    padding_idx: int = 0,
    ignore_idx: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad batch dimension to target size."""
    B, L = input_ids.shape
    if B >= target_batch_size:
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


def process_text_with_images(
    text: list[str],
    image_tokens: list[tuple[int, int, int]],
    tokenizer,
    special_tokens,
    add_eos: bool = True,
) -> str:
    """Process text by interleaving image tokens for Qwen3-VL.

    Args:
        text: List of text parts
        image_tokens: List of (total_tokens, width, height) for each image
        tokenizer: Tokenizer with special tokens
        special_tokens: Special token definitions
        add_eos: Whether to add EOS token

    Returns:
        Processed text with image tokens inserted
    """
    parts = []
    image_idx = 0

    for part in text:
        if part is None and image_idx < len(image_tokens):
            num_image_tokens, _, _ = image_tokens[image_idx]

            # Qwen3-VL uses <|vision_start|> and <|vision_end|> markers
            # Insert image tokens (placeholders) between these markers
            parts.extend(
                [
                    special_tokens.vision_start_token,
                    *([special_tokens.img_token] * num_image_tokens),
                    special_tokens.vision_end_token,
                ]
            )
            image_idx += 1
        else:
            parts.append(part)

    result = "".join(parts)
    return result.strip() + (tokenizer.eos_token if add_eos else "")
