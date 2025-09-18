# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


def pad_text_batch(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    seq_len: int,
    padding_idx: int = 0,
    ignore_idx: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad input_ids and labels to desired sequence length.

    Args:
        input_ids: Tensor of shape (B, L)
        labels: Tensor of shape (B, L)
        seq_len: Desired sequence length
        padding_idx: Token ID to use for padding
        ignore_idx: Token ID to use for ignored positions in labels

    Returns:
        padded_input_ids: Tensor of shape (B, seq_len)
        padded_labels: Tensor of shape (B, seq_len)
    """
    B, L = input_ids.shape

    if L < seq_len:
        # Pad to desired length
        padding_length = seq_len - L
        padding_input = torch.full(
            (B, padding_length), padding_idx, dtype=torch.long, device=input_ids.device
        )
        padding_labels = torch.ones(
            (B, padding_length), dtype=torch.long, device=labels.device
        )

        input_ids = torch.cat([input_ids, padding_input], dim=1)
        labels = torch.cat([labels, padding_labels], dim=1)

    elif L > seq_len:
        # Truncate to desired length
        input_ids = input_ids[:, :seq_len]
        labels = labels[:, :seq_len]

    # Convert padding tokens to ignore_idx in labels
    labels[labels == padding_idx] = ignore_idx

    return input_ids, labels


def pad_input_ids_and_labels_to_target_batch_size(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    target_batch_size: int,
    padding_idx: int = 0,
    ignore_idx: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad batch dimension to target size.

    Args:
        input_ids: Tensor of shape (B, L)
        labels: Tensor of shape (B, L)
        target_batch_size: Desired batch size
        padding_idx: Token ID to use for padding
        ignore_idx: Token ID to use for ignored positions in labels

    Returns:
        padded_input_ids: Tensor of shape (target_batch_size, L)
        padded_labels: Tensor of shape (target_batch_size, L)
    """
    B, L = input_ids.shape
    if B >= target_batch_size:
        return input_ids, labels

    padding_needed = target_batch_size - B
    padding_input = torch.full(
        (padding_needed, L), padding_idx, dtype=torch.long, device=input_ids.device
    )
    padding_labels = torch.full(
        (padding_needed, L), padding_idx, dtype=torch.long, device=labels.device
    )

    input_ids = torch.cat([input_ids, padding_input], dim=0)
    labels = torch.cat([labels, padding_labels], dim=0)

    # Convert padding tokens to ignore_idx in labels
    labels[labels == padding_idx] = ignore_idx

    return input_ids, labels


def process_text_with_images(
    text: list[str],
    image_tokens: list[tuple[int, int, int]],  # [(total, width, height), ...]
    tokenizer,
    special_tokens,
    add_eos: bool = True,
) -> str:
    """Process text by interleaving image tokens efficiently.

    Args:
        text: Raw text string
        image_tokens: List of (total_tokens, width, height) for each image
        tokenizer: Tokenizer with special tokens
        add_eos: Whether to add EOS token

    Returns:
        Processed text with image tokens inserted

    Example:
        >>> text = ["<image>", "photo of a cat"]
        >>> image_tokens = [(16, 4, 4)]  # 4x4 grid = 16 tokens
        >>> result = process_text_with_images(text, image_tokens, tokenizer)
        >>> print(result)  # <|begin_of_image|><|image|>...<|end_of_image|> A photo...
    """
    parts = []  # Build parts list instead of string concat
    image_idx = 0

    for part in text:
        if part == special_tokens.img_token and image_idx < len(image_tokens):
            num_image_tokens, _, _ = image_tokens[image_idx]

            parts.extend(
                [
                    special_tokens.boi_token,
                    *([special_tokens.img_token] * num_image_tokens),
                    special_tokens.eoi_token,
                ]
            )
            image_idx += 1
        else:
            parts.append(part)

    # Join all parts with spaces and add EOS if needed
    result = "".join(parts)
    return result.strip() + (tokenizer.eos_token if add_eos else "")
