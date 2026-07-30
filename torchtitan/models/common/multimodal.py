# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-agnostic vision<->text fusion for VLMs.

The decoder embeds the full token sequence; the placeholder tokens
get a throwaway text embedding that ``scatter_vision_embeds``
overwrites with the vision encoder's per-item features at the positions
``get_vision_positions`` locates.
"""

import torch


def get_vision_positions(
    tokens: torch.Tensor,
    num_vision_tokens_per_item: torch.Tensor,
    placeholder_id: int,
) -> list[tuple[int, int, int, int]]:
    """Locate each visual item's placeholder run in the token sequence.

    Args:
        tokens: (bsz, seq_len) token IDs.
        num_vision_tokens_per_item: (num_items,) valid token count per visual item, in
            the order the items appear in ``tokens``.
        placeholder_id: token id whose contiguous runs mark vision spans.

    Returns:
        ``(item_idx, sample_idx, vision_start, n_tokens)`` per item, where
        ``vision_start`` is the position of the run's first placeholder token
        within its sample.

    Raises:
        ValueError: if the number of placeholder runs does not equal the number
            of visual items, or a run's length does not match the item's token
            count. Either mismatch means the text and vision streams are
            misaligned; scattering anyway would silently corrupt the embeddings,
            so fail loudly with the offending counts.
    """
    vision_mask = tokens == placeholder_id  # (bsz, seq_len)
    # Shift within each row (row boundaries padded False) so a placeholder
    # ending one sample and starting the next are NOT merged into one run across
    # the flattened batch boundary.
    prev_mask = torch.zeros_like(vision_mask)
    prev_mask[:, 1:] = vision_mask[:, :-1]
    next_mask = torch.zeros_like(vision_mask)
    next_mask[:, :-1] = vision_mask[:, 1:]
    flat_mask = vision_mask.view(-1)
    region_starts = torch.where(flat_mask & ~prev_mask.view(-1))[0]
    region_ends = torch.where(flat_mask & ~next_mask.view(-1))[0]
    seq_len = tokens.shape[1]

    num_items = int(num_vision_tokens_per_item.shape[0])
    num_runs = int(region_starts.shape[0])
    if num_runs != num_items:
        raise ValueError(
            f"Multimodal misalignment: found {num_runs} contiguous run(s) of "
            f"placeholder id {placeholder_id} in the token sequence but received "
            f"{num_items} visual item(s). Each visual item must correspond to "
            f"exactly one placeholder run."
        )

    run_lengths = (region_ends - region_starts + 1).tolist()
    positions: list[tuple[int, int, int, int]] = []
    for i in range(num_items):
        start = int(region_starts[i].item())
        n_tokens = int(num_vision_tokens_per_item[i].item())
        if run_lengths[i] != n_tokens:
            raise ValueError(
                f"Multimodal misalignment: placeholder run {i} spans "
                f"{run_lengths[i]} token(s) but visual item {i} produced "
                f"{n_tokens} embedding(s). The placeholder count in the prompt "
                f"must match the vision token count for that item."
            )
        positions.append((i, start // seq_len, start % seq_len, n_tokens))
    return positions


def scatter_vision_embeds(
    inputs_embeds: torch.Tensor,
    *,
    vision_embeds: torch.Tensor,
    vision_positions: list[tuple[int, int, int, int]],
) -> torch.Tensor:
    """Copy padded vision features into the text sequence at placeholder runs.

    Args:
        inputs_embeds: (batch, seq_len, dim) text embeddings, modified in place.
        vision_embeds: (num_items, max_tokens, dim) padded vision features.
        vision_positions: from ``get_vision_positions``.
    """
    for item_idx, sample_idx, vision_start, n_tokens in vision_positions:
        inputs_embeds[
            sample_idx, vision_start : vision_start + n_tokens, :
        ] = vision_embeds[item_idx, :n_tokens, :].to(inputs_embeds.dtype)
    return inputs_embeds
