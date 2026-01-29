# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Loss masking for Harmony chat format.

The Harmony format uses special tokens to delimit messages:
  <|start|>role<|channel|>channel_name<|message|>content<|end|>

For assistant turns, content typically ends with <|return|> (final response),
but can also end with <|end|> (analysis/commentary continuation) or <|call|>
(tool invocations).

This module provides functions to create loss masks that compute loss on
assistant turns AFTER the last user message. This ensures train/inference
consistency:
  - At inference, prior turn outputs were generated with their own context
  - Training only on the "current turn" (after last user) matches this

The mask identifies the last user message and only marks assistant tokens
that appear after it. This includes:
  1. Analysis/reasoning tokens (model generates these)
  2. Tool call tokens (model generates these)
  3. Final response tokens (model generates these)

Prior turn assistant outputs (before last user) have mask=0 since they
serve as context, not training targets.

Token IDs for o200k_harmony:
  - <|start|>: 200006
  - <|end|>: 200007
  - <|message|>: 200008
  - <|return|>: 200002
  - <|channel|>: 200005
  - <|call|>: 200012
  - <|constrain|>: 200003
  - assistant (literal text): 173781
  - user (literal text): 1428
"""

import torch
from typing import Optional

# Harmony special token IDs (o200k_harmony tokenizer)
# Reference: /mnt/git/harmony/src/tiktoken_ext/public_encodings.rs
HARMONY_START = 200006      # <|start|>
HARMONY_END = 200007        # <|end|>
HARMONY_MESSAGE = 200008    # <|message|>
HARMONY_RETURN = 200002     # <|return|>
HARMONY_CHANNEL = 200005    # <|channel|>
HARMONY_CALL = 200012       # <|call|>
HARMONY_ASSISTANT = 173781  # "assistant" token
HARMONY_USER = 1428         # "user" token

# Exported for callers but not used in masking logic directly.
# <|constrain|> appears in assistant headers (e.g., <|constrain|>json for tool calls)
# and is automatically included in the loss mask as part of the header tokens.
HARMONY_CONSTRAIN = 200003  # <|constrain|> - for structured output format


def create_loss_mask(
    input_ids: torch.Tensor,
    start_token: int = HARMONY_START,
    message_token: int = HARMONY_MESSAGE,
    end_token: int = HARMONY_END,
    return_token: int = HARMONY_RETURN,
    call_token: int = HARMONY_CALL,
    assistant_token: int = HARMONY_ASSISTANT,
    user_token: int = HARMONY_USER,
    padding_token: Optional[int] = None,
) -> torch.Tensor:
    """
    Create a loss mask for assistant tokens AFTER the last user message.

    This ensures train/inference consistency: at inference, prior turn outputs
    were generated in a different context. We only train on the "current turn"
    (assistant responses after the last user message).

    The mask identifies:
    1. The last user message position (<|start|>user<|message|>...)
    2. Assistant turns that start AFTER this position

    Assistant turns after the last user are marked with 1, including:
    - Header tokens: <|start|>, assistant, <|channel|>, channel_name, etc.
    - Content tokens: everything between <|message|> and end token
    - End tokens: <|return|> (final), <|end|> (continuation), or <|call|> (tool)

    Prior turn assistant outputs (before last user) have mask=0.

    Args:
        input_ids: Tensor of token IDs, shape (seq_len,) or (batch, seq_len)
        start_token: Token ID for <|start|>
        message_token: Token ID for <|message|>
        end_token: Token ID for <|end|>
        return_token: Token ID for <|return|>
        call_token: Token ID for <|call|>
        assistant_token: Token ID for "assistant"
        user_token: Token ID for "user"
        padding_token: Optional padding token ID to explicitly mask

    Returns:
        mask: Tensor of same shape as input_ids, with 1 for assistant tokens
              after the last user message and 0 elsewhere
    """
    # Handle batched input
    if input_ids.dim() == 1:
        return _create_loss_mask_1d(
            input_ids, start_token, message_token, end_token,
            return_token, call_token, assistant_token, user_token, padding_token
        )
    elif input_ids.dim() == 2:
        masks = []
        for i in range(input_ids.shape[0]):
            mask = _create_loss_mask_1d(
                input_ids[i], start_token, message_token, end_token,
                return_token, call_token, assistant_token, user_token, padding_token
            )
            masks.append(mask)
        return torch.stack(masks)
    else:
        raise ValueError(f"Expected 1D or 2D input, got {input_ids.dim()}D")


def _create_loss_mask_1d(
    input_ids: torch.Tensor,
    start_token: int,
    message_token: int,
    end_token: int,
    return_token: int,
    call_token: int,
    assistant_token: int,
    user_token: int,
    padding_token: Optional[int],
) -> torch.Tensor:
    """Create loss mask for a single sequence.

    Only masks assistant tokens that appear AFTER the last user message.
    This ensures train/inference consistency.

    Uses vectorized tensor operations for efficiency with long sequences (100K+ tokens).
    """
    seq_len = input_ids.shape[0]
    device = input_ids.device

    if seq_len == 0:
        return torch.zeros(0, dtype=torch.bool, device=device)

    # Use bool dtype for 4x memory savings vs float32 (1 byte vs 4 bytes)
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

    # Vectorized: find all positions where token equals each special token
    is_start = input_ids == start_token
    is_user = input_ids == user_token
    is_assistant = input_ids == assistant_token
    is_message = input_ids == message_token
    is_end = input_ids == end_token
    is_return = input_ids == return_token
    is_call = input_ids == call_token

    # Find last user position: pattern <|start|> user
    # Check for start[i] and user[i+1]
    if seq_len > 1:
        user_turn_starts = is_start[:-1] & is_user[1:]
        user_positions = user_turn_starts.nonzero(as_tuple=True)[0]
        if len(user_positions) > 0:
            last_user_pos = user_positions[-1].item()
        else:
            last_user_pos = 0
    else:
        last_user_pos = 0

    # Find assistant turn starts: pattern <|start|> assistant
    if seq_len > 1:
        assistant_turn_starts = is_start[:-1] & is_assistant[1:]
        assistant_positions = assistant_turn_starts.nonzero(as_tuple=True)[0]
    else:
        assistant_positions = torch.tensor([], dtype=torch.long, device=device)

    # End tokens for assistant turns
    is_end_token = is_return | is_end | is_call

    # For each assistant turn that starts AFTER last_user_pos, mark tokens
    # We need to find the extent of each turn (from start to end token or next start)
    for start_pos in assistant_positions.tolist():
        # Only process assistant turns after the last user message
        if start_pos <= last_user_pos:
            continue

        # Find <|message|> token after this start
        # Search from start_pos + 2 (skip <|start|> and assistant)
        message_search_start = start_pos + 2
        if message_search_start >= seq_len:
            # Truncated turn, mark what we have
            mask[start_pos:seq_len] = True
            continue

        # Find first <|message|> after start
        message_positions_after = is_message[message_search_start:].nonzero(as_tuple=True)[0]
        if len(message_positions_after) == 0:
            # No <|message|> found, mark rest of sequence
            mask[start_pos:seq_len] = True
            continue

        message_pos = message_search_start + message_positions_after[0].item()

        # Find end of this assistant turn: first end token or next <|start|> after message
        content_start = message_pos + 1
        if content_start >= seq_len:
            # Content starts at end of sequence
            mask[start_pos:seq_len] = True
            continue

        # Look for end token (return, end, call) or next start
        end_or_start = is_end_token[content_start:] | is_start[content_start:]
        end_positions = end_or_start.nonzero(as_tuple=True)[0]

        if len(end_positions) == 0:
            # No end token found, mark to end of sequence (truncated)
            mask[start_pos:seq_len] = True
        else:
            end_pos = content_start + end_positions[0].item()
            # Check if it's an end token or a new start
            if is_end_token[end_pos]:
                # Include the end token
                mask[start_pos:end_pos + 1] = True
            else:
                # Hit a new <|start|>, don't include it
                mask[start_pos:end_pos] = True

    # Explicitly mask padding tokens if specified
    if padding_token is not None:
        padding_mask = input_ids == padding_token
        mask[padding_mask] = False

    return mask


def apply_loss_mask(
    loss: torch.Tensor,
    mask: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Apply a loss mask to per-token losses.

    Args:
        loss: Per-token loss tensor, shape (batch, seq_len) or (seq_len,)
        mask: Loss mask tensor, same shape as loss, with 1 for tokens to
              include in loss and 0 for tokens to ignore
        reduction: How to reduce the masked loss:
            - "mean": Mean over non-masked tokens
            - "sum": Sum over non-masked tokens
            - "none": Return masked per-token losses

    Returns:
        Reduced loss according to reduction parameter
    """
    # Ensure mask is same dtype as loss for multiplication
    mask = mask.to(loss.dtype)

    # Apply mask
    masked_loss = loss * mask

    if reduction == "none":
        return masked_loss
    elif reduction == "sum":
        return masked_loss.sum()
    elif reduction == "mean":
        # Mean over non-masked tokens only
        num_tokens = mask.sum()
        if num_tokens > 0:
            return masked_loss.sum() / num_tokens
        else:
            # No tokens to compute loss on - return 0
            return masked_loss.sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def get_mask_statistics(mask: torch.Tensor) -> dict:
    """
    Get statistics about a loss mask for debugging/logging.

    Args:
        mask: Loss mask tensor

    Returns:
        Dictionary with mask statistics
    """
    total = mask.numel()
    masked_in = mask.sum().item()
    masked_out = total - masked_in

    return {
        "total_tokens": total,
        "assistant_tokens": int(masked_in),
        "non_assistant_tokens": int(masked_out),
        "assistant_ratio": masked_in / total if total > 0 else 0.0,
    }
