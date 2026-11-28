# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Token packing shared by supervised and RL microbatches."""

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from torchtitan.models.common.mtp import get_mtp_token_counts
from torchtitan.tools.utils import round_up


def append_padded_segment(
    parts: dict[str, list[np.ndarray]],
    fields: dict[str, np.ndarray],
    *,
    per_sample_pad_multiple: int,
    pad_values: dict[str, int | float | bool],
    padding_mask: np.ndarray | None,
) -> int:
    """Append one segment's fields, positions and padding mask; return its length.

    Callers shift labels, split documents, choose field-specific fill values,
    and pad row tails. ``fields`` contains equally long 1-D arrays, including
    ``input_ids``; ``parts`` also holds ``positions`` and ``padding_mask`` lists.
    Five tokens padded to eight have positions 0..7; the last three are masked,
    and existing padding stays masked. Keep tokens and padding separate until
    row assembly to avoid copying tokens twice.
    """
    num_tokens = len(fields["input_ids"])
    if any(len(values) != num_tokens for values in fields.values()):
        raise ValueError("Segment fields must share one length")
    padded_length = round_up(num_tokens, per_sample_pad_multiple)
    num_padding = padded_length - num_tokens
    for key, values in fields.items():
        parts[key].append(values)
        if num_padding:
            parts[key].append(np.full(num_padding, pad_values[key], dtype=values.dtype))
    parts["positions"].append(np.arange(padded_length, dtype=np.int64))
    mask = np.ones(padded_length, dtype=np.bool_)
    mask[:num_tokens] = False if padding_mask is None else padding_mask
    parts["padding_mask"].append(mask)
    return padded_length


def pack_microbatch_fields(
    parts: Mapping[str, Sequence[np.ndarray | torch.Tensor]],
    *,
    num_tokens: int,
    max_context_length: int,
    fill_values: Mapping[str, int | float | bool],
    dtypes: Mapping[str, torch.dtype],
    pin_memory: bool,
) -> dict[str, torch.Tensor]:
    """Concatenate each per-token field's segments into one ``num_tokens`` tensor, padding the
    tail.

    ``parts`` maps ``positions``, ``padding_mask`` and every key of ``fill_values`` to its
    segments, in microbatch order. The tail takes ``fill_values``, ``padding_mask`` True, and
    positions ``0, 1, ...`` modulo ``max_context_length``, one padding segment per
    ``max_context_length`` tokens. Each field is written once, into a buffer of its final
    size (page-locked with ``pin_memory``).

    Example (num_tokens 6, max_context_length 8, fill_values {"input_ids": 0, "labels": -100}):

        parts input_ids [[5, 6], [7]], labels [[6, -100], [8]], positions [[0, 1], [0]],
              padding_mask [[False, False], [False]]
        -> input_ids [5, 6, 7, 0, 0, 0], labels [6, -100, 8, -100, -100, -100],
           positions [0, 1, 0, 0, 1, 2], padding_mask [F, F, F, T, T, T]
    """
    num_real = sum(len(segment) for segment in parts["positions"])
    if num_real > num_tokens:
        raise ValueError(f"{num_real} tokens exceed the {num_tokens}-token microbatch")
    fields = {
        **{key: (fill_value, dtypes[key]) for key, fill_value in fill_values.items()},
        "positions": (0, torch.int64),
        "padding_mask": (True, torch.bool),
    }
    packed = {}
    for key, (fill_value, dtype) in fields.items():
        buffer = torch.full(
            (num_tokens,), fill_value, dtype=dtype, pin_memory=pin_memory
        )
        if num_real:
            torch.cat(
                [torch.as_tensor(segment, dtype=dtype) for segment in parts[key]],
                out=buffer[:num_real],
            )
        packed[key] = buffer
    if num_real < num_tokens:
        torch.arange(num_tokens - num_real, out=packed["positions"][num_real:])
        packed["positions"][num_real:].remainder_(max_context_length)
    return packed


def get_microbatch_token_counts(
    *,
    target_mask: torch.Tensor,
    positions: torch.Tensor,
    padding_mask: torch.Tensor,
    num_mtp_layers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A microbatch's ``(loss_token_counts, routing_token_counts)``: `get_mtp_token_counts`,
    with the loss count a scalar without MTP layers."""
    loss_token_counts, routing_token_counts = get_mtp_token_counts(
        target_mask=target_mask,
        positions=positions,
        padding_mask=padding_mask,
        num_mtp_layers=num_mtp_layers,
    )
    if num_mtp_layers == 0:
        loss_token_counts = loss_token_counts[0]
    return loss_token_counts, routing_token_counts
