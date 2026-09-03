# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-agnostic vision<->text fusion for VLMs.

``get_vision_positions`` and ``scatter_vision_embeds`` support span-based
fusion over a full token sequence. ``build_vision_bank_indices`` and
``gather_vision_embeds`` support gather-based fusion by carrying an absolute
packed-bank row for every placeholder token.
"""

import contextlib

import spmd_types as spmd
import torch

from torchtitan.distributed.spmd_types import spmd_mesh_size
from torchtitan.distributed.utils import get_spmd_backend


def multimodal_context() -> contextlib.AbstractContextManager[None]:
    """Use a DP-local mesh while preparing multimodal inputs.

    Under ``spmd_types`` the vision encoder and the vision->text scatter run
    per-DP-rank on that rank's own images: the pixel tensors are DP-local
    (``V@DP``), so the region must execute with DP treated as a local axis.
    After the scatter the tensor is token-aligned again and global DP batch
    sharding resumes. A no-op outside ``spmd_types`` (or when DP is size 1).
    """
    if get_spmd_backend() == "spmd_types" and spmd_mesh_size("dp") > 1:
        return spmd.set_current_mesh(local_axes=("dp",))
    return contextlib.nullcontext()


def get_vision_positions(
    tokens: torch.Tensor,
    num_vision_tokens_per_item: torch.Tensor,
    placeholder_id: int,
) -> list[tuple[int, int, int]]:
    """Locate each visual item's placeholder run in the token sequence.

    Args:
        tokens: ``(T,)`` token IDs.
        num_vision_tokens_per_item: (num_items,) valid token count per visual item, in
            the order the items appear in ``tokens``.
        placeholder_id: token id whose contiguous runs mark vision spans.

    Returns:
        ``(item_idx, vision_start, n_tokens)`` per item.

    Raises:
        ValueError: if the number of placeholder runs does not equal the number
            of visual items, or a run's length does not match the item's token
            count. Either mismatch means the text and vision streams are
            misaligned; scattering anyway would silently corrupt the embeddings,
            so fail loudly with the offending counts.
    """
    vision_mask = tokens == placeholder_id
    prev_mask = torch.zeros_like(vision_mask)
    prev_mask[1:] = vision_mask[:-1]
    next_mask = torch.zeros_like(vision_mask)
    next_mask[:-1] = vision_mask[1:]
    region_starts = torch.where(vision_mask & ~prev_mask)[0]
    region_ends = torch.where(vision_mask & ~next_mask)[0]

    num_items = int(num_vision_tokens_per_item.shape[0])
    num_runs = int(region_starts.shape[0])
    if num_runs != num_items:
        raise ValueError(
            f"Multimodal misalignment: found {num_runs} contiguous run(s) of "
            f"placeholder id {placeholder_id} in the token sequence but received "
            f"{num_items} visual item(s). Each visual item must correspond to "
            f"exactly one placeholder run."
        )

    # Convert each metadata tensor once. Per-item ``.item()`` calls would
    # synchronize CUDA once per scalar.
    region_starts_list = region_starts.tolist()
    run_lengths = (region_ends - region_starts + 1).tolist()
    num_vision_tokens_per_item_list = num_vision_tokens_per_item.tolist()
    positions: list[tuple[int, int, int]] = []
    for i in range(num_items):
        start = int(region_starts_list[i])
        n_tokens = int(num_vision_tokens_per_item_list[i])
        if run_lengths[i] != n_tokens:
            raise ValueError(
                f"Multimodal misalignment: placeholder run {i} spans "
                f"{run_lengths[i]} token(s) but visual item {i} produced "
                f"{n_tokens} embedding(s). The placeholder count in the prompt "
                f"must match the vision token count for that item."
            )
        positions.append((i, start, n_tokens))
    return positions


def build_vision_bank_indices(
    tokens_T: torch.Tensor,
    *,
    placeholder_id: int,
) -> torch.Tensor:
    """Map vision placeholder tokens to absolute packed-bank rows."""
    vision_mask_T = tokens_T == placeholder_id
    vision_bank_indices_T = torch.cumsum(vision_mask_T.to(torch.long), dim=0) - 1
    return vision_bank_indices_T.masked_fill(~vision_mask_T, -1)


def gather_vision_embeds(
    inputs_TD: torch.Tensor,
    *,
    vision_bank_VD: torch.Tensor,
    vision_bank_indices_T: torch.Tensor,
) -> torch.Tensor:
    """Gather packed vision features into their placeholder token positions."""
    if vision_bank_VD.shape[0] == 0:
        return inputs_TD
    vision_bank_VD = vision_bank_VD.to(inputs_TD.dtype)
    is_vision_T1 = (vision_bank_indices_T >= 0).unsqueeze(-1)
    gathered_TD = vision_bank_VD[vision_bank_indices_T.clamp(min=0)]
    # The vision bank is DP-local, so global propagation through where omits
    # DP from the token PartitionSpec. Validate locally, then restore the exact
    # token layout at the fusion boundary.
    with spmd.local():
        fused_TD = torch.where(is_vision_T1, gathered_TD, inputs_TD)
    if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
        spmd.assert_type_like(fused_TD, inputs_TD)
    return fused_TD


def scatter_vision_embeds(
    inputs_embeds: torch.Tensor,
    *,
    vision_embeds: torch.Tensor,
    vision_positions: list[tuple[int, int, int]],
) -> torch.Tensor:
    """Copy packed vision features into the text sequence at placeholder runs.

    Args:
        inputs_embeds: ``(T, D)`` text embeddings, modified in place.
        vision_embeds: Packed vision features ``(total_tokens, dim)``.
        vision_positions: from ``get_vision_positions``.
    """
    vision_offset = 0
    for _, vision_start, num_tokens in vision_positions:
        inputs_embeds[vision_start : vision_start + num_tokens] = vision_embeds[
            vision_offset : vision_offset + num_tokens
        ].to(inputs_embeds.dtype)
        vision_offset += num_tokens

    if vision_offset != vision_embeds.shape[0]:
        raise ValueError(
            f"Vision placeholder runs consume {vision_offset} embeddings but "
            f"the packed vision output contains {vision_embeds.shape[0]}."
        )
    return inputs_embeds
