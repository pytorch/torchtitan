# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from typing import Any, Callable, ClassVar, Optional

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    create_block_mask,
    flex_attention,
)


# Flex Attention mask type. For each mask type, we initialize it at most once per
# batch. To record what it is initialized, FLEX_ATTN_MASK_T is used as the key to
# track the initialized mask.
FLEX_ATTN_MASK_T = tuple[Any, ...]


class FlexAttention(torch.nn.Module):
    """FlexAttention module that uses torch.nn.attention.flex_attention.

    This module is a wrapper around torch.nn.attention.flex_attention. This module
    implements certain common attention types, such as causal and block_causal.

    Args:
        attn_mask_type (str): The type of attention mask. Currently, we support
            "causal" and "block_causal". "causal" means the lower triangle of the
            attention matrix is masked. "block_causal" means the attention matrix
            is divided into blocks, where block boundary is defined by EOS token,
            and the lower triangle of each block is masked.
        batchify_size (Optional[int]): The size to be batchified. If specified, each
            sequence will be further divided to batches, where each batch has the
            maximum size of ``batchify_size``. A query will only attend to the keys
            within the same batch.
    """

    # We registered flex_attention related attributes as class variables as we
    # need to amortize the cost of compilation.
    flex_attn: ClassVar[Callable] = torch.compile(
        flex_attention, mode="max-autotune-no-cudagraphs"
    )
    compiled_create_block_mask: ClassVar[Callable] = torch.compile(create_block_mask)
    used_attn_mask_types: ClassVar[set[FLEX_ATTN_MASK_T]] = set()
    # Attention mask type to the created BlockMask.
    # This allows us to keep track the created block masks for each
    # new batch. We will use this to update the block mask when a
    # new batch is created. This also allows user to create different
    # block masks for different layers.
    block_masks: ClassVar[dict[FLEX_ATTN_MASK_T, BlockMask]] = {}

    # Instance variables.
    attn_mask_type: str

    def __init__(
        self, attn_mask_type: str, batchify_size: Optional[int] = None
    ) -> None:
        super().__init__()
        if attn_mask_type not in ["causal", "block_causal"]:
            raise ValueError(f"Unrecognized attn_mask_type {attn_mask_type}.")
        self.attn_mask_type = attn_mask_type
        self.batchify_size = batchify_size

        FlexAttention.used_attn_mask_types.add(self.mask_key)

    @property
    def mask_key(self) -> FLEX_ATTN_MASK_T:
        return (self.attn_mask_type, self.batchify_size)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        block_mask = FlexAttention.block_masks[self.mask_key]
        return FlexAttention.flex_attn(q, k, v, block_mask=block_mask)

    @staticmethod
    def _get_causal_mask_mod() -> _mask_mod_signature:
        def causal_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return q_idx >= kv_idx

        return causal_mask

    @staticmethod
    def _get_block_causal_mask_mod(
        batch: torch.Tensor, eos_id: int
    ) -> _mask_mod_signature:
        # batch is [b, s, h, d] shape
        mask = batch == eos_id
        mask[:, -1] = True
        acc_mask = torch.cumsum(torch.where(mask, 1, 0), dim=1)
        seq_idx = torch.zeros_like(acc_mask, dtype=torch.int32)
        seq_idx[:, 1:] = acc_mask[:, :-1]

        def block_causal_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (seq_idx[b, q_idx] == seq_idx[b, kv_idx]) & (q_idx >= kv_idx)

        return block_causal_mask

    def batchify_mask_mod(
        mask_mod: _mask_mod_signature, batchify_size: int
    ) -> _mask_mod_signature:
        """Given arbirary mask_mod, batchify it to only allow attention within the same batch.

        Args:
            mask_mod: The mask mod to apply to the documents
            batch_size: The number of tokens in each batch.
        """

        # Credit to @drisspg.
        def batched_mask_mod(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            # Get the batch index of the query and key
            q_batch = q_idx // batchify_size
            kv_batch = kv_idx // batchify_size
            # Only allow attention within the same batch
            same_batch = q_batch == kv_batch
            # Apply the original mask mod
            inner_mask = mask_mod(b, h, q_idx % batchify_size, kv_idx % batchify_size)

            return same_batch & inner_mask

        batched_mask_mod.__name__ = (
            f"batched_mask_mod_{mask_mod.__name__}_batch_size_{batchify_size}"
        )

        return batched_mask_mod

    @staticmethod
    @torch.no_grad()
    def init_attention_mask(batch: torch.Tensor, eos_id: Optional[int] = None) -> None:
        # batch is [b, s, h, d] shape
        for mask_key in FlexAttention.used_attn_mask_types:
            attn_mask_type, batchify_size = mask_key
            match attn_mask_type:
                case "causal":
                    if FlexAttention.block_masks.get(mask_key, None) is not None:
                        continue
                    # We don't care about batch dimension --
                    # all samples have the same lower triangle mask.
                    batch_dimension = 1
                    mask_mod = FlexAttention._get_causal_mask_mod()
                    if batchify_size is not None and batchify_size > 0:
                        mask_mod = FlexAttention.batchify_mask_mod(
                            mask_mod, batchify_size
                        )
                case "block_causal":
                    if eos_id is None:
                        raise RuntimeError(
                            "eos_id must be provided for block_causal mask."
                        )
                    batch_dimension = batch.shape[0]
                    mask_mod = FlexAttention._get_block_causal_mask_mod(batch, eos_id)
                    if batchify_size is not None and batchify_size > 0:
                        mask_mod = FlexAttention.batchify_mask_mod(
                            mask_mod, batchify_size
                        )
                case _:
                    raise RuntimeError(f"Shouldn't reach here. {attn_mask_type}")

            seq_len = batch.shape[1]
            block_mask = FlexAttention.compiled_create_block_mask(
                mask_mod, batch_dimension, None, seq_len, seq_len
            )
            FlexAttention.block_masks[mask_key] = block_mask


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, attn_mask_type: str) -> None:
        super().__init__()
        if attn_mask_type != "causal":
            raise ValueError(
                "TorchTitan with SDPA currently only supports causal mask."
            )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def build_attention(
    use_flex_attn: bool, attn_mask_type: str, batchify_size: Optional[int] = None
):
    if use_flex_attn:
        return FlexAttention(attn_mask_type, batchify_size)
    else:
        if batchify_size is not None:
            raise ValueError(
                "TorchTitan with SDPA currently does not support batchify_size."
            )
        if attn_mask_type != "causal":
            raise ValueError(
                "TorchTitan with SDPA currently only supports causal mask."
            )
        return ScaledDotProductAttention(attn_mask_type)


def init_attention_mask(batch: torch.Tensor, eos_id: Optional[int] = None) -> None:
    FlexAttention.init_attention_mask(batch, eos_id)
