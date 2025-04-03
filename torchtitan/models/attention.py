# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from typing import Callable, ClassVar, Optional

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)


class FlexAttention(torch.nn.Module):
    # We registered flex_attention related attributes as class variables as we
    # need to amortize the cost of compilation.
    flex_attn: ClassVar[Callable] = torch.compile(
        flex_attention, mode="max-autotune-no-cudagraphs"
    )
    compiled_create_block_mask: ClassVar[Callable] = torch.compile(create_block_mask)
    used_attn_mask_types: ClassVar[set[str]] = set()
    # Attention mask type to the created BlockMask.
    # This allows us to keep track the created block masks for each
    # new batch. We will use this to update the block mask when a
    # new batch is created. This also allows user to create different
    # block masks for different layers.
    block_masks: ClassVar[dict[str, BlockMask]] = {}

    # Instance variables.
    attn_mask_type: str

    def __init__(self, attn_mask_type: str) -> None:
        super().__init__()
        if attn_mask_type not in ["causal", "block_causal"]:
            raise ValueError(f"Unrecognized attn_mask_type {attn_mask_type}.")
        self.attn_mask_type = attn_mask_type
        FlexAttention.used_attn_mask_types.add(attn_mask_type)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        block_mask = FlexAttention.block_masks[self.attn_mask_type]
        return FlexAttention.flex_attn(q, k, v, block_mask=block_mask)

    @staticmethod
    def _get_causal_mask_fn() -> Callable:
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        return causal_mask

    @staticmethod
    def _get_block_causal_mask_fn(batch: torch.Tensor, eos_id: int) -> Callable:
        # batch is [b, s, h, d] shape
        mask = batch == eos_id
        mask[:, -1] = True
        acc_mask = torch.cumsum(torch.where(mask, 1, 0), dim=1)
        seq_idx = torch.zeros_like(acc_mask, dtype=torch.int32)
        seq_idx[:, 1:] = acc_mask[:, :-1]

        def block_causal_mask(b, h, q_idx, kv_idx):
            return (seq_idx[b, q_idx] == seq_idx[b, kv_idx]) & (q_idx >= kv_idx)

        return block_causal_mask

    @staticmethod
    @torch.no_grad()
    def init_attention_mask(batch: torch.Tensor, eos_id: Optional[int] = None) -> None:
        # batch is [b, s, h, d] shape
        for attn_mask_type in FlexAttention.used_attn_mask_types:
            match attn_mask_type:
                case "causal":
                    if FlexAttention.block_masks.get(attn_mask_type, None) is not None:
                        continue
                    # We don't care about batch dimension --
                    # all samples have the same lower triangle mask.
                    batch_dimension = 1
                    mask_fn = FlexAttention._get_causal_mask_fn()
                case "block_causal":
                    if eos_id is None:
                        raise RuntimeError(
                            "eos_id must be provided for block_causal mask."
                        )
                    batch_dimension = batch.shape[0]
                    mask_fn = FlexAttention._get_block_causal_mask_fn(batch, eos_id)
                case _:
                    raise RuntimeError(f"Shouldn't reach here. {attn_mask_type}")

            seq_len = batch.shape[1]
            block_mask = FlexAttention.compiled_create_block_mask(
                mask_fn, batch_dimension, None, seq_len, seq_len
            )
            FlexAttention.block_masks[attn_mask_type] = block_mask


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


def build_attention(use_flex_attn: bool, attn_mask_type: str):
    if use_flex_attn:
        return FlexAttention(attn_mask_type)
    else:
        return ScaledDotProductAttention(attn_mask_type)


def init_attention_mask(batch: torch.Tensor, eos_id: Optional[int] = None) -> None:
    FlexAttention.init_attention_mask(batch, eos_id)
