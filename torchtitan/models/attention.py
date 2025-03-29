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


class FlexAttn(torch.nn.Module):
    # We registered flex_attention related attributes as class variables as we
    # need to amortize the cost of compilation. Enabling per-instance flex_attention
    # is not supported.
    block_mask: ClassVar[Optional[BlockMask]] = None
    flex_attn: ClassVar[Optional[Callable]] = None
    attn_bias_type: ClassVar[Optional[str]] = None
    compiled_create_block_mask: ClassVar[Optional[Callable]] = None

    def __init__(self, attn_bias_type: str) -> None:
        super().__init__()
        if FlexAttn.attn_bias_type is not None:
            assert (
                FlexAttn.attn_bias_type == attn_bias_type
            ), "All FlexAttention must have the same configurations."
        else:
            if attn_bias_type not in ["causal", "block_causal"]:
                raise ValueError(f"Unrecognized attn_bias_type {attn_bias_type}.")
            FlexAttn.attn_bias_type = attn_bias_type

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        assert FlexAttn.block_mask is not None
        assert FlexAttn.flex_attn is not None
        return FlexAttn.flex_attn(q, k, v, block_mask=FlexAttn.block_mask)

    @classmethod
    def _get_causal_mask_fn(cls) -> Callable:
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        return causal_mask

    @classmethod
    def _get_block_causal_mask_fn(cls, batch: torch.Tensor, eos_id: int) -> Callable:
        mask = batch == eos_id
        mask[:, -1] = True
        acc_mask = torch.cumsum(torch.where(mask, 1, 0).flatten(), dim=0)
        seq_idx = torch.zeros_like(acc_mask, dtype=torch.int32)
        seq_idx[1:] = acc_mask[:-1]

        def block_causal_mask(b, h, q_idx, kv_idx):
            return (seq_idx[q_idx] == seq_idx[kv_idx]) & (q_idx >= kv_idx)

        return block_causal_mask

    @classmethod
    @torch.no_grad()
    def init_attention_bias(
        cls, batch: torch.Tensor, eos_id: Optional[int] = None
    ) -> None:
        if cls.block_mask is not None and cls.attn_bias_type == "causal":
            # We don't need to create another block mask for causal masking if existed.
            return

        match cls.attn_bias_type:
            case "causal":
                mask_fn = cls._get_causal_mask_fn()
            case "block_causal":
                mask_fn = cls._get_block_causal_mask_fn(batch, eos_id)
            case _:
                raise RuntimeError(f"Shouldn't reach here. {cls.attn_bias_type}")

        seq_len = batch.shape[1]
        if cls.compiled_create_block_mask is None:
            cls.compiled_create_block_mask = torch.compile(create_block_mask)
        cls.block_mask = cls.compiled_create_block_mask(
            mask_fn, None, None, seq_len, seq_len
        )
        cls.flex_attn = torch.compile(flex_attention, mode="max-autotune-no-cudagraphs")


class SDPA(torch.nn.Module):
    def __init__(self, attn_bias_type: str) -> None:
        super().__init__()
        if attn_bias_type != "causal":
            raise ValueError(
                "TorchTitan with SDPA currently only supports causal mask."
            )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    @classmethod
    @torch.no_grad()
    def init_attention_bias(
        cls,
        batch: torch.Tensor,
        eos_id: Optional[int] = None,
    ) -> None:
        # For SDPA, we don't need to do anything.
        return


_selected_attention = None


def build_attention(use_flex_attn: bool, attn_bias_type: str):
    global _selected_attention
    if use_flex_attn:
        assert _selected_attention is None or _selected_attention == FlexAttn
        _selected_attention = FlexAttn
        return FlexAttn(attn_bias_type)
    else:
        assert _selected_attention is None or _selected_attention == SDPA
        _selected_attention = SDPA
        return SDPA(attn_bias_type)


def init_attention_bias(batch: torch.Tensor, eos_id: Optional[int] = None) -> None:
    global _selected_attention
    _selected_attention.init_attention_bias(batch, eos_id)
