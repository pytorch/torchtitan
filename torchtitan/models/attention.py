# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import functools
from collections.abc import Callable
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    AuxOutput,
    BlockMask,
    create_block_mask,
    flex_attention,
)


__all__ = [
    "FlexAttentionWrapper",
    "ScaledDotProductAttentionWrapper",
    "get_causal_mask_mod",
    "get_document_mask_mod",
    "get_fixed_block_mask_mod",
    "create_attention_mask",
]


class FlexAttentionWrapper(torch.nn.Module):
    """Wrapper around `flex_attention` to make it torch.compile and CP compatible.

    This wrapper surves two purposes:
    1) Invoke `torch.compile` with a valid mode "max-autotune-no-cudagraphs" to
       achieve a good performance.
    2) This wrapper being a wrapper allows us to apply _ContextParallal to it.

    Note:
    The forward function must have q, k, v as the first three arguments, and
    block_mask as a keyword argument to be compatible with _ContextParallel.
    """

    _compiled_flex_attn: ClassVar[Callable] = torch.compile(
        flex_attention, mode="max-autotune-no-cudagraphs"
    )

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        block_mask: BlockMask,
        scale: float | None = None,
    ) -> [torch.Tensor | tuple[torch.Tensor, AuxOutput]]:
        # 1. _compiled_flex_attn has to be a class variable, otherwise there will
        #    be multiple complied flex_attention, which can be slow.
        # 2. `self._compiled_flex_attn` is not correct, `self` will be passed in
        #    as the first argument, which will cause an error.
        #    `FlexAttentionOp._compiled_flex_attn` is correct.
        return FlexAttentionWrapper._compiled_flex_attn(
            q, k, v, block_mask=block_mask, scale=scale
        )


class ScaledDotProductAttentionWrapper(torch.nn.Module):
    """Wrapper around `F.scaled_dot_product_attention` to make it CP compatible.

    This wrapper is needed because `F.scaled_dot_product_attention` is not
    a torch.nn.Module, and thus cannot be applied _ContextParallel.
    So we need to wrap it into a torch.nn.Module.

    Note:
    The forward function must have q, k, v as the first three arguments to be
    compatible with _ContextParallel.
    """

    # TODO: remove sdpa_backends after PyTorch 2.9 is released.
    sdpa_backends: ClassVar[list[SDPBackend]] = []

    def __init__(self) -> None:
        super().__init__()
        self._init_sdpa_backend()

    @classmethod
    def _init_sdpa_backend(cls) -> None:
        if cls.sdpa_backends:
            return

        # Always make CuDNN as the highest priority if available.
        cls.sdpa_backends = [
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
    ) -> torch.Tensor:
        assert self.sdpa_backends, "SDPA Backends should not be empty."
        with sdpa_kernel(self.sdpa_backends, set_priority=True):
            return F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=True)


# We cannot do inner function/closure because we won't be able to cache it --
# if we an inner function, a new closure will be created every time
# `get_causal_mask_mod` is called.
def _causal_mask(
    b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
):
    return q_idx >= kv_idx


def get_causal_mask_mod() -> _mask_mod_signature:
    return _causal_mask


def get_document_mask_mod(
    mask_mod: _mask_mod_signature, batch: torch.Tensor, eos_id: int
) -> _mask_mod_signature:
    # batch is [b, s, h, d] shape
    mask = batch == eos_id
    mask[:, -1] = True
    acc_mask = torch.cumsum(torch.where(mask, 1, 0), dim=1)
    seq_idx = torch.zeros_like(acc_mask, dtype=torch.int32)
    seq_idx[:, 1:] = acc_mask[:, :-1]

    def document_mask(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ):
        inner_mask = mask_mod(b, h, q_idx, kv_idx)
        return seq_idx[b, q_idx] == seq_idx[b, kv_idx] & inner_mask

    return document_mask


def get_fixed_block_mask_mod(
    mask_mod: _mask_mod_signature, fixed_block_size: int
) -> _mask_mod_signature:
    """
    Given an arbitrary mask_mod, divide the input sequence to blocks
    and only allow attention within the same block.

    Args:
        mask_mod: The mask mod to apply to the documents
        fixed_block_size: The number of tokens in each block.
    """

    # Credit to @drisspg.
    def blocked_mask_mod(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ):
        # Get the block index of the query and key
        q_block = q_idx // fixed_block_size
        kv_block = kv_idx // fixed_block_size
        # Only allow attention within the same block
        same_block = q_block == kv_block
        # Apply the original mask mod
        inner_mask = mask_mod(b, h, q_idx % fixed_block_size, kv_idx % fixed_block_size)

        return same_block & inner_mask

    blocked_mask_mod.__name__ = (
        f"blocked_mask_mod_{mask_mod.__name__}_fixed_block_size_{fixed_block_size}"
    )

    return blocked_mask_mod


_compiled_create_block_mask = torch.compile(create_block_mask)


@functools.lru_cache(4)
def create_attention_mask(*args, **kwargs):
    return _compiled_create_block_mask(*args, **kwargs)
