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

    This wrapper serves two purposes:
    1) Invoke `torch.compile` with a valid mode "max-autotune-no-cudagraphs" to
       achieve good performance.
    2) Being a wrapper allows us to apply _ContextParallel to it.

    Note:
        The forward function must have q, k, v as the first three arguments, and
        block_mask as a keyword argument to be compatible with _ContextParallel.
    """

    _compiled_flex_attn: ClassVar[Callable] = torch.compile(
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
        self, attn_mask_type: str, fixed_block_size: int | None = None
    ) -> None:
        super().__init__()
        if attn_mask_type not in ["causal", "block_causal"]:
            raise ValueError(f"Unrecognized attn_mask_type {attn_mask_type}.")
        self.attn_mask_type = attn_mask_type
        self.fixed_block_size = fixed_block_size

        self.mask_cache = {}
        FlexAttention.used_attn_mask_types.add(self.mask_key)

    @property
    def mask_key(self) -> FLEX_ATTN_MASK_T:
        return (self.attn_mask_type, self.fixed_block_size)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float | None = None,
        sink_weights: torch.Tensor | None = None,
        sliding_window: int = 0,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        if sink_weights is None:
            block_mask = FlexAttention.block_masks[self.mask_key]
            return FlexAttention.flex_attn(q, k, v, block_mask=block_mask, scale=scale)

        B, H_q, S_q, D = q.shape
        _, H_kv, S_kv, _ = k.shape

        # regular (no-sink) mask + no extra KV col
        mask_key = (sliding_window, S_q, S_kv)
        if mask_key not in self.mask_cache:
            if sliding_window is not None and sliding_window > 0:
                mask_mod = FlexAttention._get_sliding_window_mask_mod(sliding_window)
            else:
                mask_mod = FlexAttention._get_causal_mask_mod()
            block_mask = create_block_mask(
                mask_mod, B, H_q, S_q, S_kv,
                _compile=True, device=q.device # NOTE: set _compile=False if sampling for debugging
            )
            self.mask_cache[mask_key] = block_mask

        block_mask = self.mask_cache[mask_key]

        # run fast flex_attn and return LSE
        out, lse = FlexAttention.flex_attn(
            q, k, v,
            block_mask=block_mask,
            enable_gqa=enable_gqa,
            return_lse=True
        )

        # rescale by sigma(lse - w[h]) and broadcast over D
        if sink_weights is not None:
            w = sink_weights  # [H]
            scale = torch.sigmoid(lse - w.view(1, -1, 1)).unsqueeze(-1)  # [B,H,S,1]
            out = out * scale

        out = out.to(q.dtype)
        return out

    @staticmethod
    def _get_sliding_window_mask_mod(window: int):
        """
        Returns a mask_mod function that
        - only allows kv_idx ≤ q_idx (causal)
        - and only if (q_idx - kv_idx) ≤ window
        """
        def sliding_mod(b, h, q_idx, kv_idx):
            # causal within window
            keep = (kv_idx <= q_idx) & (q_idx - kv_idx <= window)
            return keep
        return sliding_mod

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

    @staticmethod
    def _fixed_block_mask_mod(
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
            inner_mask = mask_mod(
                b, h, q_idx % fixed_block_size, kv_idx % fixed_block_size
            )

            return same_block & inner_mask

        blocked_mask_mod.__name__ = (
            f"blocked_mask_mod_{mask_mod.__name__}_fixed_block_size_{fixed_block_size}"
>>>>>>> 0313a6fb (gptoss experimental support)
        )


class ScaledDotProductAttentionWrapper(torch.nn.Module):
    """Wrapper around `F.scaled_dot_product_attention` to make it CP compatible.

    This wrapper is needed because `F.scaled_dot_product_attention` is not
    a torch.nn.Module, and thus cannot be applied with _ContextParallel.
    We need to wrap it into a torch.nn.Module.

    Note:
        The forward function must have q, k, v as the first three arguments to be
        compatible with _ContextParallel.
    """

    # TODO: remove sdpa_backends after PyTorch 2.9 is released.
    sdpa_backends: ClassVar[list[SDPBackend]] = []

    def __init__(self) -> None:
        super().__init__()
        if not self.sdpa_backends:
            self.sdpa_backends = [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
            ]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
    ) -> torch.Tensor:
        with sdpa_kernel(self.sdpa_backends, set_priority=True):
            return F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=True)


# We cannot do inner function/closure because we won't be able to cache it --
# if we an inner function, a new closure will be created every time
# `get_causal_mask_mod` is called.
def _causal_mask(
    b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
) -> torch.Tensor:
    """Causal mask that prevents attention to future tokens."""
    return q_idx >= kv_idx


def get_causal_mask_mod() -> _mask_mod_signature:
    """Returns a causal mask modifier for flex attention.

    Returns:
        A mask modifier function that implements causal masking.
    """
    return _causal_mask


def get_document_mask_mod(batch: torch.Tensor, eos_id: int) -> _mask_mod_signature:
    """Creates a document mask that prevents attention across document boundaries.

    Args:
        batch: Input batch tensor with shape [b, s, h, d]
        eos_id: End-of-sequence token ID that marks document boundaries

    Returns:
        A mask modifier function that implements document-level masking.
    """
    # batch is [b, s, h, d] shape
    eos_mask = batch == eos_id
    eos_mask[:, -1] = True
    cumulative_mask = torch.cumsum(torch.where(eos_mask, 1, 0), dim=1)
    sequence_indices = torch.zeros_like(cumulative_mask, dtype=torch.int32)
    sequence_indices[:, 1:] = cumulative_mask[:, :-1]

    def document_mask(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        return sequence_indices[b, q_idx] == sequence_indices[b, kv_idx]

    return document_mask


def get_fixed_block_mask_mod(fixed_block_size: int) -> _mask_mod_signature:
    """
    Divide the input sequence into blocks and only allow attention within the same block.

    Args:
        fixed_block_size: The number of tokens in each block.

    Returns:
        A mask modifier function that implements block-wise attention masking.
    """

    # Credit to @drisspg.
    def blocked_mask_mod(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        # Get the block index of the query and key
        q_block = q_idx // fixed_block_size
        kv_block = kv_idx // fixed_block_size
        # Only allow attention within the same block
        return q_block == kv_block

    blocked_mask_mod.__name__ = f"blocked_mask_mod_fixed_block_size_{fixed_block_size}"

    return blocked_mask_mod


_compiled_create_block_mask = torch.compile(create_block_mask)


@functools.lru_cache(4)
def create_attention_mask(*args, **kwargs):
    """Create an attention mask using compiled create_block_mask.

    This function is cached to avoid recreating BlockMasks for the same
    argumens.
    """
    return _compiled_create_block_mask(*args, **kwargs)
