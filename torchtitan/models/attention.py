# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import functools
from typing import Callable, ClassVar

import torch
import torch.nn.functional as F
from torch.distributed.tensor.experimental._attention import create_cp_block_mask
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    AuxOutput,
    BlockMask,
    create_block_mask,
    flex_attention,
)


__all__ = [
    "AttentionOp",
    "get_causal_mask_mod",
    "get_document_mask_mod",
    "get_fixed_block_mask_mod",
    "get_create_mask_fn",
]


class _FlexAttentionWrapper(torch.nn.Module):
    _flex_attn: ClassVar[Callable] = torch.compile(
        flex_attention, mode="max-autotune-no-cudagraphs"
    )

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, *args: object, **kwargs: object
    ) -> [torch.Tensor | tuple[torch.Tensor, AuxOutput]]:
        # 1. _flex_attn has to be a class variable, otherwise there will
        #    be multiple complied flex_attention, which can be slow.
        # 2. `self._flex_attn` is not correct, `self` will be passed in
        #    as the first argument, which will cause an error.
        #    `FlexAttentionOp._flex_attn` is correct.
        return _FlexAttentionWrapper._flex_attn(*args, **kwargs)


class _ScaledDotProductAttentionWrapper(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        return F.scaled_dot_product_attention(*args, **kwargs)


class AttentionOp(torch.nn.Module):
    sdpa_backends: ClassVar[list[SDPBackend]] = []

    def __init__(self, use_flex_attn: bool) -> None:
        super().__init__()
        self.use_flex_attn = use_flex_attn
        if use_flex_attn:
            self._attn_op = _FlexAttentionWrapper()
        else:
            self._attn_op = _ScaledDotProductAttentionWrapper()
            self._init_sdpa_backend()

    @classmethod
    def _init_sdpa_backend(cls) -> None:
        if cls.sdpa_backends:
            return

        # Add CuDNN on B200 w/ highest priority
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
        attention_mask: BlockMask | None = None,
        scale: float | None = None,
    ) -> torch.Tensor:
        if self.use_flex_attn:
            assert attention_mask is not None
            return self._attn_op(q, k, v, block_mask=attention_mask, scale=scale)
        else:
            assert attention_mask is None
            assert self.sdpa_backends, "SDPA Backends should not be empty."
            with sdpa_kernel(self.sdpa_backends, set_priority=True):
                return self._attn_op(q, k, v, scale=scale)


# We cannot do inner function because we won't be able to cache it --
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


_compiled_create_block_mask: Callable | None = None


def get_create_mask_fn(
    use_flex_attn: bool,
    cp_mesh: torch.distributed.device_mesh.DeviceMesh | None = None,
) -> Callable:
    """
    Get a function that creates an attention mask. This is currently only
    mneaningful when FlexAttention is used. For SDPA, the returned
    function should never be called as SDPA doesn't support varlen yet.

    Note: the returned create_mask_fn will cache the masks if the
    input is the same.

    Args:
        use_flex_attn: Whether to use FlexAttention or not.
        cp_mesh: The device mesh to use for creating the block mask. If None, the
            function will return a function that creates a block mask for the current
            device mesh.

    Returns:
        create_mask_fn: a function hat creates an attention mask. This function
        will catch the results if the input is the same.
    """

    # This is not functional yet because we currently gate the use of Flex + CP
    # while we continue debugging accuracy issues. However, we want to evaluate
    # the user experience with CP enabled.
    global _compiled_create_block_mask
    if cp_mesh is not None:
        from torch.distributed.tensor.experimental._attention import _DispatchMode

        torch.distributed.tensor.experimental._attention._dispatch_mode = (
            _DispatchMode.MODULE_WRAPPER
        )
        if _compiled_create_block_mask is None:
            _compiled_create_block_mask = functools.partial(
                create_cp_block_mask, device_mesh=cp_mesh
            )
    elif _compiled_create_block_mask is None:
        _compiled_create_block_mask = torch.compile(create_block_mask)

    def dontcall(*args, **kwargs):
        raise RuntimeError(
            "This function should not be called. Please use the compiled version."
        )

    if not use_flex_attn:
        return dontcall

    # TODO: this cache number is kind of random, we should find a better way to set it.
    @functools.lru_cache(4)
    def create_block_mask_fn(*args, **kwargs):
        return _compiled_create_block_mask(*args, **kwargs)

    return create_block_mask_fn
