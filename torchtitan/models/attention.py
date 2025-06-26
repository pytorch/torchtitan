# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from typing import Callable, ClassVar

import torch
import torch.nn.functional as F
from torch.nested import nested_tensor_from_jagged
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    create_block_mask,
    flex_attention,
)

from torchtitan.tools.utils import has_cuda_capability

# Attention mask type. For each mask type, we initialize it at most once per
# batch. To record what it is initialized, ATTN_MASK_T is used as the key to
# track the initialized mask.
ATTN_MASK_T = tuple[str, int | None]


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
        fixed_block_size (int | None): The block size to be used to perform attention.
            If specified, each sequence will be further divided to blocks, where each
            block has the maximum size of ``fixed_block_size``. A query will only attend
            to the keys within the same block.
    """

    # We registered flex_attention related attributes as class variables as we
    # need to amortize the cost of compilation.
    flex_attn: ClassVar[Callable] = torch.compile(
        flex_attention, mode="max-autotune-no-cudagraphs"
    )
    compiled_create_block_mask: ClassVar[Callable] = torch.compile(create_block_mask)
    used_attn_mask_types: ClassVar[set[ATTN_MASK_T]] = set()
    # Attention mask type to the created BlockMask.
    # This allows us to keep track the created block masks for each
    # new batch. We will use this to update the block mask when a
    # new batch is created. This also allows user to create different
    # block masks for different layers.
    block_masks: ClassVar[dict[ATTN_MASK_T, BlockMask]] = {}

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

        FlexAttention.used_attn_mask_types.add(self.mask_key)

    @property
    def mask_key(self) -> ATTN_MASK_T:
        return (self.attn_mask_type, self.fixed_block_size)

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

    @staticmethod
    def _fixed_block_mask_mod(
        mask_mod: _mask_mod_signature, fixed_block_size: int
    ) -> _mask_mod_signature:
        """
        Given an arbirary mask_mod, divide the input sequence to blocks
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
        )

        return blocked_mask_mod

    @staticmethod
    @torch.no_grad()
    def init_attention_mask(batch: torch.Tensor, eos_id: int | None = None) -> None:
        # batch is [b, s, h, d] shape
        for mask_key in FlexAttention.used_attn_mask_types:
            attn_mask_type, fixed_block_size = mask_key
            match attn_mask_type:
                case "causal":
                    if FlexAttention.block_masks.get(mask_key, None) is not None:
                        continue
                    # We don't care about batch dimension --
                    # all samples have the same lower triangle mask.
                    batch_dimension = 1
                    mask_mod = FlexAttention._get_causal_mask_mod()
                case "block_causal":
                    if eos_id is None:
                        raise RuntimeError(
                            "eos_id must be provided for block_causal mask."
                        )
                    batch_dimension = batch.shape[0]
                    mask_mod = FlexAttention._get_block_causal_mask_mod(batch, eos_id)
                case _:
                    raise RuntimeError(f"Shouldn't reach here. {attn_mask_type}")

            if fixed_block_size is not None and fixed_block_size > 0:
                mask_mod = FlexAttention._fixed_block_mask_mod(
                    mask_mod, fixed_block_size
                )

            seq_len = batch.shape[1]
            block_mask = FlexAttention.compiled_create_block_mask(
                mask_mod, batch_dimension, None, seq_len, seq_len
            )
            FlexAttention.block_masks[mask_key] = block_mask


class ScaledDotProductAttention(torch.nn.Module):
    backends: ClassVar[list[SDPBackend]] = []

    # Offsets between the packed sequences in the batch used to create nested tensors.
    offsets: ClassVar[torch.Tensor | None] = None

    used_attn_mask_types: ClassVar[set[ATTN_MASK_T]] = set()

    def __init__(self, attn_mask_type: str) -> None:
        super().__init__()
        
        ScaledDotProductAttention._init_backend()

        self.attn_mask_type = attn_mask_type

        ScaledDotProductAttention.used_attn_mask_types.add(self.mask_key)

    @classmethod
    def _init_backend(cls) -> None:
        if cls.backends:
            return

        # Add CuDNN on B200 w/ highest priority
        cls.backends = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
        if has_cuda_capability(10, 0):
            cls.backends.insert(0, SDPBackend.CUDNN_ATTENTION)

    @property
    def mask_key(self) -> ATTN_MASK_T:
        return (self.attn_mask_type, None)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        assert self.backends, "SDPA Backends should not be empty."
        with sdpa_kernel(self.backends, set_priority=True):

            if ScaledDotProductAttention.offsets is None:
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                original_shape = q.shape
                if q.size(0) == 1:
                    # Create nested tensor: [1, h, s, d] -> [num_samples, h, j1, d]
                    q_nested = nested_tensor_from_jagged(
                        q.view(q.shape[1:]),
                        ScaledDotProductAttention.offsets,
                        jagged_dim=2,
                    )
                    k_nested = nested_tensor_from_jagged(
                        k.view(k.shape[1:]),
                        ScaledDotProductAttention.offsets,
                        jagged_dim=2,
                    )
                    v_nested = nested_tensor_from_jagged(
                        v.view(v.shape[1:]),
                        ScaledDotProductAttention.offsets,
                        jagged_dim=2,
                    )

                    act_nested = F.scaled_dot_product_attention(
                        q_nested, k_nested, v_nested, is_causal=True
                    )

                    return act_nested.values().view(original_shape)
                else:
                    # Flatten the packed samples along dim 2: [bs, h, s, d] -> [1, h, bs*s, d]
                    q_packed = (
                        q.permute(0, 2, 1, 3)
                        .reshape(1, -1, q.shape[1], q.shape[3])
                        .permute(0, 2, 1, 3)
                    )
                    del q
                    # Create nested tensor: [1, h, bs*s, d] -> [num_samples, h, j1, d]
                    q_nested = nested_tensor_from_jagged(
                        q_packed.view(q_packed.shape[1:]),
                        ScaledDotProductAttention.offsets,
                        jagged_dim=2,
                    )

                    k_packed = (
                        k.permute(0, 2, 1, 3)
                        .reshape(1, -1, k.shape[1], k.shape[3])
                        .permute(0, 2, 1, 3)
                    )
                    del k
                    k_nested = nested_tensor_from_jagged(
                        k_packed.view(k_packed.shape[1:]),
                        ScaledDotProductAttention.offsets,
                        jagged_dim=2,
                    )

                    v_packed = (
                        v.permute(0, 2, 1, 3)
                        .reshape(1, -1, v.shape[1], v.shape[3])
                        .permute(0, 2, 1, 3)
                    )
                    del v
                    v_nested = nested_tensor_from_jagged(
                        v_packed.view(v_packed.shape[1:]),
                        ScaledDotProductAttention.offsets,
                        jagged_dim=2,
                    )

                    act_nested = F.scaled_dot_product_attention(
                        q_nested, k_nested, v_nested, is_causal=True
                    )

                    # Repack samples along dim 2 and restore original shape: [num_samples, h, j1, d] -> [bs, h, s, d]
                    return (
                        act_nested.values()
                        .unsqueeze(0)
                        .permute(0, 2, 1, 3)
                        .reshape(
                            original_shape[0],
                            -1,
                            act_nested.shape[1],
                            act_nested.shape[3],
                        )
                        .permute(0, 2, 1, 3)
                    )

    @staticmethod
    @torch.no_grad()
    def _get_offsets(batch: torch.Tensor, eos_id: int) -> torch.Tensor:
        # Determine packed sequence boundaries.
        mask = batch == eos_id

        indices = mask.flatten().nonzero().flatten()

        # In case the last token is not EOS, we need to add an extra element to the indices.
        if indices.numel() == 0 or indices[-1] != batch.numel() - 1:
            addition_elements = 2
        else:
            addition_elements = 1

        # Store the offsets between the packed sequences in the batch.
        offsets = torch.empty(
            (indices.size(0) + addition_elements),
            dtype=indices.dtype,
            device=batch.device,
        )
        offsets[0] = 0
        offsets[1 : indices.size(0) + 1] = indices.flatten() + 1
        offsets[-1] = batch.numel()

        return offsets

    @staticmethod
    @torch.no_grad()
    def init_attention_mask(batch: torch.Tensor, eos_id: int | None = None) -> None:

        for mask_key in ScaledDotProductAttention.used_attn_mask_types:
            attn_mask_type, _ = mask_key
            match attn_mask_type:
                case "causal":
                    return
                case "block_causal":
                    if eos_id is None:
                        raise RuntimeError(
                            "eos_id must be provided for block_causal mask."
                        )
                    ScaledDotProductAttention.offsets = (
                        ScaledDotProductAttention._get_offsets(batch, eos_id)
                    )
                case _:
                    raise RuntimeError(f"Shouldn't reach here. {attn_mask_type}")


def build_attention(
    use_flex_attn: bool, attn_mask_type: str, fixed_block_size: int | None = None
):
    if use_flex_attn:
        return FlexAttention(attn_mask_type, fixed_block_size)
    else:
        if fixed_block_size is not None:
            raise ValueError(
                "TorchTitan with SDPA currently does not support fixed_block_size."
            )
        return ScaledDotProductAttention(attn_mask_type)


def init_attention_mask(
    batch: torch.Tensor, eos_id: int | None = None, use_flex_attn: bool = True
) -> None:
    if use_flex_attn:
        FlexAttention.init_attention_mask(batch, eos_id)
    else:
        ScaledDotProductAttention.init_attention_mask(batch, eos_id)
