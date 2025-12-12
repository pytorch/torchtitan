# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import functools
from collections.abc import Callable
from typing import ClassVar, NamedTuple

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    create_block_mask,
    flex_attention,
)

from torch.nn.attention.varlen import varlen_attn


__all__ = [
    "FlexAttentionWrapper",
    "ScaledDotProductAttentionWrapper",
    "VarlenAttentionWrapper",
    "VarlenMetadata",
    "get_causal_mask_mod",
    "get_document_mask_mod",
    "get_sliding_window_mask_mod",
    "get_fixed_block_mask_mod",
    "create_attention_mask",
]


class VarlenMetadata(NamedTuple):
    """
    Cumulative sequence positions for queries and keys/values.

    """

    cu_seq_q: torch.Tensor
    cu_seq_k: torch.Tensor
    max_q: int
    max_k: int


class VarlenAttentionWrapper(torch.nn.Module):
    _compiled_varlen_attn: ClassVar[Callable] = torch.compile(
        varlen_attn, mode="max-autotune-no-cudagraphs"
    )

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        head_dim: torch.Tensor,
        attention_masks: VarlenMetadata,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        cu_seq_q = attention_masks.cu_seq_q
        cu_seq_k = attention_masks.cu_seq_k
        max_q = attention_masks.max_q
        max_k = attention_masks.max_k

        n_local_heads = xq.shape[1]
        xq_packed = xq.transpose(1, 2).reshape(-1, n_local_heads, head_dim)
        xk_packed = xk.transpose(1, 2).reshape(-1, n_local_heads, head_dim)
        xv_packed = xv.transpose(1, 2).reshape(-1, n_local_heads, head_dim)

        return VarlenAttentionWrapper._compiled_varlen_attn(
            xq_packed,
            xk_packed,
            xv_packed,
            cu_seq_q,
            cu_seq_k,
            max_q,
            max_k,
            is_causal=True,
        )


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
        flex_attention,
        # This options also encapsulate max-autotune-no-cudagraphs.
        options={
            "wrap_inductor_compiled_regions": True,
            "max_autotune": True,
            "coordinate_descent_tuning": True,
            "triton.cudagraphs": False,
        },
    )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        block_mask: BlockMask,
        scale: float | None = None,
        return_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # 1. _compiled_flex_attn has to be a class variable, otherwise there will
        #    be multiple compiled flex_attention instances, which can be slow.
        # 2. `self._compiled_flex_attn` is not correct, `self` will be passed in
        #    as the first argument, which will cause an error.
        #    `FlexAttentionWrapper._compiled_flex_attn` is correct.
        # 3. Used `return_lse` instead of `return_aux` because of easier TP module notation
        #    to convert `lse` to be DTensor.
        return FlexAttentionWrapper._compiled_flex_attn(
            q,
            k,
            v,
            block_mask=block_mask,
            scale=scale,
            return_lse=return_lse,
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


def get_document_mask_mod_using_eos_id(
    batch: torch.Tensor,
    eos_id: int,
) -> _mask_mod_signature:
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


def get_document_mask_mod_using_sequence_indices(
    batch: torch.Tensor,
    seq_lengths: torch.Tensor,  # Shape: [B, Num_Docs]
    seq_start_indices: torch.Tensor,  # Shape: [B, Num_Docs]
) -> _mask_mod_signature:
    device = batch.device
    B, S = batch.shape[:2]  # Derive S from the actual batch, not metadata
    # Initialize the ID map [B, S]
    document_ids = torch.zeros((B, S), dtype=torch.int32, device=device)

    # Filter valid starts (ignore -1 padding)
    valid_mask = seq_start_indices != -1

    # Scatter '1' at the start of every document
    # This creates a map like: [0, 0, 1, 0, 0, 1, 0...]
    batch_rows = (
        torch.arange(B, device=device)
        .unsqueeze(1)
        .expand_as(seq_start_indices)[valid_mask]
    )
    start_cols = seq_start_indices[valid_mask].long()

    start_cols = start_cols.clamp(0, S - 1)
    document_ids[batch_rows, start_cols] = 1

    # Cumsum to propagate IDs
    # [0, 0, 1, 0, 0] -> [0, 0, 1, 1, 1]
    # We start from 0, so the first document becomes ID 1, next is 2, etc.
    seq_ids = document_ids.cumsum(dim=1)

    # Calculate where the valid data actually ends per row
    doc_ends = seq_start_indices + seq_lengths
    # Get the max end index per batch row
    last_valid_end = torch.max(torch.where(valid_mask, doc_ends, -1), dim=1).values

    # Create mask: True where position >= last_valid_end
    pos_indices = torch.arange(S, device=device).unsqueeze(0)
    is_padding = pos_indices >= last_valid_end.unsqueeze(1)

    # Set padding to -1
    seq_ids = seq_ids.masked_fill(is_padding, -1)

    def document_mask(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        # Check if tokens belong to same document AND are not padding
        return (seq_ids[b, q_idx] == seq_ids[b, kv_idx]) & (seq_ids[b, q_idx] > 0)

    return document_mask


def get_document_mask_mod(
    batch: torch.Tensor,
    eos_id: int,
    extra_inputs: dict[str, torch.Tensor] | None = None,
) -> _mask_mod_signature:
    """Creates a document mask that prevents attention across document boundaries.

    Args:
        batch: Input batch tensor with shape [b, s, h, d]
        eos_id: End-of-sequence token ID that marks document boundaries
        extra_inputs: Extra inputs to the mask modifier function

    Returns:
        A mask modifier function that implements document-level masking.
    """
    # batch is [b, s, h, d] shape

    # this functioned can be called in two ways:
    # 1. with eos_id: (we use this in the pre-training while eos_id is relible to get the document boundaries)
    # 2. with seq_lengths and seq_start_indices:
    #    - seq_lengths: the length of each sequence
    #    - seq_start_indices: the start index of each sequence
    #    - we use this in the post-training while eos_id is not relible to get the document boundaries

    if extra_inputs is None and eos_id is not None:
        return get_document_mask_mod_using_eos_id(batch, eos_id)
    elif extra_inputs is not None:
        seq_lens = extra_inputs.pop("seq_lens", None)
        seq_start_indices = extra_inputs.pop("seq_start_indices", None)
        assert seq_lens is not None and seq_start_indices is not None
        return get_document_mask_mod_using_sequence_indices(
            batch, seq_lens, seq_start_indices
        )


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


def get_sliding_window_mask_mod(window_size: int) -> _mask_mod_signature:
    """Creates a sliding window mask that only attends to tokens within a fixed window size.

    This implements causal sliding window attention where each token can only attend to:
    - Itself (current token)
    - Up to `window_size - 1` previous tokens
    Args:
        window_size: The maximum number of tokens to attend to (including current token).
                    Must be >= 1. A window_size of 1 means attend only to self.

    Returns:
        A mask modifier function that implements causal sliding window masking.
    """

    if window_size < 1:
        raise ValueError(
            f"window_size must be >= 1 for sliding window attention mask, got {window_size}"
        )

    def sliding_window_mod(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        # Window mask: can only attend within the window
        # q_idx - kv_idx < window_size ensures we look at most window_size-1 tokens back
        return (kv_idx <= q_idx) & (q_idx - kv_idx < window_size)

    sliding_window_mod.__name__ = f"sliding_window_mod_window_size_{window_size}"

    return sliding_window_mod


_compiled_create_block_mask = torch.compile(create_block_mask)


@functools.lru_cache(4)
def create_attention_mask(*args, **kwargs):
    """Create an attention mask using compiled create_block_mask.

    This function is cached to avoid recreating BlockMasks for the same
    arguments.
    """
    return _compiled_create_block_mask(*args, **kwargs)


def create_varlen_metadata_for_document(
    input_batch: torch.Tensor,
    eos_id: int,
    extra_inputs: dict[str, torch.Tensor] | None = None,
) -> _mask_mod_signature:
    """
    Creates an attention mask for document-level attention.
    """
    if extra_inputs is None and eos_id is not None:
        return create_varlen_metadata_for_document_using_eos_id(input_batch, eos_id)
    elif extra_inputs is not None:
        seq_lens = extra_inputs.pop("seq_lens", None)
        seq_start_indices = extra_inputs.pop("seq_start_indices", None)
        assert seq_lens is not None and seq_start_indices is not None
        return create_varlen_metadata_for_document_using_sequence_indices(
            input_batch, seq_lens, seq_start_indices
        )


def create_varlen_metadata_for_document_using_eos_id(
    input_batch: torch.Tensor, eos_id: int
) -> VarlenMetadata:
    """
    Creates cumulative sequence length indices needed for variable length attention

    Args:
        input_batch
        eos_id: the EOS id marker

    Returns:
        VarlenMetadata containing cumulative sequence length indices for q, k, and max_seq_len
    """
    batch_size, seq_len = input_batch.shape
    device = input_batch.device
    cu_seqlens_list, all_seq_lengths = [], []
    offset = 0
    max_seqlen = 0

    for b in range(batch_size):
        tokens = input_batch[b]
        eos_positions = (tokens == eos_id).nonzero(as_tuple=True)[0].to(torch.int32)
        sample_cu_seqlens = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32, device=device),
                eos_positions + 1,
                torch.tensor([seq_len], dtype=torch.int32, device=device),
            ]
        )
        sample_cu_seqlens = torch.unique_consecutive(sample_cu_seqlens)

        seq_lengths = torch.diff(sample_cu_seqlens)
        all_seq_lengths.append(seq_lengths)

        cu_seqlens_adjusted = sample_cu_seqlens[:-1] + offset
        cu_seqlens_list.append(cu_seqlens_adjusted)

        offset += seq_len

    packed_cu_seqlens = torch.cat(
        cu_seqlens_list + [torch.tensor([offset], dtype=torch.int32, device=device)]
    )

    max_seqlen = 0
    if len(all_seq_lengths) > 0:
        all_seq_lengths = torch.cat(all_seq_lengths)
        # device to host sync but only done once per model forward
        max_seqlen = all_seq_lengths.max().item()

    return VarlenMetadata(
        cu_seq_q=packed_cu_seqlens,
        cu_seq_k=packed_cu_seqlens,
        max_q=max_seqlen,
        max_k=max_seqlen,
    )


def create_varlen_metadata_for_document_using_sequence_indices(
    input_batch, seq_lens, seq_start_indices
):
    B, S = input_batch.shape[:2]
    device = input_batch.device

    valid_mask = (seq_start_indices >= 0) & (seq_lens > 0)

    batch_offsets = (torch.arange(B, device=device, dtype=torch.int32) * S).unsqueeze(1)
    global_starts = seq_start_indices.to(torch.int32) + batch_offsets

    flat_starts = global_starts[valid_mask].reshape(-1)

    global_ends = global_starts + seq_lens.to(torch.int32)

    doc_counts = valid_mask.sum(dim=1)
    last_doc_idx = (doc_counts - 1).clamp(min=0)

    last_ends = global_ends.gather(1, last_doc_idx.unsqueeze(1)).reshape(-1)
    last_ends = last_ends[doc_counts > 0]

    batch_boundaries = torch.arange(B + 1, device=device, dtype=torch.int32) * S

    all_points = torch.cat([batch_boundaries, flat_starts, last_ends]).clamp(0, B * S)
    sorted_points, _ = torch.sort(all_points)
    cu = torch.unique_consecutive(sorted_points)

    seg_lens = torch.diff(cu)
    max_seqlen = int(seg_lens.max().item()) if seg_lens.numel() else 0

    return VarlenMetadata(
        cu_seq_q=cu,
        cu_seq_k=cu,
        max_q=max_seqlen,
        max_k=max_seqlen,
    )
