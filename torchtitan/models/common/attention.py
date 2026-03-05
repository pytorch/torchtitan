# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    _score_mod_signature,
    BlockMask,
    create_block_mask,
    flex_attention,
)
from torch.nn.attention.varlen import varlen_attn
from torch.types import Number

from torchtitan.models.common.rope import (
    apply_rotary_emb_complex,
    apply_rotary_emb_cos_sin,
)
from torchtitan.models.common.utils import trunc_normal_
from torchtitan.protocols.module import Module


__all__ = [
    "FlexAttentionWrapper",
    "GQAttention",
    "ScaledDotProductAttentionWrapper",
    "VarlenAttentionWrapper",
    "VarlenMetadata",
    "create_attention_mask",
    "create_varlen_metadata_for_document",
    "get_causal_mask_mod",
    "get_document_mask_mod",
    "get_fixed_block_mask_mod",
    "get_sliding_window_mask_mod",
]


class VarlenMetadata(NamedTuple):
    """
    Cumulative sequence positions for queries and keys/values.

    """

    cu_seq_q: torch.Tensor
    cu_seq_k: torch.Tensor
    max_q: Number
    max_k: Number


AttentionMasksType = dict[str, BlockMask] | BlockMask | VarlenMetadata


class VarlenAttentionWrapper(torch.nn.Module):
    _compiled_varlen_attn: ClassVar[Callable] = torch.compile(
        varlen_attn, mode="max-autotune-no-cudagraphs"
    )

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        attention_masks: VarlenMetadata,
        scale: float | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        cu_seq_q = attention_masks.cu_seq_q
        cu_seq_k = attention_masks.cu_seq_k
        max_q = attention_masks.max_q
        max_k = attention_masks.max_k

        xq_packed = xq.transpose(1, 2).flatten(0, 1)  # (bs * seqlen, n_heads, head_dim)
        xk_packed = xk.transpose(1, 2).flatten(
            0, 1
        )  # (bs * seqlen, n_kv_heads, head_dim)
        xv_packed = xv.transpose(1, 2).flatten(
            0, 1
        )  # (bs * seqlen, n_kv_heads, head_dim)

        return VarlenAttentionWrapper._compiled_varlen_attn(
            xq_packed,
            xk_packed,
            xv_packed,
            cu_seq_q,
            cu_seq_k,
            max_q,
            max_k,
            scale=scale,
            # window_size=(left, right) controls the attention window relative to each
            # query position. 'left' is how many tokens before the query to attend to,
            # and 'right' is how many tokens after. A value of -1 means unlimited.
            #
            # This replaces the is_causal flag:
            #   - (-1, 0): Causal attention - each token attends to all previous tokens
            #              and itself, but no future tokens. Equivalent to is_causal=True.
            #   - (-1, -1): Full bidirectional attention (no masking). Equivalent to
            #               is_causal=False.
            #   - (W, 0): Sliding window causal - attend to at most W previous tokens.
            window_size=(-1, 0),
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
            # TODO: turn on this after PyTorch fix is landed again
            # https://github.com/pytorch/pytorch/pull/175733.
            "wrap_inductor_compiled_regions": False,
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
        score_mod: _score_mod_signature | None = None,
        block_mask: BlockMask | None = None,
        scale: float | None = None,
        return_lse: bool = False,
        enable_gqa: bool = False,
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
            enable_gqa=enable_gqa,
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

    sdpa_backends: list[SDPBackend] = []

    def __init__(self) -> None:
        super().__init__()
        if not self.sdpa_backends:
            self.sdpa_backends = [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.MATH,
            ]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
        enable_gqa: bool = False,
        is_causal: bool = True,
    ) -> torch.Tensor:
        with sdpa_kernel(self.sdpa_backends, set_priority=True):
            return F.scaled_dot_product_attention(
                q, k, v, scale=scale, is_causal=is_causal, enable_gqa=enable_gqa
            )


def get_causal_mask_mod() -> _mask_mod_signature:
    """Returns a causal mask modifier for flex attention.

    Returns:
        A mask modifier function that implements causal masking.
    """

    def _causal_mask(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        """Causal mask that prevents attention to future tokens."""
        return q_idx >= kv_idx

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


def create_attention_mask(*args, **kwargs):
    """Create an attention mask using compiled create_block_mask."""
    return _compiled_create_block_mask(*args, **kwargs)


def create_varlen_metadata_for_document(
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


class BaseAttention(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        n_heads: int
        attn_backend: str
        attn_mask_type: str

        def __post_init__(self):
            assert self.n_heads > 0, "n_heads must be > 0"
            assert self.attn_backend in [
                "flex",
                "varlen",
                "sdpa",
            ], f"attn_backend must be one of ['flex', 'varlen', 'sdpa'], got {self.attn_backend}"
            assert self.attn_mask_type in [
                "causal",
                "block_causal",
            ], f"attn_mask_type must be one of ['causal', 'block_causal'], got {self.attn_mask_type}"
            if self.attn_backend == "sdpa" and self.attn_mask_type == "block_causal":
                raise ValueError(
                    "attn_mask_type 'block_causal' is not supported with attn_backend 'sdpa'"
                )


class GQAttention(BaseAttention):
    """Grouped-Query Attention module shared across Llama3, Llama4, Qwen3.

    Supports GQA (grouped-query attention) with optional QK normalization,
    optional RoPE (for iRoPE layers), and multiple attention backends
    (flex, varlen, sdpa).

    Config parameters define the attention head structure. Runtime ``dim``
    is passed via ``build(dim=...)``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int
        n_kv_heads: int | None = None
        head_dim: int | None = None
        qk_norm: bool = False
        norm_eps: float = 1e-5
        bias: bool = False
        use_rope: bool = True
        attn_backend: str = "sdpa"
        attn_mask_type: str = "causal"
        rope_backend: str = "complex"  # "complex" or "cos_sin"

    def __init__(self, config: Config, *, dim: int):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = (
            config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        )
        self.head_dim = (
            config.head_dim if config.head_dim is not None else dim // config.n_heads
        )
        self.enable_gqa = self.n_heads > self.n_kv_heads
        self.use_rope = config.use_rope
        self.rope_backend = config.rope_backend

        # Optional QK normalization (Qwen3-style)
        self.q_norm: nn.RMSNorm | None = None
        self.k_norm: nn.RMSNorm | None = None
        if config.qk_norm:
            self.q_norm = nn.RMSNorm(
                self.head_dim, eps=config.norm_eps, elementwise_affine=True
            )
            self.k_norm = nn.RMSNorm(
                self.head_dim, eps=config.norm_eps, elementwise_affine=True
            )

        # Scaling factor (needed when head_dim differs from dim // n_heads)
        self.scaling = self.head_dim**-0.5 if config.head_dim is not None else None

        self.wq = nn.Linear(dim, self.n_heads * self.head_dim, bias=config.bias)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=config.bias)

        self.attn_backend = config.attn_backend
        self.inner_attention: nn.Module
        match self.attn_backend:
            case "flex":
                self.inner_attention = FlexAttentionWrapper()
            case "varlen":
                self.inner_attention = VarlenAttentionWrapper()
            case "sdpa":
                self.inner_attention = ScaledDotProductAttentionWrapper()
            case _:
                raise ValueError(f"Unknown attention type: {self.attn_backend}")

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        # Optional QK normalization (before RoPE, per Qwen3)
        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
            xk = self.k_norm(xk)

        # Apply rotary embeddings
        if self.use_rope:
            if self.rope_backend == "cos_sin":
                xq, xk = apply_rotary_emb_cos_sin(xq, xk, rope_cache, positions)
            else:
                xq, xk = apply_rotary_emb_complex(
                    xq, xk, freqs_cis=rope_cache, positions=positions
                )

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)  # (bs, n_kv_heads, seqlen, head_dim)
        xv = xv.transpose(1, 2)  # (bs, n_kv_heads, seqlen, head_dim)

        scale_kwargs = {"scale": self.scaling} if self.scaling is not None else {}

        match self.attn_backend:
            case "flex":
                # For iRoPE (Llama4), attention_masks may be a dict
                if isinstance(attention_masks, dict):
                    mask_key = "rope" if self.use_rope else "nope"
                    block_mask = attention_masks[mask_key]
                else:
                    assert isinstance(attention_masks, BlockMask), attention_masks
                    block_mask = attention_masks
                output = (
                    self.inner_attention(
                        xq,
                        xk,
                        xv,
                        block_mask=block_mask,
                        enable_gqa=self.enable_gqa,
                        **scale_kwargs,
                    )
                    .transpose(1, 2)
                    .contiguous()
                )
            case "varlen":
                assert isinstance(attention_masks, VarlenMetadata), attention_masks
                output = self.inner_attention(
                    xq, xk, xv, attention_masks, **scale_kwargs
                )
            case "sdpa":
                assert attention_masks is None
                output = (
                    self.inner_attention(
                        xq,
                        xk,
                        xv,
                        enable_gqa=self.enable_gqa,
                        **scale_kwargs,
                    )
                    .transpose(1, 2)
                    .contiguous()
                )
            case _:
                raise ValueError(f"Unknown attention type: {self.attn_backend}")

        output = output.view(bs, seqlen, -1)
        return self.wo(output)

    def init_weights(self, init_std: float = 0.02, **kwargs) -> None:
        for linear in (self.wq, self.wk, self.wv):
            trunc_normal_(linear.weight, mean=0.0, std=0.02)
            if linear.bias is not None:
                trunc_normal_(linear.bias, mean=0.0, std=0.02)
        trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
        if self.wo.bias is not None:
            trunc_normal_(self.wo.bias, mean=0.0, std=init_std)
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        if self.k_norm is not None:
            self.k_norm.reset_parameters()
