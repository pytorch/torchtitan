# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar, NamedTuple

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.experimental import local_map
from torch.nn.attention import (
    activate_flash_attention_impl,
    current_flash_attention_impl,
    sdpa_kernel,
    SDPBackend,
)
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    _mask_mod_signature,
    _score_mod_signature,
    AuxRequest,
    BlockMask,
    create_block_mask,
    flex_attention,
)
from torch.nn.attention.varlen import varlen_attn

from torchtitan.distributed.utils import is_in_batch_invariant_mode

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.models.common.rope import (
    apply_rotary_emb_complex,
    apply_rotary_emb_cos_sin,
)
from torchtitan.protocols.module import Module


__all__ = [
    "FlexAttention",
    "GQAttention",
    "LocalMapInnerAttention",
    "ScaledDotProductAttention",
    "VarlenAttention",
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
    max_q: int
    max_k: int


AttentionMasksType = dict[str, BlockMask] | BlockMask | VarlenMetadata


class LocalMapInnerAttention(Module):
    """Base class for inner attention with DTensor support.

    When q, k, v are DTensors (e.g., from TP with ``use_local_output=False``),
    overrides ``__call__`` to wrap ``nn.Module.__call__`` with ``local_map``.
    This converts TP DTensors to local **before** any ``forward_pre_hook``
    (e.g., CP's ``sdpa_input_fn``) fires, and wraps outputs back to TP
    DTensors **after** all ``forward_hook``s complete.

    Placements and device mesh are inferred from the input DTensors.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__()
        self._local_map_fn: Callable | None = None

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(q, DTensor):
            assert isinstance(k, DTensor) and isinstance(
                v, DTensor
            ), "q, k, v should all be DTensors"
            # All placements must be Shard. We set
            # out_placements and in_grad_placements equal to
            # in_placements below. This is only valid for attention
            # as qkv are sharded on the n_heads dim. CP is handled
            # independently by _ContextParallel hooks inside
            # nn.Module.__call__.
            assert q.placements == k.placements == v.placements, (
                f"q, k, v must have the same placements, "
                f"but got q={q.placements}, k={k.placements}, v={v.placements}"
            )
            for i, p in enumerate(q.placements):
                assert isinstance(p, Shard), (
                    f"LocalMapInnerAttention requires Shard placements "
                    f"(n_heads dim), but got {p} at position {i}"
                )
            # Ensure all Shard placements use the same tensor dim
            # pyrefly: ignore [missing-attribute]
            shard_dims = {p.dim for p in q.placements}
            assert len(shard_dims) == 1, (
                f"All Shard placements must shard on the same dim, "
                f"but got dims {shard_dims}"
            )
            # return_lse=True (e.g. gpt_oss attention sinks) produces
            # 2 outputs instead of 1, requiring different out_placements.
            return_lse = kwargs.get("return_lse", False)
            out_placements = (
                (q.placements, q.placements) if return_lse else (q.placements,)
            )
            if self._local_map_fn is None:
                self._local_map_fn = local_map(
                    super().__call__,
                    in_placements=(q.placements, k.placements, v.placements),
                    out_placements=out_placements,
                    in_grad_placements=(q.placements, k.placements, v.placements),
                    device_mesh=q.device_mesh,
                )
            # pyrefly: ignore [bad-argument-count]
            return self._local_map_fn(q, k, v, **kwargs)
        return super().__call__(q, k, v, **kwargs)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError


class VarlenAttention(LocalMapInnerAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(LocalMapInnerAttention.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        from torchtitan.tools.utils import has_cuda_capability

        # Hopper (SM 9.0) uses FA3
        if has_cuda_capability(9, 0):
            if current_flash_attention_impl() != "FA3":
                activate_flash_attention_impl("FA3")

    # pyrefly: ignore [bad-param-name-override, bad-override]
    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        *,
        attention_masks: VarlenMetadata,
        scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(
            attention_masks, VarlenMetadata
        ), f"attention_masks must be instance of VarlenMetadata but got {type(attention_masks)}"

        cu_seq_q = attention_masks.cu_seq_q
        cu_seq_k = attention_masks.cu_seq_k
        max_q = attention_masks.max_q
        max_k = attention_masks.max_k

        batch_size, seq_len, _, head_dim = xq.shape

        # varlen attention expects (bs*seqlen, n_heads, head_dim)
        xq_packed = xq.reshape(batch_size * seq_len, -1, head_dim)
        xk_packed = xk.reshape(batch_size * seq_len, -1, head_dim)
        xv_packed = xv.reshape(batch_size * seq_len, -1, head_dim)

        # Some operators can upcast under AMP, but varlen attention currently only
        # supports bf16/fp16 inputs. If this changes, or fp16 training support
        # is added, this may need to be revisited.
        xq_packed = xq_packed.to(torch.bfloat16)
        xk_packed = xk_packed.to(torch.bfloat16)
        xv_packed = xv_packed.to(torch.bfloat16)

        varlen_kwargs = dict()

        if is_in_batch_invariant_mode():
            if current_flash_attention_impl() == "FA3":
                # Fix split count to 1 to prevent non-deterministic split-k
                # reductions that vary with batch composition.
                # Only needed for FA3; FA2 is automatically batch-invariant.
                varlen_kwargs["num_splits"] = 1

        # Forward enable_gqa from GQAttention when Q and KV head counts differ
        if kwargs.get("enable_gqa", False):
            varlen_kwargs["enable_gqa"] = True

        out_packed = varlen_attn(
            xq_packed,
            xk_packed,
            xv_packed,
            cu_seq_q,
            cu_seq_k,
            max_q,  # pyrefly: ignore [bad-argument-type]
            max_k,  # pyrefly: ignore [bad-argument-type]
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
            **varlen_kwargs,  # pyrefly: ignore [bad-argument-type]
        )
        assert isinstance(out_packed, torch.Tensor)
        # Reshape back to the format expected by GQAttention.forward()
        out = out_packed.view(batch_size, seq_len, -1, head_dim)

        return out.to(xq.dtype)


class FlexAttention(LocalMapInnerAttention):
    """Inner attention using ``flex_attention`` with torch.compile and CP support.

    Each backend handles its own layout transpose: ``forward()`` transposes from
    ``(bs, seq, heads, dim)`` to ``(bs, heads, seq, dim)`` before calling
    ``flex_attention``, and transposes back before returning.

    Note:
        The forward function must have q, k, v as the first three arguments
        to be compatible with _ContextParallel.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(LocalMapInnerAttention.Config):
        block_size: int | tuple[int, int] = _DEFAULT_SPARSE_BLOCK_SIZE
        kernel_options: dict = field(default_factory=dict)

    inductor_configs: ClassVar[dict[str, bool]] = {
        "wrap_inductor_compiled_regions": True,
        # Recommended workflow: run once with max_autotune=True to discover
        # good kernel_options, then set kernel_options explicitly in the config
        # and keep max_autotune disabled for faster compilation.
        "max_autotune": True,
        # When enabled, after max_autotune selects the best kernel config,
        # coordinate descent iteratively tunes individual parameters (block
        # sizes, num_warps, num_stages) one at a time -- doubling/halving each
        # and accepting changes that improve runtime by >0.1%. This can also
        # run without max_autotune but starts from a weaker baseline config.
        # See torch/_inductor/runtime/coordinate_descent_tuner.py.
        "coordinate_descent_tuning": True,
        "triton.cudagraphs": False,
    }

    # pyrefly: ignore[no-matching-overload]
    _compiled_flex_attn: ClassVar[Callable] = torch.compile(
        flex_attention,
        options=inductor_configs,
    )

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.kernel_options = config.kernel_options

    # pyrefly: ignore [bad-override]
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_masks: BlockMask | None = None,
        score_mod: _score_mod_signature | None = None,
        scale: float | None = None,
        return_lse: bool = False,
        enable_gqa: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(
            attention_masks, (BlockMask, type(None))
        ), f"attention_masks must be instance of BlockMask or None, got {type(attention_masks)}"

        # Transpose to (bs, heads, seq, dim) for flex_attention
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # 1. _compiled_flex_attn has to be a class variable, otherwise there will
        #    be multiple compiled flex_attention instances, which can be slow.
        # 2. `self._compiled_flex_attn` is not correct, `self` will be passed in
        #    as the first argument, which will cause an error.
        #    `FlexAttention._compiled_flex_attn` is correct.
        out, aux = FlexAttention._compiled_flex_attn(
            q,
            k,
            v,
            block_mask=attention_masks,
            scale=scale,
            enable_gqa=enable_gqa,
            return_aux=AuxRequest(lse=return_lse),
            kernel_options=self.kernel_options,
        )
        # Transpose back to (bs, seq, heads, dim)
        if return_lse:
            return out.transpose(1, 2), aux.lse.transpose(1, 2)
        return out.transpose(1, 2)


class ScaledDotProductAttention(LocalMapInnerAttention):
    """Inner attention using ``F.scaled_dot_product_attention`` with CP support.

    Each backend handles its own layout transpose: ``forward()`` transposes from
    ``(bs, seq, heads, dim)`` to ``(bs, heads, seq, dim)`` before calling
    ``scaled_dot_product_attention``, and transposes back before returning.

    Note:
        The forward function must have q, k, v as the first three arguments to be
        compatible with _ContextParallel.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(LocalMapInnerAttention.Config):
        pass

    sdpa_backends: list[SDPBackend] = []

    def __init__(self, config: Config) -> None:
        if config is None:
            config = ScaledDotProductAttention.Config()
        super().__init__(config)
        if not self.sdpa_backends:
            self.sdpa_backends = [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.MATH,
            ]

    # pyrefly: ignore [bad-override]
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
        enable_gqa: bool = False,
        is_causal: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        # Transpose to (bs, heads, seq, dim) for SDPA
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        with sdpa_kernel(self.sdpa_backends, set_priority=True):
            out = F.scaled_dot_product_attention(
                q, k, v, scale=scale, is_causal=is_causal, enable_gqa=enable_gqa
            )
        # Transpose back to (bs, seq, heads, dim)
        return out.transpose(1, 2)


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

    max_seqlen: int = 0
    if len(all_seq_lengths) > 0:
        all_seq_lengths = torch.cat(all_seq_lengths)
        # device to host sync but only done once per model forward
        # pyrefly: ignore[bad-assignment]
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
        inner_attention: LocalMapInnerAttention.Config
        mask_type: str

        def __post_init__(self):
            assert self.n_heads > 0, "n_heads must be > 0"
            assert isinstance(self.inner_attention, LocalMapInnerAttention.Config), (
                f"inner_attention must be a LocalMapInnerAttention.Config, "
                f"got {type(self.inner_attention)}"
            )
            assert self.mask_type in [
                "causal",
                "block_causal",
            ], f"mask_type must be one of ['causal', 'block_causal'], got {self.mask_type}"
            if (
                isinstance(self.inner_attention, ScaledDotProductAttention.Config)
                and self.mask_type == "block_causal"
            ):
                raise ValueError(
                    "mask_type 'block_causal' is not supported with "
                    "ScaledDotProductAttention"
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
        dim: int
        wq: Linear.Config
        wkv: Linear.Config
        wo: Linear.Config
        qk_norm: RMSNorm.Config | None = None
        n_kv_heads: int | None = None
        head_dim: int | None = None
        use_rope: bool = True
        inner_attention: LocalMapInnerAttention.Config
        mask_type: str = "causal"
        rope_backend: str = "complex"  # "complex" or "cos_sin"

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = (
            config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        )
        self.head_dim = (
            config.head_dim
            if config.head_dim is not None
            else config.dim // config.n_heads
        )
        self.enable_gqa = self.n_heads > self.n_kv_heads
        self.use_rope = config.use_rope
        self.rope_backend = config.rope_backend

        # Optional QK normalization (Qwen3-style)
        self.q_norm: RMSNorm | None = None
        self.k_norm: RMSNorm | None = None
        if config.qk_norm is not None:
            self.q_norm = config.qk_norm.build()
            self.k_norm = config.qk_norm.build()

        # Scaling factor (needed when head_dim differs from dim // n_heads)
        self.scaling = self.head_dim**-0.5 if config.head_dim is not None else None

        self.wq = config.wq.build()
        self.wk = config.wkv.build()
        self.wv = config.wkv.build()
        self.wo = config.wo.build()

        self.inner_attention = config.inner_attention.build()

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
        if self.q_norm is not None or self.k_norm is not None:
            assert self.q_norm is not None and self.k_norm is not None
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        # Apply rotary embeddings
        if self.use_rope:
            if self.rope_backend == "cos_sin":
                xq, xk = apply_rotary_emb_cos_sin(xq, xk, rope_cache, positions)
            else:
                xq, xk = apply_rotary_emb_complex(
                    xq, xk, freqs_cis=rope_cache, positions=positions
                )

        # Handle iRoPE dict masks (Llama4)
        if isinstance(attention_masks, dict):
            mask_key = "rope" if self.use_rope else "nope"
            attention_masks = attention_masks[mask_key]

        output = self.inner_attention(
            xq,
            xk,
            xv,
            attention_masks=attention_masks,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
        ).contiguous()
        output = output.view(bs, seqlen, -1)
        return self.wo(output)
