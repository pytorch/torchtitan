# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   B = batch, L = sequence length, D = model dimension,
#   N = num heads (N is used for both query and kv heads in GQA;
#       the variable name xq/xk/xv disambiguates),
#   H = head dimension (per-head dim),
#   T = packed tokens (B*L, used by VarlenAttention)

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, NamedTuple

import spmd_types as spmd

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Replicate
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
from torch.nn.attention.varlen import AuxRequest as VarlenAuxRequest, varlen_attn

from torchtitan.distributed.compile import maybe_regional_inductor
from torchtitan.distributed.utils import get_spmd_backend, is_in_batch_invariant_mode

from torchtitan.models.common.nn_modules import Linear, RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import Module
from torchtitan.tools.utils import round_up


__all__ = [
    "FlexAttention",
    "BaseQKVLinear",
    "FusedQKVLinear",
    "GQAttention",
    "QKVLinear",
    "ScaledDotProductAttention",
    "VarlenAttention",
    "VarlenMetadata",
    "create_attention_mask",
    "create_varlen_metadata_for_document",
    "get_causal_mask_mod",
    "get_document_mask_mod",
    "get_efficient_causal_mask_mod_for_packed_document",
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


def _resolve_varlen_attention_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return dtype
    return torch.bfloat16


class VarlenAttention(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        window_size: tuple[int, int] = (-1, 0)
        """ window_size=(left, right) controls the attention window relative to each
            query position. 'left' is how many tokens before the query to attend to,
            and 'right' is how many tokens after. A value of -1 means unlimited.

              - (-1, 0): Causal attention - each token attends to all previous tokens
                         and itself, but no future tokens. Equivalent to is_causal=True.
              - (-1, -1): Full bidirectional attention (no masking). Equivalent to
                          is_causal=False.
              - (W, 0): Sliding window causal - attend to at most W previous tokens.
        """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.window_size = config.window_size

        from torchtitan.tools.utils import has_cuda_capability

        # Hopper (SM 9.0) uses FA3
        if has_cuda_capability(9, 0):
            if current_flash_attention_impl() != "FA3":
                activate_flash_attention_impl("FA3")

    def forward(
        self,
        q_BLNH: torch.Tensor,
        k_BLNH: torch.Tensor,
        v_BLNH: torch.Tensor,
        *,
        attention_masks: VarlenMetadata,
        scale: float | None = None,
        out_transform: (
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None
        ) = None,
        **kwargs,
    ) -> torch.Tensor:
        assert isinstance(
            attention_masks, VarlenMetadata
        ), f"attention_masks must be instance of VarlenMetadata but got {type(attention_masks)}"

        cu_seq_q = attention_masks.cu_seq_q
        cu_seq_k = attention_masks.cu_seq_k
        max_q = attention_masks.max_q
        max_k = attention_masks.max_k

        B, L, _, H = q_BLNH.shape
        T = B * L

        # varlen attention expects (T, N, H)
        q_TNH = q_BLNH.reshape(T, -1, H)
        k_TNH = k_BLNH.reshape(T, -1, H)
        v_TNH = v_BLNH.reshape(T, -1, H)

        # Some operators can upcast under AMP, but varlen attention currently only
        # supports bf16/fp16 inputs. Preserve low-precision inputs for fp16
        # training; otherwise fall back to bf16 for the attention kernel.
        attn_dtype = _resolve_varlen_attention_dtype(q_TNH.dtype)
        q_TNH = q_TNH.to(attn_dtype)
        k_TNH = k_TNH.to(attn_dtype)
        v_TNH = v_TNH.to(attn_dtype)

        varlen_kwargs: dict[str, Any] = {}

        # TODO(pytorch/pytorch#179760): FA2's auto num_splits heuristic
        # produces NaN intermittently with paged KV (block_table). Force
        # num_splits=1 as a workaround. current_flash_attention_impl()
        # returns None when FA2 is the implicit default (SM < 9.0).
        # For FA3, only force num_splits=1 in batch-invariant mode
        # to prevent non-deterministic split-k reductions.
        # ROCm's _flash_attention_forward rejects num_splits entirely.
        fa_impl = current_flash_attention_impl()
        if (
            fa_impl in (None, "FA2") or is_in_batch_invariant_mode()
        ) and torch.version.hip is None:
            varlen_kwargs["num_splits"] = 1

        # Forward enable_gqa from GQAttention when Q and KV head counts differ
        if kwargs.get("enable_gqa", False):
            varlen_kwargs["enable_gqa"] = True

        if out_transform is not None:
            varlen_kwargs["return_aux"] = VarlenAuxRequest(lse=True)

        # FA3 varlen attention takes rank-local metadata tensors.
        # TODO(pianpwk): Move this op contract into pytorch/spmd_types.
        with spmd.no_typecheck():
            result = varlen_attn(
                q_TNH,
                k_TNH,
                v_TNH,
                cu_seq_q,
                cu_seq_k,
                max_q,
                max_k,
                scale=scale,
                window_size=self.window_size,
                **varlen_kwargs,
            )

        # varlen_attn returns the packed output (T, N, H), plus the LSE when an
        # out_transform epilogue was requested.
        if out_transform is None:
            assert isinstance(result, torch.Tensor)
            if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
                # exclude CP from typecheck as varlen + CP is not yet supported.
                spmd.assert_type(result, spmd.V, spmd.PartitionSpec("dp", "tp", None))
            out_BLNH = result.view(B, L, -1, H).to(q_BLNH.dtype)
            return out_BLNH

        out_TNH, lse_NT = result
        if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
            spmd.assert_type(out_TNH, spmd.V, spmd.PartitionSpec("dp", "tp", None))
            spmd.assert_type(lse_NT, spmd.V, spmd.PartitionSpec("tp", "dp"))

        out_BLNH = out_TNH.view(B, L, -1, H).to(q_BLNH.dtype)
        # FA varlen returns the LSE as (N, T); reorder to (B, L, N) so
        # out_transform can broadcast per (token, head).
        lse_BLN = lse_NT.transpose(0, 1).reshape(B, L, -1)
        return out_transform(out_BLNH, lse_BLN)


class FlexAttention(Module):
    """Inner attention using ``flex_attention`` with torch.compile and CP support.

    Each backend handles its own layout transpose: ``forward()`` transposes from
    ``(B, L, N, H)`` to ``(B, N, L, H)`` before calling
    ``flex_attention``, and transposes back before returning.

    Note:
        The forward function must have q, k, v as the first three arguments
        to be compatible with _ContextParallel.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
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
        super().__init__()
        self.kernel_options = config.kernel_options

    @staticmethod
    def compiled_flex_attn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        score_mod: _score_mod_signature | None,
        block_mask: BlockMask | None,
        scale: float | None,
        enable_gqa: bool,
        return_aux: AuxRequest,
        kernel_options: dict,
    ):
        """Run compiled FlexAttention outside SPMD typechecking.

        Compiled regions are not currently compatible with SPMD typechecking,
        so propagate types at the boundary instead of typechecking into Flex.
        TODO(pianpwk): Move flex-typechecking into pytorch/spmd_types.
        """
        with spmd.no_typecheck():
            out, aux = FlexAttention._compiled_flex_attn(
                q,
                k,
                v,
                score_mod=score_mod,
                block_mask=block_mask,
                scale=scale,
                enable_gqa=enable_gqa,
                return_aux=return_aux,
                kernel_options=kernel_options,
            )
        if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
            spmd.assert_type(out, spmd.V, spmd.PartitionSpec("dp", "tp", "cp", None))
            if return_aux.lse:
                spmd.assert_type(aux.lse, spmd.V, spmd.PartitionSpec("dp", "tp", "cp"))
        return out, aux

    def forward(
        self,
        q_BLNH: torch.Tensor,
        k_BLNH: torch.Tensor,
        v_BLNH: torch.Tensor,
        *,
        attention_masks: BlockMask,
        score_mod: _score_mod_signature | None = None,
        scale: float | None = None,
        enable_gqa: bool = False,
        # TODO: make this into a config function and during fwd accept kwargs
        out_transform: (
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None
        ) = None,
        **kwargs,
    ) -> torch.Tensor:
        assert isinstance(
            attention_masks, BlockMask
        ), f"attention_masks must be instance of BlockMask, got {type(attention_masks)}"

        # Transpose to (B, N, L, H) for flex_attention
        q_BNLH = q_BLNH.transpose(1, 2)
        k_BNLH = k_BLNH.transpose(1, 2)
        v_BNLH = v_BLNH.transpose(1, 2)

        # 1. _compiled_flex_attn has to be a class variable, otherwise there will
        #    be multiple compiled flex_attention instances, which can be slow.
        # 2. `self._compiled_flex_attn` is not correct, `self` will be passed in
        #    as the first argument, which will cause an error.
        #    `FlexAttention._compiled_flex_attn` is correct.
        # Mark the flex region so that, when the enclosing model is compiled with
        # a non-inductor backend, regional_inductor scoops just this region into
        # an inductor sub-compile (see distributed/compile.py). A null context on
        # the default inductor / eager paths, so no dead metadata is emitted.
        with maybe_regional_inductor(FlexAttention.inductor_configs):
            out_BNLH, aux = FlexAttention.compiled_flex_attn(
                q_BNLH,
                k_BNLH,
                v_BNLH,
                score_mod=score_mod,
                block_mask=attention_masks,
                scale=scale,
                enable_gqa=enable_gqa,
                return_aux=AuxRequest(lse=out_transform is not None),
                kernel_options=self.kernel_options,
            )
        # Transpose back to (B, L, N, H)
        out_BLNH = out_BNLH.transpose(1, 2)
        if out_transform is None:
            return out_BLNH
        lse_BLN = aux.lse.transpose(1, 2)
        return out_transform(out_BLNH, lse_BLN)


class ScaledDotProductAttention(Module):
    """Inner attention using ``F.scaled_dot_product_attention`` with CP support.

    Each backend handles its own layout transpose: ``forward()`` transposes from
    ``(B, L, N, H)`` to ``(B, N, L, H)`` before calling
    ``scaled_dot_product_attention``, and transposes back before returning.

    Note:
        The forward function must have q, k, v as the first three arguments to be
        compatible with _ContextParallel.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    sdpa_backends: list[SDPBackend] = []

    def __init__(self, config: Config) -> None:
        if config is None:
            config = ScaledDotProductAttention.Config()
        super().__init__()
        if not self.sdpa_backends:
            self.sdpa_backends = [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.MATH,
            ]

    def forward(
        self,
        q_BLNH: torch.Tensor,
        k_BLNH: torch.Tensor,
        v_BLNH: torch.Tensor,
        *,
        attention_masks: AttentionMasksType | None = None,
        scale: float | None = None,
        enable_gqa: bool = False,
        is_causal: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if attention_masks is not None:
            raise ValueError(
                "ScaledDotProductAttention does not support attention_masks; it "
                "only supports causal/non-causal attention via is_causal."
            )
        # Transpose to (B, N, L, H) for SDPA
        q_BNLH, k_BNLH, v_BNLH = (
            q_BLNH.transpose(1, 2),
            k_BLNH.transpose(1, 2),
            v_BLNH.transpose(1, 2),
        )
        with sdpa_kernel(self.sdpa_backends, set_priority=True):
            out_BNLH = F.scaled_dot_product_attention(
                q_BNLH,
                k_BNLH,
                v_BNLH,
                scale=scale,
                is_causal=is_causal,
                enable_gqa=enable_gqa,
            )
        # Transpose back to (B, L, N, H)
        return out_BNLH.transpose(1, 2)


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


def get_document_mask_mod(positions: torch.Tensor) -> _mask_mod_signature:
    """Creates a document mask that prevents attention across document boundaries.

    Document boundaries are detected where ``positions`` resets to 0, which
    marks the start of a new packed document.

    Args:
        positions: Per-token position tensor with shape ``[b, s]``. Positions
            reset to 0 at each document start.

    Returns:
        A mask modifier function that implements document-level masking.
    """
    doc_ids = torch.cumsum((positions == 0).int(), dim=1) - 1

    def document_mask(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        return doc_ids[b, q_idx] == doc_ids[b, kv_idx]

    return document_mask


def get_efficient_causal_mask_mod_for_packed_document(
    positions: torch.Tensor,
) -> _mask_mod_signature:
    """Creates an efficient document mask to compose with a causal mask.

    This uses the same convention as get_document_mask_mod: per-token positions
    reset to 0 at each packed document boundary and then increase by 1 within the
    document. It is a manually tuned FlexAttention/FlexFlash fast path for
    causal packed-document masking, which is why it coexists with the generic
    document-id mask.

    For ``batch_size == 1``, the causal mask supplies the upper bound,
    ``kv_idx <= q_idx``, and this mask supplies the lower bound,
    ``doc_start[q_idx] <= kv_idx``. For ``batch_size > 1``, the causal mask
    supplies the lower bound, ``kv_idx <= q_idx``, and this mask supplies the
    upper bound, ``q_idx <= doc_end[kv_idx]``. A query in the next document
    cannot attend to a key in the previous document because that key's
    ``doc_end`` is before the query.

    The result is same-document causal masking. This mask is not intended for
    non-causal use.
    """
    batch_size, seq_len = positions.shape
    if batch_size == 1:
        document_starts = positions[0] == 0
        document_id = torch.cumsum(document_starts.int(), dim=0).to(torch.int32) - 1
        token_idx = torch.arange(seq_len, device=positions.device, dtype=torch.int32)
        offsets = torch.full(
            (round_up(seq_len + 1, 128),),
            seq_len,
            device=positions.device,
            dtype=torch.int32,
        )
        offsets.scatter_(
            0,
            torch.where(
                document_starts, document_id, torch.full_like(document_id, seq_len)
            ).to(torch.int64),
            torch.where(
                document_starts, token_idx, torch.full_like(token_idx, seq_len)
            ),
        )

        def packed_document_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ) -> torch.Tensor:
            return kv_idx >= offsets[document_id[q_idx]]

        return packed_document_mask

    document_starts = positions == 0
    document_id = torch.cumsum(document_starts.int(), dim=1).to(torch.int32) - 1
    token_idx = torch.arange(
        seq_len, device=positions.device, dtype=torch.int32
    ).expand_as(positions)
    next_doc_start = torch.full(
        (batch_size, seq_len), seq_len, device=positions.device, dtype=torch.int32
    )
    next_doc = document_starts & (document_id > 0)
    next_doc_start.scatter_(
        1,
        torch.where(
            next_doc, document_id - 1, torch.full_like(document_id, seq_len - 1)
        ).to(torch.int64),
        torch.where(next_doc, token_idx, torch.full_like(token_idx, seq_len)),
    )
    doc_end_by_token = next_doc_start.gather(1, document_id.to(torch.int64)) - 1

    def packed_document_mask(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        return q_idx <= doc_end_by_token[b, kv_idx]

    return packed_document_mask


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
    positions: torch.Tensor,
) -> VarlenMetadata:
    """Creates cumulative sequence length indices needed for variable length attention.

    Document boundaries are detected where ``positions`` resets to 0 (same
    convention as :func:`get_document_mask_mod`).

    Args:
        positions: Per-token position tensor with shape ``[b, s]``. Positions
            reset to 0 at each document start.

    Returns:
        VarlenMetadata containing cumulative sequence length indices for q, k, and max_seq_len
    """
    batch_size, seq_len = positions.shape
    device = positions.device
    cu_seqlens_list, all_seq_lengths = [], []
    offset = 0

    for b in range(batch_size):
        doc_starts = (positions[b] == 0).nonzero(as_tuple=True)[0].to(torch.int32)
        sample_cu_seqlens = torch.cat(
            [doc_starts, torch.tensor([seq_len], dtype=torch.int32, device=device)]
        )

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
        inner_attention: Module.Config

        def __post_init__(self):
            assert self.n_heads > 0, "n_heads must be > 0"


class BaseQKVLinear(Module):
    """Base class for Q/K/V projection strategies.

    Subclasses implement different projection approaches (separate or fused)
    while providing a uniform interface to :class:`GQAttention`.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        head_dim: int

    def __init__(self, config: Config):
        super().__init__()
        self.head_dim = config.head_dim

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project input into Q, K, V tensors.

        Returns:
            (xq, xk, xv) each with shape [B, L, local_heads, head_dim].
        """
        raise NotImplementedError


class QKVLinear(BaseQKVLinear):
    """Three separate linear projections for Q, K, V."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseQKVLinear.Config):
        wq: Linear.Config
        wkv: Linear.Config

    def __init__(self, config: Config):
        super().__init__(config)
        self.wq = config.wq.build()
        self.wk = config.wkv.build()
        self.wv = config.wkv.build()

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # Use -1 instead of n_heads (or n_kv_heads) to infer the
        # actual local heads from sizes as TP may have sharded them.

        def local_qkv_head_split(x):
            # Drop into local region, we can't propagate S(2) -> qkv head unflatten.
            # TODO(pianpwk): this should be doable once spmd_types tracks sharding evenness.
            with spmd.local():
                x_ = x.view(bs, seqlen, -1, self.head_dim)
                if get_spmd_backend() == "spmd_types":
                    spmd.assert_type(
                        x_, spmd.V, spmd.PartitionSpec("dp", "cp", "tp", None)
                    )
            return x_

        xq, xk, xv = (
            local_qkv_head_split(xq),
            local_qkv_head_split(xk),
            local_qkv_head_split(xv),
        )
        return xq, xk, xv


class FusedQKVLinear(BaseQKVLinear):
    """Single fused linear projection, split along R dimension.

    Uses a single linear layer and splits the output along the R dimension,
    where R = n_heads // n_kv_heads + 2 (Q-heads-per-KV-group + K + V).
    Reduces kernel launch overhead compared to three separate projections.

    Compatible with ColwiseParallel on the ``wqkv`` linear layer.

    Checkpoints in the stock ``QKVLinear`` layout (``wq.weight`` / ``wk.weight`` /
    ``wv.weight``) via state_dict hooks, so checkpoints interoperate with the
    non-fused module and the HF adapter.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseQKVLinear.Config):
        n_heads: int
        n_kv_heads: int
        wqkv: Linear.Config

    def __init__(self, config: Config):
        super().__init__(config)
        if config.n_heads % config.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({config.n_heads}) must be divisible by "
                f"n_kv_heads ({config.n_kv_heads}) for fused QKV"
            )
        self.wqkv = config.wqkv.build()
        self.heads_per_kv = config.n_heads // config.n_kv_heads
        self.r_dim = self.heads_per_kv + 2
        self.register_state_dict_post_hook(self._split_qkv_on_save)
        self.register_load_state_dict_pre_hook(self._merge_qkv_on_load)

    @spmd.local_map(
        out_types=({"dp": spmd.S(0), "cp": spmd.S(1), "tp": spmd.S(2)},) * 3
    )
    def forward(  # pyrefly: ignore[bad-override]
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs, seqlen, _ = x.shape
        # Fused QKV: single matmul, then reshape and split along R dim.
        # [B, L, n_kv_heads * R * head_dim] -> [B, L, n_kv_heads, R, head_dim]
        # Use -1 for n_kv_heads so TP sharding is handled automatically.
        qkv = self.wqkv(x)
        with spmd.local():  # TODO(pianpwk): same QKV:S(2) unflatten case handled by even sharding
            qkv = qkv.view(bs, seqlen, -1, self.r_dim, self.head_dim)
            if get_spmd_backend() == "spmd_types":
                spmd.assert_type(
                    qkv, spmd.V, spmd.PartitionSpec("dp", "cp", "tp", None, None)
                )

        hpk, hd = self.heads_per_kv, self.head_dim

        def _split(t):
            # Use the (possibly local) tensor's own B/L so this is correct both
            # for a plain tensor and inside local_map, where ``t`` is the local
            # (e.g. CP-sharded) shard rather than the global tensor.
            b, s = t.shape[0], t.shape[1]
            xq, xk, xv = torch.split(t, [hpk, 1, 1], dim=-2)
            # split leaves xk/xv as strided views into the fused buffer; vLLM
            # attention/KV-cache kernels read raw memory assuming a contiguous
            # head-major layout, so materialize all three contiguously here.
            return (
                xq.reshape(b, s, -1, hd).contiguous(),
                xk.reshape(b, s, -1, hd).contiguous(),
                xv.reshape(b, s, -1, hd).contiguous(),
            )

        if isinstance(qkv, DTensor):
            # TEMPORARY: run the split on local tensors so its backward (cat)
            # does not mix DTensor and plain grads under CP+PP. The asymmetric
            # q vs k/v paths (RoPE on q/k; CP all-gathers k/v) otherwise feed
            # cat() inconsistent grad types in PP's backward metadata inference.
            # q/k/v reuse qkv's placements (symmetric at the split: TP shards the
            # head axis, CP shards seq). TODO: remove it after spmd_types/full_dtensor
            _split = local_map(
                _split,
                out_placements=(qkv.placements,) * 3,
                in_placements=(qkv.placements,),
                in_grad_placements=(qkv.placements,),
                device_mesh=qkv.device_mesh,
            )
        return _split(qkv)

    @staticmethod
    def _split_qkv_on_save(module, state_dict, prefix, local_metadata) -> None:
        """Split fused ``wqkv`` into stock ``wq``/``wk``/``wv`` (weight and bias)."""
        hd, hpk, r = module.head_dim, module.heads_per_kv, module.r_dim

        for param, ndim in (("weight", 4), ("bias", 3)):
            key = f"{prefix}wqkv.{param}"
            if key not in state_dict:
                continue
            tensor = state_dict.pop(key)
            # Gather to Replicate so the n_kv-leading reshape is local (dim 0
            # unsharded) when a Shard(0) split would not divide n_kv_heads
            # (e.g. dp_shard=8, n_kv_heads=4); stays a DTensor for the copy.
            if isinstance(tensor, DTensor):
                tensor = tensor.redistribute(
                    tensor.device_mesh, [Replicate()] * tensor.device_mesh.ndim
                )
            n_kv = tensor.shape[0] // (r * hd)
            tail = (tensor.shape[1],) if ndim == 4 else ()
            w = tensor.reshape(n_kv, r, hd, *tail)
            state_dict[f"{prefix}wq.{param}"] = (
                w[:, :hpk].reshape(-1, *tail).contiguous()
            )
            state_dict[f"{prefix}wk.{param}"] = (
                w[:, hpk].reshape(-1, *tail).contiguous()
            )
            state_dict[f"{prefix}wv.{param}"] = (
                w[:, hpk + 1].reshape(-1, *tail).contiguous()
            )

    @staticmethod
    def _merge_qkv_on_load(module, state_dict, prefix, *args) -> None:
        """Merge stock ``wq``/``wk``/``wv`` back into fused ``wqkv`` (weight and bias)."""
        hd, hpk = module.head_dim, module.heads_per_kv

        for param, ndim in (("weight", 4), ("bias", 3)):
            keys = [f"{prefix}{w}.{param}" for w in ("wq", "wk", "wv")]
            if not all(k in state_dict for k in keys):
                continue
            wq, wk, wv = (state_dict.pop(k) for k in keys)
            # TODO: check if we could avoid this All-gather
            # Gather to Replicate so the n_kv reshape is local; stays a DTensor so the
            # fused result can be copied into the sharded wqkv param.
            if isinstance(wq, DTensor):
                wq, wk, wv = (
                    t.redistribute(t.device_mesh, [Replicate()] * t.device_mesh.ndim)
                    for t in (wq, wk, wv)
                )
            n_kv = wk.shape[0] // hd
            tail = (wq.shape[1],) if ndim == 4 else ()
            q = wq.reshape(n_kv, hpk, hd, *tail)
            k = wk.reshape(n_kv, 1, hd, *tail)
            v = wv.reshape(n_kv, 1, hd, *tail)
            state_dict[f"{prefix}wqkv.{param}"] = torch.cat([q, k, v], dim=1).reshape(
                -1, *tail
            )


class GQAttention(BaseAttention):
    """Grouped-Query Attention with pluggable Q/K/V projection.

    The QKV projection strategy is determined by the ``qkv_linear`` config field:
    use :class:`QKVLinear` for three independent projections, or
    :class:`FusedQKVLinear` for a single fused projection.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int
        dim: int
        qkv_linear: BaseQKVLinear.Config
        wo: Linear.Config
        qk_norm: RMSNorm.Config | None = None
        n_kv_heads: int | None = None
        head_dim: int | None = None
        inner_attention: Module.Config
        rope: RoPE.Config

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
        self.rope = config.rope.build()

        # Pluggable QKV projection
        self.qkv_linear = config.qkv_linear.build()
        self.wo = config.wo.build()
        self.inner_attention = config.inner_attention.build()

        # Optional QK normalization (Qwen3-style)
        self.q_norm: RMSNorm | None = None
        self.k_norm: RMSNorm | None = None
        if config.qk_norm is not None:
            self.q_norm = config.qk_norm.build()
            self.k_norm = config.qk_norm.build()

        # Scaling factor (needed when head_dim differs from dim // n_heads)
        self.scaling = self.head_dim**-0.5 if config.head_dim is not None else None

    def forward(
        self,
        x_BLD: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, _ = x_BLD.shape
        xq_BLNH, xk_BLNH, xv_BLNH = self.qkv_linear(x_BLD)

        # Optional QK normalization (before RoPE, per Qwen3)
        if self.q_norm is not None or self.k_norm is not None:
            assert self.q_norm is not None and self.k_norm is not None
            xq_BLNH = self.q_norm(xq_BLNH)
            xk_BLNH = self.k_norm(xk_BLNH)

        # Apply rotary embeddings
        xq_BLNH, xk_BLNH = self.rope(xq_BLNH, xk_BLNH, positions)

        # inner_attention returns (B, L, N, H)
        out_BLNH = self.inner_attention(
            xq_BLNH,
            xk_BLNH,
            xv_BLNH,
            attention_masks=attention_masks,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
        ).contiguous()
        out_BLD = out_BLNH.view(B, L, -1)
        return self.wo(out_BLD)
