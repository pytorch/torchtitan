# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   B = batch, T = packed tokens, L = sequence length,
#   D = model dimension,
#   H = attention heads (H is used for both query and kv heads in GQA;
#       the variable name xq/xk/xv disambiguates),
#   K = query/key head dimension, V = value head dimension.

import dataclasses
import functools
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, NamedTuple, TYPE_CHECKING, TypeAlias

import spmd_types as spmd
import torch
import torch.nn.functional as F
import torch_remat as remat
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard
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
    and_masks,
    AuxRequest,
    BlockMask,
    create_block_mask,
    flex_attention,
)
from torch.nn.attention.varlen import (
    AuxRequest as VarlenAuxRequest,
    varlen_attn as _varlen_attn,
)

from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import Module
from torchtitan.tools.utils import round_up

if TYPE_CHECKING:
    from .kda import KDAAttentionMetadata

logger = logging.getLogger(__name__)

__all__ = [
    "FlexAttentionMetadata",
    "FlexInnerAttention",
    "GQAttention",
    "InnerAttention",
    "PaddedQKVLinear",
    "QKVLinear",
    "ScaledDotProductInnerAttention",
    "SlidingWindowFlexInnerAttention",
    "VarlenInnerAttention",
    "VarlenAttentionMetadata",
    "create_attention_mask",
    "create_varlen_metadata_for_document",
    "get_causal_mask_mod",
    "get_document_mask_mod",
    "get_efficient_causal_mask_mod_for_packed_document",
    "get_fixed_block_mask_mod",
    "get_sliding_window_mask_mod",
    "local_head_split",
    "validate_tp_head_sharding",
]


FlexAttentionMetadata: TypeAlias = BlockMask


class VarlenAttentionMetadata(NamedTuple):
    """
    Cumulative sequence positions for queries and keys/values.

    """

    cu_seq_q: torch.Tensor
    cu_seq_k: torch.Tensor
    max_q: int
    max_k: int

    _OFFSETS_SPMD_TYPE = spmd.SpmdType(
        {
            MeshAxisName.DP: spmd.V,
            MeshAxisName.TP: spmd.R,
        },
        partition_spec=spmd.PartitionSpec(MeshAxisName.DP),
    )

    def annotate_spmd_types(self) -> None:
        """Annotate offsets under the active dense model-parallel mesh."""
        spmd.assert_type(self.cu_seq_q, self._OFFSETS_SPMD_TYPE)
        if self.cu_seq_k is not self.cu_seq_q:
            spmd.assert_type(self.cu_seq_k, self._OFFSETS_SPMD_TYPE)


@spmd.no_typecheck(out_types=spmd.PartitionSpec(("dp", "cp"), "tp", None))
def varlen_attn(*args, **kwargs):
    return _varlen_attn(*args, **kwargs)


@spmd.no_typecheck(
    out_types=(
        spmd.PartitionSpec(("dp", "cp"), "tp", None),
        spmd.PartitionSpec("tp", ("dp", "cp")),
    )
)
def varlen_attn_with_lse(*args, **kwargs):
    return _varlen_attn(*args, return_aux=VarlenAuxRequest(lse=True), **kwargs)


def local_head_split(
    t: torch.Tensor,
    head_dim: int,
    *,
    dp_shard_dim: int = 0,
    cp_shard_dim: int | None = None,
) -> torch.Tensor:
    # TODO(pianpwk): Remove once spmd_types tracks sharding evenness.
    input_type = {"dp": spmd.S(dp_shard_dim), "tp": spmd.S(t.ndim - 1)}
    output_type = {"dp": spmd.S(dp_shard_dim), "tp": spmd.S(t.ndim - 1)}
    if cp_shard_dim is not None:
        input_type["cp"] = spmd.S(cp_shard_dim)
        output_type["cp"] = spmd.S(cp_shard_dim)
    with spmd.local():
        if spmd.is_type_checking():
            spmd.assert_type(t, input_type)
        out = t.view(*t.shape[:-1], -1, head_dim)
        if spmd.is_type_checking():
            spmd.assert_type(out, output_type)
    return out


class InnerAttention(Module):
    """Base class for attention kernels used by outer attention modules."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        def build_attention_metadata(
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> "FlexAttentionMetadata | VarlenAttentionMetadata | KDAAttentionMetadata | None":
            """Build metadata consumed by this inner attention, if any.

            Inner attentions that do not require metadata inherit the default
            ``None`` result.
            """
            del positions, padding_mask, max_num_documents, max_context_length
            return None

    def __init__(self) -> None:
        super().__init__()
        # SimpleFSDP may replace the runtime class; preserve the backend key.
        self.attention_metadata_key: type[InnerAttention] = type(self)


class VarlenInnerAttention(InnerAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
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

        def build_attention_metadata(
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> VarlenAttentionMetadata:
            """Build packed-sequence metadata consumed by Varlen attention."""
            return create_varlen_metadata_for_document(
                positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.window_size = config.window_size

        from torchtitan.tools.utils import get_cuda_flash_attention_impl

        flash_attention_impl = get_cuda_flash_attention_impl()
        if (
            flash_attention_impl is not None
            and current_flash_attention_impl() != flash_attention_impl
        ):
            activate_flash_attention_impl(flash_attention_impl)

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        attention_metadata: VarlenAttentionMetadata,
        scale: float | None = None,
        out_transform: (
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None
        ) = None,
        **kwargs,
    ) -> torch.Tensor:
        assert isinstance(
            attention_metadata, VarlenAttentionMetadata
        ), f"attention_metadata must be instance of VarlenAttentionMetadata but got {type(attention_metadata)}"

        cu_seq_q = attention_metadata.cu_seq_q
        cu_seq_k = attention_metadata.cu_seq_k
        max_q = attention_metadata.max_q
        max_k = attention_metadata.max_k

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

        varlen_attn_fn = varlen_attn if out_transform is None else varlen_attn_with_lse

        result = varlen_attn_fn(
            q_THK.to(torch.bfloat16),
            k_THK.to(torch.bfloat16),
            v_THV.to(torch.bfloat16),
            cu_seq_q,
            cu_seq_k,
            max_q,
            max_k,
            scale=scale,
            window_size=self.window_size,
            **varlen_kwargs,
        )

        # varlen_attn returns the packed output (T, H, V), plus the LSE when an
        # out_transform epilogue was requested.
        if out_transform is None:
            assert isinstance(result, torch.Tensor)
            return result.to(q_THK.dtype)

        out_THV, lse_HT = result
        out_THV = out_THV.to(q_THK.dtype)
        lse_TH = lse_HT.transpose(0, 1)
        return out_transform(out_THV, lse_TH)


class FlexInnerAttention(InnerAttention):
    """Inner attention using ``flex_attention`` with torch.compile.

    Query/key inputs use ``[T, H, K]`` and value inputs use ``[T, H, V]``.
    The FlexInnerAttention kernel requires a batch dimension, so inputs are adapted
    to ``[1, H, T, K]`` and ``[1, H, T, V]`` only at the kernel boundary.

    Note:
        The forward function must have q, k, v as the first three arguments
        to be compatible with _ContextParallel.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        block_size: int | tuple[int, int] = _DEFAULT_SPARSE_BLOCK_SIZE
        kernel_options: dict = field(default_factory=dict)

        def build_attention_metadata(
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> FlexAttentionMetadata:
            """Build the standard document-causal FlexAttention BlockMask."""
            del padding_mask, max_num_documents, max_context_length
            seq_len = positions.shape[0]
            return create_attention_mask(
                and_masks(
                    get_causal_mask_mod(),
                    get_efficient_causal_mask_mod_for_packed_document(positions),
                ),
                1,
                None,
                seq_len,
                seq_len,
                device=positions.device,
                BLOCK_SIZE=self.block_size,
                # when separate_full_blocks = True, kernel iterates through
                # full blocks first (blocks where all elements are unmasked)
                # but which blocks are "full" vs "partial" changes depending
                # on the particular batch
                # for batch invariance, we disable this optimization
                separate_full_blocks=not is_in_batch_invariant_mode(),
            )

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

    def _get_aux_request(self, *, return_lse: bool) -> AuxRequest:
        """Return the auxiliary outputs needed from this attention call."""
        return AuxRequest(lse=return_lse)

    def _process_aux(self, aux: Any) -> None:
        """Consume auxiliary outputs requested by ``_get_aux_request``.

        For example, a subclass may request ``max_scores`` and accumulate
        per-head attention maxima across forwards. The base implementation is
        a no-op.
        """
        pass

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
        """Run compiled FlexInnerAttention outside SPMD typechecking.

        Compiled regions are not currently compatible with SPMD typechecking, so
        the opaque kernel output is re-typed at the boundary instead of
        typechecking into Flex. Attention preserves the query's sharding (output
        is (1, H, T, V) with the same kernel-batch/head/token layout as ``q``),
        so ``out``
        takes ``q``'s full SPMD type (local type + shard-dim PartitionSpec), and
        ``lse`` takes the same minus the trailing (unsharded) head dim.
        TODO(pianpwk): Move flex-typechecking into pytorch/spmd_types.
        """
        with spmd.no_typecheck():
            out, aux = FlexInnerAttention._compiled_flex_attn(
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
        if spmd.is_type_checking():
            q_local = spmd.get_local_type(q)
            q_ps = spmd.get_partition_spec(q)
            spmd.assert_type(out, q_local, q_ps)
            # Aux outputs are (1, H, T) = q minus the trailing head dim.
            aux_ps = None if q_ps is None else spmd.PartitionSpec(*q_ps[:-1])
            if return_aux.lse:
                spmd.assert_type(aux.lse, q_local, aux_ps)
            if return_aux.max_scores:
                spmd.assert_type(aux.max_scores, q_local, aux_ps)
        return out, aux

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        attention_metadata: FlexAttentionMetadata,
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
            attention_metadata, BlockMask
        ), f"attention_metadata must be instance of BlockMask, got {type(attention_metadata)}"

        q_1HTK = q_THK.transpose(0, 1).unsqueeze(0)
        k_1HTK = k_THK.transpose(0, 1).unsqueeze(0)
        v_1HTV = v_THV.transpose(0, 1).unsqueeze(0)
        aux_request = self._get_aux_request(return_lse=out_transform is not None)

        # 1. _compiled_flex_attn has to be a class variable, otherwise there will
        #    be multiple compiled flex_attention instances, which can be slow.
        # 2. `self._compiled_flex_attn` is not correct, `self` will be passed in
        #    as the first argument, which will cause an error.
        #    `FlexInnerAttention._compiled_flex_attn` is correct.
        out_1HTV, aux = FlexInnerAttention.compiled_flex_attn(
            q_1HTK,
            k_1HTK,
            v_1HTV,
            score_mod=score_mod,
            block_mask=attention_metadata,
            scale=scale,
            enable_gqa=enable_gqa,
            return_aux=aux_request,
            kernel_options=self.kernel_options,
        )
        self._process_aux(aux)
        out_THV = out_1HTV.squeeze(0).transpose(0, 1)
        if out_transform is None:
            return out_THV
        lse_TH = aux.lse.squeeze(0).transpose(0, 1)
        return out_transform(out_THV, lse_TH)


class SlidingWindowFlexInnerAttention(FlexInnerAttention):
    """FlexAttention backend with a causal sliding-window mask."""

    @dataclass(kw_only=True, slots=True)
    class Config(FlexInnerAttention.Config):
        window_size: int

        def build_attention_metadata(
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> FlexAttentionMetadata:
            """Build the document-causal sliding-window FlexAttention BlockMask."""
            del padding_mask, max_num_documents, max_context_length
            seq_len = positions.shape[0]
            return create_attention_mask(
                and_masks(
                    get_causal_mask_mod(),
                    get_efficient_causal_mask_mod_for_packed_document(positions),
                    get_sliding_window_mask_mod(self.window_size),
                ),
                1,
                None,
                seq_len,
                seq_len,
                device=positions.device,
                BLOCK_SIZE=self.block_size,
                # when separate_full_blocks = True, kernel iterates through
                # full blocks first (blocks where all elements are unmasked)
                # but which blocks are "full" vs "partial" changes depending
                # on the particular batch
                # for batch invariance, we disable this optimization
                separate_full_blocks=not is_in_batch_invariant_mode(),
            )


# TODO: Verify whether SDPA support can be removed without losing performance
# after folding: https://github.com/pytorch/torchtitan/pull/4218#pullrequestreview-4977638012
class ScaledDotProductInnerAttention(InnerAttention):
    """Inner attention using ``F.scaled_dot_product_attention`` with CP support.

    ``forward()`` adapts Q/K from ``(B, L, H, K)`` to ``(B, H, L, K)`` and V
    from ``(B, L, H, V)`` to ``(B, H, L, V)``, then converts the result back to
    ``(B, L, H, V)``.

    Note:
        The forward function must have q, k, v as the first three arguments to be
        compatible with _ContextParallel.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        pass

    sdpa_backends: list[SDPBackend] = []

    def __init__(self, config: Config) -> None:
        if config is None:
            config = ScaledDotProductInnerAttention.Config()
        super().__init__()
        if not self.sdpa_backends:
            self.sdpa_backends = [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.MATH,
            ]

    def forward(
        self,
        q_BLHK: torch.Tensor,
        k_BLHK: torch.Tensor,
        v_BLHV: torch.Tensor,
        *,
        attention_metadata: None = None,
        scale: float | None = None,
        enable_gqa: bool = False,
        is_causal: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if attention_metadata is not None:
            raise ValueError(
                "ScaledDotProductInnerAttention does not support attention_metadata; it "
                "only supports causal/non-causal attention via is_causal."
            )
        q_BHLK, k_BHLK, v_BHLV = (
            q_BLHK.transpose(1, 2),
            k_BLHK.transpose(1, 2),
            v_BLHV.transpose(1, 2),
        )
        with sdpa_kernel(self.sdpa_backends, set_priority=True):
            out_BHLV = F.scaled_dot_product_attention(
                q_BHLK,
                k_BHLK,
                v_BHLV,
                scale=scale,
                is_causal=is_causal,
                enable_gqa=enable_gqa,
            )
        return out_BHLV.transpose(1, 2)


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
        positions: Per-token position tensor with shape ``[T]``. Positions
            reset to 0 at each document start.

    Returns:
        A mask modifier function that implements document-level masking.
    """
    doc_ids = torch.cumsum((positions == 0).int(), dim=0) - 1

    def document_mask(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        return doc_ids[q_idx] == doc_ids[kv_idx]

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

    The causal mask supplies the upper bound, ``kv_idx <= q_idx``, and this
    mask supplies the lower bound, ``doc_start[q_idx] <= kv_idx``.

    The result is same-document causal masking. This mask is not intended for
    non-causal use.
    """
    seq_len = positions.shape[0]
    document_starts = positions == 0
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
        torch.where(document_starts, token_idx, torch.full_like(token_idx, seq_len)),
    )

    def packed_document_mask(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        return kv_idx >= offsets[document_id[q_idx]]

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
    *,
    padding_mask: torch.Tensor | None = None,
    max_num_documents: int | None = None,
    max_context_length: int | None = None,
) -> VarlenAttentionMetadata:
    """Creates cumulative sequence length indices needed for variable length attention.

    Document boundaries are detected where ``positions`` resets to 0 (same
    convention as :func:`get_document_mask_mod`).

    Args:
        positions: Per-token position tensor with shape ``[T]``. Positions
            reset to 0 at each document start.
        padding_mask: Per-token boolean tensor that is true for padding. This
            distinguishes padding position resets from real document starts so
            their fixed metadata capacity can be reserved separately.
        max_num_documents: Upper bound on non-padding document segments in the
            local token batch. When set, the device offsets have a fixed shape
            as required for CUDA graph capture. Padding segments are reserved
            separately when ``padding_mask`` is provided.
        max_context_length: Maximum length of one document segment. Required
            with ``max_num_documents`` so the fixed-shape metadata can avoid a
            device-to-host synchronization.

    Returns:
        VarlenAttentionMetadata containing cumulative sequence length indices for q, k,
        and max_seq_len.
    """
    num_tokens = positions.shape[0]
    device = positions.device

    real_doc_starts = positions == 0
    padding_doc_starts = None
    if padding_mask is None:
        is_doc_start = real_doc_starts
    else:
        padding_mask = padding_mask.to(torch.bool)
        real_doc_starts = real_doc_starts & ~padding_mask
        padding_doc_starts = (positions == 0) & padding_mask
        is_doc_start = real_doc_starts | padding_doc_starts

    if max_num_documents is not None:
        if max_context_length is None:
            raise ValueError(
                "max_context_length is required when max_num_documents is set"
            )

        max_num_padding_segments = (
            (num_tokens + max_context_length - 1) // max_context_length
            if padding_mask is not None
            else 0
        )
        max_num_segments = max_num_documents + max_num_padding_segments
        num_slots = max_num_segments + 1
        slot = torch.cumsum(is_doc_start, 0) - 1
        scatter_index = torch.where(
            is_doc_start & (slot < max_num_segments),
            slot,
            torch.full_like(slot, num_slots),
        )
        packed_cu_seqlens = torch.full(
            (num_slots + 1,), num_tokens, dtype=torch.int32, device=device
        )
        packed_cu_seqlens.scatter_(
            0,
            scatter_index,
            torch.arange(num_tokens, dtype=torch.int32, device=device),
        )
        torch._assert_async(real_doc_starts.sum() <= max_num_documents)
        if padding_doc_starts is not None:
            torch._assert_async(padding_doc_starts.sum() <= max_num_padding_segments)
        packed_cu_seqlens = packed_cu_seqlens[:num_slots]
        max_seqlen = max_context_length
    else:
        doc_starts = is_doc_start.nonzero(as_tuple=True)[0].to(torch.int32)
        packed_cu_seqlens = torch.cat(
            [
                doc_starts,
                torch.tensor([num_tokens], dtype=torch.int32, device=device),
            ]
        )
        seq_lengths = torch.diff(packed_cu_seqlens)

        if seq_lengths.numel() > 0:
            # device to host sync but only done once per model forward
            max_seqlen = int(seq_lengths.max().item())
        else:
            max_seqlen = 0

    if spmd.is_type_checking():
        # Packed document boundaries are rank-local ragged metadata, so they
        # vary across DP ranks even when construction initially infers R.
        spmd.mutate_type(packed_cu_seqlens, "dp", src=spmd.R, dst=spmd.V)
    return VarlenAttentionMetadata(
        cu_seq_q=packed_cu_seqlens,
        cu_seq_k=packed_cu_seqlens,
        max_q=max_seqlen,
        max_k=max_seqlen,
    )


class BaseAttention(Module):
    inner_attention: InnerAttention

    @property
    def attention_metadata_key(self) -> type[InnerAttention]:
        return self.inner_attention.attention_metadata_key

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        n_heads: int
        inner_attention: Module.Config

        def __post_init__(self):
            assert self.n_heads > 0, "n_heads must be > 0"


class QKVLinear(Module):
    """Single fused linear projection, split along R dimension.

    Uses a single linear layer and splits the output along the R dimension,
    where R = n_heads // n_kv_heads + 2 (Q-heads-per-KV-group + K + V).
    Reduces kernel launch overhead compared to three separate projections.

    Compatible with ColwiseParallel on the ``wqkv`` linear layer.

    Native state dicts retain the physical ``wqkv`` parameter. Hugging Face
    state-dict adapters split and merge the logical Q/K/V projections at the
    external checkpoint boundary.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        head_dim: int
        n_heads: int
        n_kv_heads: int
        wqkv: Linear.Config

        def validate_tp_degree(self, tp: int, *, hint: str = "") -> None:
            """Raise if ``tp`` cannot shard whole KV-head groups."""
            if self.n_heads % tp != 0:
                raise ValueError(
                    f"tensor parallel degree ({tp}) must divide "
                    f"n_heads ({self.n_heads}).{hint}"
                )
            if self.n_kv_heads % tp != 0:
                raise ValueError(
                    f"tensor parallel degree ({tp}) must divide "
                    f"n_kv_heads ({self.n_kv_heads}).{hint}"
                )

    def __init__(self, config: Config):
        super().__init__()
        self.head_dim = config.head_dim
        # Head counts the projection actually allocates; variants that pad
        # heads for parallelism report the padded counts.
        self.num_padded_q_heads = config.n_heads
        self.num_padded_kv_heads = config.n_kv_heads
        if config.n_heads % config.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({config.n_heads}) must be divisible by "
                f"n_kv_heads ({config.n_kv_heads}) for fused QKV"
            )
        self.wqkv = config.wqkv.build()
        self.heads_per_kv = config.n_heads // config.n_kv_heads
        self.r_dim = self.heads_per_kv + 2

    def build_output_projection(self, wo: Linear.Config) -> Linear:
        """Build the attention output projection that consumes this layer's heads.

        Variants that pad heads override this to size and initialize ``wo``
        for the padded heads.
        """
        return wo.build()

    @spmd.local_map(
        out_types=(
            (
                {"dp": spmd.V, "cp": spmd.V, "tp": spmd.V},
                spmd.PartitionSpec(("dp", "cp"), "tp", None),
            ),
        )
        * 3
    )
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Fused QKV: single matmul, then reshape and split along R dim.
        # [T, n_kv_heads * R * head_dim] -> [T, n_kv_heads, R, head_dim]
        # Use -1 for n_kv_heads so TP sharding is handled automatically.
        qkv = self.wqkv(x)
        # The split below copies the wqkv projection output with bare ops.
        remat.recompute_needs_tensor(qkv)
        num_tokens = qkv.shape[0]
        with spmd.local():  # TODO(pianpwk): same QKV:S(1) unflatten case handled by even sharding
            qkv = qkv.view(num_tokens, -1, self.r_dim, self.head_dim)
            if spmd.is_type_checking():
                spmd.assert_type(
                    qkv,
                    spmd.V,
                    spmd.PartitionSpec(("dp", "cp"), "tp", None, None),
                )

        local_num_tokens = qkv.shape[0]
        xq, xk, xv = torch.split(qkv, [self.heads_per_kv, 1, 1], dim=-2)
        # split leaves xk/xv as strided views into the fused buffer; vLLM
        # attention/KV-cache kernels read raw memory assuming a contiguous
        # head-major layout, so materialize all three contiguously here.
        return (
            xq.reshape(local_num_tokens, -1, self.head_dim).contiguous(),
            xk.reshape(local_num_tokens, -1, self.head_dim).contiguous(),
            xv.reshape(local_num_tokens, -1, self.head_dim).contiguous(),
        )


class PaddedQKVLinear(QKVLinear):
    """``QKVLinear`` that pads its heads so the TP degree divides ``n_kv_heads``.

    The fused QKV projection shards whole KV-head groups across TP ranks, so
    ``n_kv_heads`` must be a multiple of the TP degree. When it is not, append
    zero-initialized KV-head groups up to the next multiple and grow ``n_heads``
    by the same factor, keeping ``n_heads // n_kv_heads`` fixed. The config
    keeps the unpadded counts; the padded counts are
    ``num_padded_q_heads`` and ``num_padded_kv_heads``.

    Padding groups are the trailing groups of ``wqkv`` (whose rows are laid out
    as ``(n_kv_heads, heads_per_kv + 2, head_dim)``) and the matching trailing
    input columns of ``wo``, which :meth:`build_output_projection` pads. With
    their weights at zero, padded Q/K/V are zero, so padded heads produce zero
    attention output and receive exactly zero gradients; they stay zero under
    any optimizer that maps a zero gradient to a zero update (SGD, AdamW,
    Muon). Under elementwise optimizers (SGD, AdamW) the real heads therefore
    train exactly as in the unpadded attention, at the cost of the padded
    heads' compute and memory. Optimizers whose update depends on the matrix
    shape differ: Muon's aspect-ratio learning-rate adjustment sees the padded
    ``wqkv`` rows and ``wo`` columns, which changes the effective learning rate
    of the real heads unless Muon splits the update per head.

    State-dict hooks hide the padding, so DCP and Hugging Face checkpoints hold
    the unpadded shapes and load across TP degrees. The padded modules also
    expose ``padded_params``, which ``OptimizersContainer`` uses to hide the
    padding in optimizer states that share their parameter's shape (Adam,
    AdamW, SGD) and in EMA copies. Padded heads never receive a gradient, so
    their states are zero and restoring them with zeros is exact.

    Parameter counts and linear FLOPs (``6 * nparams``) include the padded
    heads, which the hardware computes, while attention-op FLOPs use the
    unpadded ``n_heads``.
    """

    @dataclass(frozen=True)
    class PaddedDim:
        """One padded dim of a parameter, and its unpadded and padded sizes.

        Real entries come first, so the padding is the trailing slice.
        """

        dim: int
        num_real: int
        num_padded: int

        def strip(self, tensor: torch.Tensor) -> torch.Tensor:
            """Drop the trailing padding from a padded-shape ``tensor``."""
            dim, length = self.dim, self.num_real
            if tensor.shape[dim] == length:
                return tensor
            if not isinstance(tensor, DTensor):
                return tensor.narrow(dim, 0, length)
            # Narrow a replicated copy: the padding boundary need not align
            # with the shards. Then reshard with plain Shard placements, which
            # unlike _StridedShard can express the uneven unpadded shards.
            # The result owns compact local shards, so the cached state dicts of
            # the checkpointer do not keep the gathered tensor alive.
            # TODO: This all-gathers each padded tensor on every state_dict()
            # call (checkpoint saves, RL weight pushes). A shard-local strip
            # would avoid it: only ranks whose shards cross the padding boundary
            # need to exchange data.
            mesh = tensor.device_mesh
            # _StridedShard is not a Shard subclass, so it needs its own case.
            placements = [
                Shard(p.dim) if isinstance(p, (Shard, _StridedShard)) else p
                for p in tensor.placements
            ]
            replicated = tensor.redistribute(mesh, [Replicate()] * mesh.ndim)
            narrowed = replicated.narrow(dim, 0, length)
            assert isinstance(narrowed, DTensor)
            return narrowed.redistribute(mesh, placements)

        def restore(self, tensor: torch.Tensor) -> torch.Tensor:
            """Zero-pad an unpadded ``tensor`` back to the padded size.

            Tensors that already have the padded size are returned unchanged.
            """
            dim, length = self.dim, self.num_padded
            if tensor.shape[dim] == length:
                return tensor
            # The default load copies the replicated result into each rank's
            # shard.
            if isinstance(tensor, DTensor):
                tensor = tensor.redistribute(
                    tensor.device_mesh, [Replicate()] * tensor.device_mesh.ndim
                )
            padding_shape = list(tensor.shape)
            padding_shape[dim] = length - tensor.shape[dim]
            return torch.cat([tensor, tensor.new_zeros(padding_shape)], dim=dim)

    @dataclass(kw_only=True, slots=True)
    class Config(QKVLinear.Config):
        def validate_tp_degree(self, tp: int, *, hint: str = "") -> None:
            # Any TP degree works: the heads are padded when built.
            pass

    def __init__(self, config: Config):
        from torchtitan.distributed.spmd_types import spmd_mesh_size

        tp = spmd_mesh_size("tp")
        head_dim = config.head_dim
        heads_per_kv = config.n_heads // config.n_kv_heads
        padded_n_kv_heads = round_up(config.n_kv_heads, tp)
        padded_n_heads = heads_per_kv * padded_n_kv_heads
        padded_config = config
        if padded_n_kv_heads != config.n_kv_heads:
            _warn_tp_head_padding(
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                padded_n_heads=padded_n_heads,
                padded_n_kv_heads=padded_n_kv_heads,
                tp=tp,
            )
            # Real heads come first along each padded dim: rows of wqkv, input
            # columns of wo. Biases are padded only along wqkv's output rows.
            padded_config = dataclasses.replace(
                config,
                n_heads=padded_n_heads,
                n_kv_heads=padded_n_kv_heads,
                wqkv=dataclasses.replace(
                    config.wqkv,
                    out_features=padded_n_kv_heads * (heads_per_kv + 2) * head_dim,
                    param_init=self._zero_padded_param_init(
                        config.wqkv,
                        pad_dims={"weight": 0, "bias": 0},
                        num_real=config.n_kv_heads * (heads_per_kv + 2) * head_dim,
                    ),
                ),
            )
        super().__init__(padded_config)
        assert self.num_padded_kv_heads % tp == 0
        self._num_q_heads = config.n_heads
        if padded_config is not config:
            rows = self.PaddedDim(
                dim=0,
                num_real=config.wqkv.out_features,
                num_padded=padded_config.wqkv.out_features,
            )
            self._register_padding_hooks(self.wqkv, {"weight": rows, "bias": rows})

    def build_output_projection(self, wo: Linear.Config) -> Linear:
        if self.num_padded_q_heads == self._num_q_heads:
            return wo.build()
        num_real_columns = wo.in_features
        num_padded_columns = self.num_padded_q_heads * self.head_dim
        output_projection = dataclasses.replace(
            wo,
            in_features=num_padded_columns,
            param_init=self._zero_padded_param_init(
                wo, pad_dims={"weight": 1}, num_real=num_real_columns
            ),
        ).build()
        self._register_padding_hooks(
            output_projection,
            {
                "weight": self.PaddedDim(
                    dim=1, num_real=num_real_columns, num_padded=num_padded_columns
                )
            },
        )
        return output_projection

    @staticmethod
    def _zero_padded_param_init(
        linear: Linear.Config, *, pad_dims: dict[str, int], num_real: int
    ) -> dict[str, Callable]:
        """Wrap ``linear``'s initializers to zero the trailing padded slices.

        ``pad_dims`` maps each padded parameter name to its padded dim, and
        ``num_real`` is the unpadded size along that dim. Each wrapped
        initializer runs the original on the real (unpadded) shape and copies
        it, followed by zeros, into the padded parameter.
        """
        param_init = linear.param_init
        if param_init is None or "weight" not in param_init:
            raise ValueError(
                "Padded parameters require an explicit projection weight "
                "initializer so the padding can be zero-initialized."
            )
        if linear.bias and "bias" in pad_dims and "bias" not in param_init:
            raise ValueError(
                "Padded parameters require an explicit projection bias "
                "initializer so the padding can be zero-initialized."
            )

        def zero_padded(init: Callable, dim: int) -> Callable:
            def _init(param: torch.Tensor) -> None:
                # As in fused_qkv_param_init: for a sharded DTensor ``param``,
                # ``new_empty`` returns a Replicate DTensor of the full real
                # shape, so ``init`` draws parallelism-independent values and
                # ``copy_`` keeps each rank's shard.
                real_shape = list(param.shape)
                real_shape[dim] = num_real
                real = param.new_empty(real_shape)
                init(real)
                padding_shape = list(param.shape)
                padding_shape[dim] -= num_real
                with torch.no_grad():
                    param.copy_(
                        torch.cat([real, real.new_zeros(padding_shape)], dim=dim)
                    )

            return _init

        return {
            name: zero_padded(init, pad_dims[name]) if name in pad_dims else init
            for name, init in param_init.items()
        }

    @staticmethod
    def _register_padding_hooks(
        module: Module, padded_params: dict[str, "PaddedQKVLinear.PaddedDim"]
    ) -> None:
        """Hide the trailing padding of ``module``'s parameters from its state dict.

        Saving drops the padding. Loading zero-pads unpadded tensors, and
        tensors that already have the padded shape (checkpoints saved before the
        padding was hidden) load unchanged. ``module.padded_params`` records
        the padding so other state, such as optimizer states, can hide it too.
        """

        def strip_padding(
            module: Module, state_dict: dict[str, Any], prefix: str, local_metadata: Any
        ) -> None:
            for name, padded_dim in padded_params.items():
                if prefix + name in state_dict:
                    state_dict[prefix + name] = padded_dim.strip(
                        state_dict[prefix + name]
                    )

        def restore_padding(
            module: Module, state_dict: dict[str, Any], prefix: str, *args: Any
        ) -> None:
            for name, padded_dim in padded_params.items():
                if prefix + name in state_dict:
                    state_dict[prefix + name] = padded_dim.restore(
                        state_dict[prefix + name]
                    )

        # nn.Module only types Module and Tensor attributes; this is a plain dict.
        module.padded_params = padded_params  # pyrefly: ignore [bad-argument-type]
        module.register_state_dict_post_hook(strip_padding)
        module.register_load_state_dict_pre_hook(restore_padding)


@functools.cache
def _warn_tp_head_padding(
    *,
    n_heads: int,
    n_kv_heads: int,
    padded_n_heads: int,
    padded_n_kv_heads: int,
    tp: int,
) -> None:
    # Cached: every layer with the same geometry would repeat the warning.
    logger.warning(
        f"Padding attention heads to a multiple of the TP degree ({tp}): "
        f"n_kv_heads {n_kv_heads} -> {padded_n_kv_heads}, n_heads {n_heads} -> "
        f"{padded_n_heads}. Padded heads are zero-initialized and inert, but "
        "add attention compute and memory."
    )


class GQAttention(BaseAttention):
    """Grouped-Query Attention with a fused Q/K/V projection.

    ``rope=None`` selects NoPE (no positional encoding) for this layer: q/k go to
    the inner attention unrotated, and positional information reaches the layer
    only through the attention mask. Interleaved RoPE/NoPE models (iRoPE) set it
    per layer.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int
        dim: int
        qkv_linear: QKVLinear.Config
        wo: Linear.Config
        qk_norm: RMSNorm.Config | None = None
        n_kv_heads: int | None = None
        head_dim: int | None = None
        inner_attention: Module.Config
        rope: RoPE.Config | None

        def __post_init__(self) -> None:
            BaseAttention.Config.__post_init__(self)
            if self.head_dim is None and self.dim % self.n_heads != 0:
                raise ValueError(
                    f"dim ({self.dim}) must be divisible by n_heads "
                    f"({self.n_heads}) when head_dim is not specified"
                )

            n_kv_heads = self.n_heads if self.n_kv_heads is None else self.n_kv_heads
            if self.n_heads % n_kv_heads != 0:
                raise ValueError(
                    f"n_heads ({self.n_heads}) must be divisible by "
                    f"n_kv_heads ({n_kv_heads})"
                )
            if (
                isinstance(self.qkv_linear, PaddedQKVLinear.Config)
                and type(self) is not GQAttention.Config
            ):
                raise ValueError(
                    "PaddedQKVLinear supports only GQAttention, got "
                    f"{type(self).__qualname__}."
                )

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
        self.rope: RoPE | None = None if config.rope is None else config.rope.build()

        # Pluggable QKV projection
        self.qkv_linear = config.qkv_linear.build()
        # The projection owns any head padding, including the input columns of wo.
        self.wo = self.qkv_linear.build_output_projection(config.wo)
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
        x_TD: torch.Tensor,
        attention_metadata: FlexAttentionMetadata | VarlenAttentionMetadata | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The projection's linear declares its own remat regions.
        xq_THK, xk_THK, xv_THV = self.qkv_linear(x_TD)

        # Optional QK normalization (before RoPE, per Qwen3)
        if self.q_norm is not None or self.k_norm is not None:
            assert self.q_norm is not None and self.k_norm is not None
            remat.recompute_needs_tensor(xq_THK)
            xq_THK = self.q_norm(xq_THK)
            remat.recompute_needs_tensor(xk_THK)
            xk_THK = self.k_norm(xk_THK)

        # Apply rotary embeddings
        if self.rope is not None:
            remat.recompute_needs_tensor(xq_THK, xk_THK)
            xq_THK, xk_THK = self.rope(xq_THK, xk_THK, positions)

        out_THV = remat.region(
            self.inner_attention,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(
            xq_THK,
            xk_THK,
            xv_THV,
            attention_metadata=attention_metadata,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
        )
        remat.recompute_needs_tensor(out_THV)
        out_THV = out_THV.contiguous()
        out_TD = out_THV.view(out_THV.shape[0], -1)
        return self.wo(out_TD)


def validate_tp_head_sharding(attention: BaseAttention.Config, *, tp: int) -> None:
    """Raise if the TP degree cannot shard ``attention``'s heads.

    Fused QKV projections validate through :meth:`QKVLinear.Config.validate_tp_degree`;
    :class:`PaddedQKVLinear` pads its heads and accepts any TP degree.
    """
    if tp == 1:
        return
    qkv_linear = getattr(attention, "qkv_linear", None)
    if isinstance(qkv_linear, QKVLinear.Config):
        # Only plain GQAttention with a plain QKVLinear can pad: other
        # attention (e.g. subclasses with an output gate, or gpt-oss attention
        # with sinks) carries extra per-head parameters that padding does not
        # cover.
        can_pad = (
            type(attention) is GQAttention.Config
            and type(qkv_linear) is QKVLinear.Config
        )
        hint = (
            " Use PaddedQKVLinear.Config (make_gqa_config(pad_heads_for_tp=True)) "
            "to pad the heads instead."
            if can_pad
            else ""
        )
        qkv_linear.validate_tp_degree(tp, hint=hint)
    elif attention.n_heads % tp != 0:
        # Attention with separate K/V projections may shard each head's
        # features instead of whole KV-head groups.
        raise ValueError(
            f"tensor parallel degree ({tp}) must divide n_heads ({attention.n_heads})."
        )
