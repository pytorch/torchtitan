# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in dense causal BF16 SDPA for one continuous, unpadded token row.

For packed text, activate both component overrides with ``--override``::

    torchtitan_recipes.overrides.dense_causal_sdpa.dense_causal_bf16_sdpa,torchtitan_recipes.overrides.dense_causal_sdpa.unmasked_text_rows

The data override applies to training and validation text collators before
MTP counting, RoPE and attention metadata. It preserves labels/padding and
removes source-document resets, not logical context-row boundaries. Dense
SDPA requires one full context row per microbatch and still rejects padding.
Already-continuous frozen sources may use the attention override alone.
Context parallelism and sliding-window attention are unsupported.
Sequence parallelism is supported when the attention projections gather the
complete token row before this kernel boundary.

For DeepSeek MLA fusion, use ``dense_causal_fused_mla`` instead of separate
fused-MLA and dense-inner overrides. One factory owns the parent/child change,
avoiding an order-dependent replacement; both existing implementations are reused.
MLA's projection/sharding wrapper is retained; only its Q/K/V attention leaf changes.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import ClassVar

import numpy as np
import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torchtitan.components.data.collators import Collator, TextCollator
from torchtitan.components.data.dataset import TextSequence
from torchtitan.components.data.loader import GrainDataLoader
from torchtitan.components.data.types import TokenizedTrainingMicrobatch

from torchtitan.config import derive, override
from torchtitan.models.common.attention import (
    InnerAttention,
    MLAAttention,
    VarlenAttentionMetadata,
    VarlenInnerAttention,
)
from torchtitan.models.deepseek_v3.attention import DeepSeekMLAAttention

__all__ = [
    "ContinuousTextCollator",
    "ContinuousTextDataLoader",
    "DenseCausalBf16SDPA",
    "dense_causal_bf16_sdpa",
    "dense_causal_fused_mla",
    "unmasked_text_rows",
]

# T = tokens, H = heads, K = query/key head dimension, V = value head dimension.


class ContinuousTextCollator(TextCollator):
    """Join source documents within context rows before native MTP counting.

    Labels and padding remain source-owned. Only positions change; the native
    collator still owns allocation, copying, padding and objective counts.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TextCollator.Config):
        pass

    def __call__(self, rows: Sequence[TextSequence]) -> TokenizedTrainingMicrobatch:
        """Supply continuous row positions to the unchanged core text collator."""
        continuous_rows = []
        offset = 0
        for row in rows:
            end = offset + len(row.input_ids)
            continuous_rows.append(
                replace(
                    row,
                    positions=np.arange(offset, end, dtype=np.int64)
                    % self._max_context_length,
                )
            )
            offset = end
        return super().__call__(continuous_rows)


class ContinuousTextDataLoader(GrainDataLoader):
    """Bound packing by tokens and attention metadata by logical context rows."""

    @dataclass(kw_only=True, slots=True)
    class Config(GrainDataLoader.Config):
        collator: Collator.Config = field(default_factory=ContinuousTextCollator.Config)
        max_num_documents: int | None = None

        def __post_init__(self) -> None:
            GrainDataLoader.Config.__post_init__(self)
            if not isinstance(self.collator, ContinuousTextCollator.Config):
                raise ValueError(
                    "Continuous row capacity requires ContinuousTextCollator."
                )
            if self.max_num_documents is not None:
                raise ValueError(
                    "Continuous text packing is bounded by tokens, not documents."
                )

        def _get_attention_document_capacity(
            self, *, max_context_length: int, num_tokens_per_microbatch: int
        ) -> int:
            """Return the number of context rows, independent of source documents."""
            return (
                num_tokens_per_microbatch + max_context_length - 1
            ) // max_context_length


@spmd.no_typecheck(out_types=spmd.PartitionSpec(("dp", "cp"), "tp", None))
def _dense_causal_sdpa(q_THK, k_THK, v_THV, *, scale, enable_gqa):
    """Run SDPA on strided batch/head views at the opaque kernel boundary."""
    with sdpa_kernel(
        [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION], set_priority=True
    ):
        out_BHTV = F.scaled_dot_product_attention(
            q_THK.transpose(0, 1).unsqueeze(0),
            k_THK.transpose(0, 1).unsqueeze(0),
            v_THV.transpose(0, 1).unsqueeze(0),
            is_causal=True,
            scale=scale,
            enable_gqa=enable_gqa,
        )
    return out_BHTV.squeeze(0).transpose(0, 1)


class DenseCausalBf16SDPA(VarlenInnerAttention):
    """Causal BF16 attention with cuDNN then Flash backend preference."""

    @dataclass(kw_only=True, slots=True)
    class Config(VarlenInnerAttention.Config):
        _requires_document_capacity: ClassVar[bool] = False

        def build_attention_metadata(
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> VarlenAttentionMetadata:
            """Validate one continuous row once during input preprocessing."""
            del max_num_documents, max_context_length
            if positions.ndim != 1 or positions.numel() == 0:
                raise ValueError(
                    "Dense causal SDPA requires a nonempty 1D position row."
                )
            torch._assert_async(
                (positions[1:] == positions[:-1] + 1).all(),
                "Dense causal SDPA requires continuous positions; enable unmasked_text_rows "
                "for packed text and use one context row per microbatch.",
            )
            if padding_mask is not None:
                if padding_mask.shape != positions.shape:
                    raise ValueError("padding_mask must match the position row.")
                torch._assert_async(
                    ~padding_mask.any(), "Dense causal SDPA does not support padding."
                )
            offsets = torch.tensor(
                [0, positions.numel()], dtype=torch.int32, device=positions.device
            )
            if spmd.is_type_checking():
                spmd.mutate_type(offsets, "dp", src=spmd.R, dst=spmd.V)
            return VarlenAttentionMetadata(
                offsets, offsets, positions.numel(), positions.numel()
            )

    def __init__(self, config: Config) -> None:
        if config.window_size != (-1, 0):
            raise ValueError("Dense causal SDPA requires window_size=(-1, 0).")
        super().__init__(config)

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        attention_metadata: VarlenAttentionMetadata,
        scale: float | None = None,
        out_transform=None,
        **kwargs,
    ) -> torch.Tensor:
        """Apply causal SDPA to BF16 projections covering the complete row."""
        if out_transform is not None:
            raise ValueError(
                "Dense causal SDPA does not expose an LSE output transform."
            )
        if not isinstance(attention_metadata, VarlenAttentionMetadata):
            raise ValueError("Dense causal SDPA requires preprocessed row metadata.")
        if any(
            tensor.ndim != 3 or tensor.dtype != torch.bfloat16
            for tensor in (q_THK, k_THK, v_THV)
        ):
            raise ValueError(
                "Dense causal SDPA requires BF16 [tokens, heads, features] projections."
            )
        if (
            attention_metadata.cu_seq_q.numel() != 2
            or attention_metadata.cu_seq_k.numel() != 2
            or attention_metadata.max_k != attention_metadata.max_q
            or any(
                tensor.shape[0] != attention_metadata.max_q
                for tensor in (q_THK, k_THK, v_THV)
            )
        ):
            raise ValueError(
                "Dense causal SDPA projections must cover one complete token row; CP or incomplete SP layouts are unsupported."
            )
        return _dense_causal_sdpa(
            q_THK, k_THK, v_THV, scale=scale, enable_gqa=kwargs.get("enable_gqa", False)
        )


@override(
    target=InnerAttention.Config,
    description="Use dense causal BF16 SDPA for continuous unpadded token rows.",
)
def dense_causal_bf16_sdpa(
    cfg: InnerAttention.Config,
) -> InnerAttention.Config:
    """Replace the causal leaf while preserving MLA projection/sharding wrappers."""
    if isinstance(cfg, MLAAttention.Config):
        return derive(
            cfg, type(cfg), inner_attention=dense_causal_bf16_sdpa(cfg.inner_attention)
        )
    if type(cfg) is not VarlenInnerAttention.Config or cfg.window_size != (-1, 0):
        raise ValueError(
            "Dense causal SDPA requires plain causal Varlen attention; CP and sliding windows are unsupported."
        )
    return derive(cfg, DenseCausalBf16SDPA.Config)


@override(
    target=GrainDataLoader.Config,
    exact=True,
    description="Use continuous context-row positions for training and validation text.",
)
def unmasked_text_rows(cfg: GrainDataLoader.Config) -> GrainDataLoader.Config:
    """Select recipe-owned continuous collation for training and validation."""
    if type(cfg.collator) is not TextCollator.Config:
        raise ValueError("Continuous text rows require the native text collator.")
    return derive(
        cfg,
        ContinuousTextDataLoader.Config,
        collator=derive(cfg.collator, ContinuousTextCollator.Config),
        max_num_documents=None,
    )


@override(
    target=DeepSeekMLAAttention.Config,
    exact=True,
    description="Compose existing fused DeepSeek MLA with dense causal BF16 attention.",
)
def dense_causal_fused_mla(
    cfg: DeepSeekMLAAttention.Config,
) -> DeepSeekMLAAttention.Config:
    """Replace MLA assembly and its inner attention as one coherent override."""
    from torchtitan_recipes.overrides.fused_mla import fused_mla

    fused_config = fused_mla(cfg)
    fused_config.inner_attention = derive(
        cfg.inner_attention,
        type(cfg.inner_attention),
        inner_attention=dense_causal_bf16_sdpa(cfg.inner_attention.inner_attention),
    )
    return fused_config
