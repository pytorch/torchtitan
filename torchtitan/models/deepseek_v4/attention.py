# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import NamedTuple

import spmd_types as spmd
import torch
import torch_remat as remat

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_dense_sp_enabled, spmd_mesh_group
from torchtitan.models.common.attention import (
    Attention,
    create_varlen_metadata_for_document,
    InnerAttention,
    VarlenAttentionMetadata,
)

from torchtitan.models.common.attn_gym import AuxRequest, gather_attn
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE

from .compressor import (
    compressed_cu_seqlens,
    Compressor,
    document_cmp_base,
    Indexer,
    SparseIndexerLoss,
)


def _assert_spmd_attention_type(tensor, *, tp):
    if spmd.is_type_checking():
        spmd.assert_type(
            tensor,
            {"dp": spmd.S(0), "cp": spmd.S(1), "tp": tp},
        )


class CompressedSparseAttentionMetadata(NamedTuple):
    """Position-derived metadata for top-k compressed attention (CSA).

    ``cmp_base_T`` is the row of the packed compressed pool where each query's
    document starts. Only torch-side code reads it: the score recompute that
    feeds the distillation loss, and the loss teacher. A fused differentiable
    selector would not need it, so it must not be removed with the recompute.
    """

    varlen: VarlenAttentionMetadata
    cu_seq_cmp: torch.Tensor
    cmp_base_T: torch.Tensor

    @property
    def cu_seq_q(self) -> torch.Tensor:
        return self.varlen.cu_seq_q

    @property
    def max_q(self) -> int:
        return self.varlen.max_q

    @property
    def max_k(self) -> int:
        return self.varlen.max_k


class HeavilyCompressedAttentionMetadata(NamedTuple):
    """Position-derived metadata for causal compressed attention (HCA).

    ``causal_cmp_TK`` lists, for every query, the compressed blocks that end at
    or before it; ``-1`` marks unusable slots. Building it once keeps the
    per-layer ``[T, n_cmp]`` mask out of the layer loop.
    """

    varlen: VarlenAttentionMetadata
    cu_seq_cmp: torch.Tensor
    causal_cmp_TK: torch.Tensor

    @property
    def cu_seq_q(self) -> torch.Tensor:
        return self.varlen.cu_seq_q

    @property
    def max_q(self) -> int:
        return self.varlen.max_q

    @property
    def max_k(self) -> int:
        return self.varlen.max_k


class DeepSeekV4InnerAttention(InnerAttention):
    """DeepSeek-V4 sparse attention core on Attention Gym's ``gather_attn``.

    Each query attends to its causal sliding window over the uncompressed KV
    (``swa_k``), to the compressed KV positions (``cmp_k``) listed for it in
    ``cmp_topk``, and to a learned per-head attention sink. ``compress_ratio``
    selects the compressed policy: ``1`` attends no compressed block, ``4``
    attends the indexer's top-k (CSA), any other value attends every causally
    complete block (HCA). K and V are the same single-head latent.

    The variant tensors are keyword-only. A local SPMD region only types its
    positional inputs, so a layer without a compressed pool passes ``None``
    without leaving that contract. Empty-``Config`` subclasses
    (``DeepSeekV4SlidingWindowAttention`` and friends) provide the per-variant
    metadata identity, because metadata is keyed by inner-attention class.

    The indexer distillation loss is implemented as a carrier-injected
    auxiliary loss on CSA layers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        window_size: int
        compress_ratio: int
        softmax_scale: float
        aux_loss: SparseIndexerLoss.Config | None = None

        def build_attention_metadata(
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> VarlenAttentionMetadata:
            """Build document offsets and this variant's compressed geometry."""
            varlen = create_varlen_metadata_for_document(
                positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )
            if self.compress_ratio <= 1:
                return varlen
            device = positions.device
            seqlen = positions.shape[0]
            cu_seq_q = varlen.cu_seq_q
            counts_T = torch.diff(cu_seq_q)
            # Documents are packed ratio-aligned, so whole-stream grouping is
            # exactly per-document grouping and the compressed offsets are the
            # query offsets divided by the ratio.
            torch._assert_async(
                (counts_T % self.compress_ratio == 0).all(),
                "DeepSeek V4 requires packed documents aligned to the "
                f"compression ratio ({self.compress_ratio}).",
            )
            cu_seq_cmp = cu_seq_q // self.compress_ratio
            # The variant metadata types stay model-local (``AttentionMetadata``
            # is a closed union in ``common``); the base signature is the
            # varlen type, and consumers read the fields dynamically.
            if self.compress_ratio == 4:
                return CompressedSparseAttentionMetadata(  # pyrefly: ignore [bad-return]
                    varlen=varlen,
                    cu_seq_cmp=cu_seq_cmp,
                    cmp_base_T=document_cmp_base(cu_seq_q, self.compress_ratio, seqlen, device),
                )
            # Every compressed block that ends at or before the query is
            # attended: query t sees blocks s < (t + 1) // compress_ratio,
            # counted from the start of its document.
            doc_start_T = torch.repeat_interleave(
                cu_seq_q[:-1], counts_T, output_size=seqlen
            )
            pos_in_doc_T = (
                torch.arange(seqlen, dtype=torch.int32, device=device) - doc_start_T
            )
            n_cmp = min(seqlen // self.compress_ratio, varlen.max_k // self.compress_ratio)
            cmp_s = torch.arange(n_cmp, dtype=torch.int32, device=device)
            limit_T = (pos_in_doc_T + 1) // self.compress_ratio
            allowed_TK = cmp_s[None, :] < limit_T[:, None]
            return HeavilyCompressedAttentionMetadata(  # pyrefly: ignore [bad-return]
                varlen=varlen,
                cu_seq_cmp=cu_seq_cmp,
                causal_cmp_TK=torch.where(
                    allowed_TK, cmp_s[None, :].expand_as(allowed_TK), -1
                ),
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.window_size = config.window_size
        self.compress_ratio = config.compress_ratio
        self.softmax_scale = config.softmax_scale
        self.aux_loss = config.aux_loss.build() if config.aux_loss is not None else None

    @property
    def uses_indexer_scores(self) -> bool:
        """Whether the distillation loss needs differentiable top-k scores."""
        return self.aux_loss is not None

    def forward(
        self,
        q,
        swa_k,
        attn_sink,
        *,
        cmp_k=None,
        topk_indices=None,
        topk_scores=None,
        attention_metadata,
        aux_loss_denominator=None,
        padding_mask_T=None,
    ) -> torch.Tensor:
        """Run one sparse attention layer.

        Args:
            q: Queries of shape ``[T, H, D]``.
            swa_k: Uncompressed KV of shape ``[T, D]``.
            attn_sink: Per-head learned sink of shape ``[H]``.
            cmp_k: Compressed KV of shape ``[S, D]``, or None without a
                compressed pool.
            topk_indices: Indexer selection of shape ``[T, K]``, document-local
                with ``-1`` in unusable slots (CSA only).
            topk_scores: Differentiable scores for the selected slots, ``-inf``
                in unusable slots (CSA only).
            attention_metadata: Position-derived metadata for this layer
                variant; always present.

        Returns:
            Attention output of shape ``[T, H, D]``.
        """
        cu_seqlens = attention_metadata.cu_seq_q
        cu_seqlens_k = self._compressed_offsets(attention_metadata)
        seqlen = q.size(0)
        if self.compress_ratio <= 1:
            cmp_k = torch.empty(0, q.size(-1), dtype=swa_k.dtype, device=q.device)
            cmp_topk = torch.empty(seqlen, 0, dtype=torch.int32, device=q.device)
        elif self.compress_ratio == 4:
            if cmp_k is None or topk_indices is None:
                raise ValueError(
                    "Compressed sparse attention requires the compressor output "
                    "and the indexer selection."
                )
            cmp_topk = topk_indices
        else:
            cmp_topk = attention_metadata.causal_cmp_TK
        return self._gather_attn(
            q,
            swa_k,
            cmp_k,
            cmp_topk,
            attn_sink,
            cu_seqlens,
            cu_seqlens_k,
            attention_metadata.cmp_base_T if self.compress_ratio == 4 else None,
            topk_scores_TK=topk_scores if self.compress_ratio == 4 else None,
            aux_loss_denominator=aux_loss_denominator,
            padding_mask_T=padding_mask_T,
        )

    def _compressed_offsets(self, attention_metadata):
        """Compressed-pool offsets handed to the kernels as ``cu_seqlens_k``."""
        if self.compress_ratio <= 1:
            # Sliding-window-only layers pass an empty compressed pool: every
            # document's span is empty.
            return torch.zeros_like(attention_metadata.cu_seq_q)
        return attention_metadata.cu_seq_cmp

    def _gather_attn(
        self,
        q_THD,
        swa_k_TD,
        cmp_k_SD,
        cmp_topk_TK,
        attn_sink,
        cu_seqlens,
        cu_seqlens_k,
        cmp_base_T,
        *,
        topk_scores_TK=None,
        aux_loss_denominator=None,
        padding_mask_T=None,
    ) -> torch.Tensor:
        """``cmp_topk_TK`` holds indices into ``cmp_k_SD``; -1 marks unused slots.

        With packed ``cu_seqlens`` the window stays inside each document and the
        indices are local to the query's document in the packed ``cmp_k_SD``.
        """
        with spmd.no_typecheck():
            # gather_attn takes [B, H, T, D]; the KV latent has one head.
            aux_loss = self.aux_loss if self.training else None
            out_1HTD, aux = gather_attn(
                q_THD.transpose(0, 1).unsqueeze(0),
                swa_k_TD[None, None],
                cmp_k_SD[None, None],
                cmp_topk_TK.unsqueeze(0),
                attention_sink=attn_sink,
                sliding_window_size=self.window_size,
                cu_seqlens=cu_seqlens,
                cu_seqlens_k=cu_seqlens_k,
                scale=self.softmax_scale,
                impl="fused" if q_THD.device.type == "cuda" else "reference",
                return_aux=AuxRequest(lse=aux_loss is not None),
            )
            out_THD = out_1HTD.squeeze(0).transpose(0, 1)
            if aux_loss is not None:
                assert topk_scores_TK is not None and aux.lse is not None
                if aux_loss_denominator is None:
                    raise RuntimeError(
                        "SparseIndexerLoss requires aux_loss_denominator."
                    )
                out_THD = aux_loss(
                    q_THD.detach(),
                    cmp_k_SD.detach(),
                    cmp_topk_TK,
                    topk_scores_TK,
                    aux.lse.squeeze(0).transpose(0, 1).detach(),
                    cmp_base_T,
                    carrier=out_THD,
                    denominator=aux_loss_denominator,
                    padding_mask_T=padding_mask_T,
                )
        # Kernel output is opaque to SPMD typechecking; it keeps q's layout.
        if spmd.is_type_checking():
            spmd.assert_type(
                out_THD, spmd.get_local_type(q_THD), spmd.get_partition_spec(q_THD)
            )
        return out_THD

class DeepSeekV4SlidingWindowAttention(DeepSeekV4InnerAttention):
    """Sliding-window-only layers: the metadata identity for ``compress_ratio`` 1."""

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV4InnerAttention.Config):
        pass


class DeepSeekV4HeavilyCompressedAttention(DeepSeekV4InnerAttention):
    """Every-causal-block layers: the metadata identity for HCA ratios."""

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV4InnerAttention.Config):
        pass


class DeepSeekV4CompressedSparseAttention(DeepSeekV4InnerAttention):
    """Top-k compressed layers: the metadata identity for CSA (ratio 4)."""

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV4InnerAttention.Config):
        pass


class DeepSeekV4Attention(Attention):
    """DeepSeek V4 attention wrapper around sparse inner attention.

    The module projects Q/KV, applies pre- and post-phase RoPE, prepares
    optional compressed/indexer tensors, and delegates sparse attention to
    ``DSV4InnerAttention``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Attention.Config):
        dim: int
        inner_attention: DeepSeekV4InnerAttention.Config  # pyrefly: ignore [bad-override]
        rope: RoPE.Config
        head_dim: int = 512
        rope_head_dim: int = 64
        q_lora_rank: int = 1024
        o_lora_rank: int = 1024
        n_groups: int = 8
        compress_ratio: int = 1
        norm_eps: float = 1e-6
        index_n_heads: int = 64
        index_head_dim: int = 128
        n_layers: int = 4
        layer_id: int = 0
        mask_type: str = "causal"

        # Sub-module configs — declared as fields so the sharding system can
        # set sharding_config on them before build().
        wq_a: Linear.Config
        q_norm: RMSNorm.Config
        wq_b: Linear.Config
        wkv: Linear.Config
        kv_norm: RMSNorm.Config
        wo_a: Linear.Config
        wo_b: Linear.Config
        attn_sink: Linear.Config

        # Compressor/indexer are conditional, so keep them here too. One
        # compressor covers every compressed ratio; its overlap history is
        # selected from ``compress_ratio`` when the config is built.
        compressor: Compressor.Config | None = None
        indexer: Indexer.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.q_lora_rank = cfg.q_lora_rank
        self.o_lora_rank = cfg.o_lora_rank
        self.n_groups = cfg.n_groups
        self.compress_ratio = cfg.compress_ratio
        self.norm_eps = cfg.norm_eps
        self.softmax_scale = cfg.head_dim**-0.5
        self.layer_id = cfg.layer_id
        self.n_layers = cfg.n_layers
        self.rope = cfg.rope.build()

        # Build all sub-modules from their configs.
        self.wq_a = cfg.wq_a.build()
        self.q_norm = cfg.q_norm.build()
        self.wq_b = cfg.wq_b.build()
        self.wkv = cfg.wkv.build()
        self.kv_norm = cfg.kv_norm.build()
        self.wo_a = cfg.wo_a.build()
        self.wo_b = cfg.wo_b.build()
        self.attn_sink = cfg.attn_sink.build()

        self.compressor = (
            cfg.compressor.build() if cfg.compressor is not None else None
        )
        self.indexer = cfg.indexer.build() if cfg.indexer is not None else None

        self.inner_attention = cfg.inner_attention.build()

    def forward(
        self,
        x,
        attention_metadata,
        positions=None,
        aux_loss_denominator=None,
        padding_mask=None,
    ):
        """Apply one DeepSeek V4 attention layer over folded tokens."""
        tp_group = spmd_mesh_group(MeshAxisName.TP)
        if tp_group is not None:
            # Query, KV, compressor, and indexer branches consume x. Gather
            # once at their common attention boundary.
            x = spmd.redistribute(
                x,
                tp_group,
                src=spmd.S(0) if spmd_dense_sp_enabled() else spmd.I,
                dst=spmd.R,
                backward_options={"op_dtype": x.dtype},
            )

        num_tokens = x.size(0)
        rd = self.rope_head_dim

        qr = self.wq_a(x)
        # q_norm reads the wq_a projection output with bare ops.
        remat.recompute_needs_tensor(qr)
        qr = self.q_norm(qr)
        q = self.wq_b(qr)
        with spmd.local():
            q = q.view(num_tokens, -1, self.head_dim)
            _assert_spmd_attention_type(q, tp=spmd.S(1))
        # The rescale reads the wq_b projection output with bare ops.
        remat.recompute_needs_tensor(q)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.norm_eps)
        q_nope, q_rope = torch.split(q, [self.head_dim - rd, rd], dim=-1)

        kv = self.wkv(x)
        # kv_norm reads the wkv projection output with bare ops.
        remat.recompute_needs_tensor(kv)
        kv = self.kv_norm(kv)
        kv_nope, kv_rope = torch.split(kv, [self.head_dim - rd, rd], dim=-1)

        q_rope, kv_rope = self.rope(q_rope, kv_rope.unsqueeze(1), positions)
        q = torch.cat([q_nope, q_rope], dim=-1)
        kv = torch.cat([kv_nope, kv_rope.squeeze(1)], dim=-1)

        cmp_k = None
        if self.compressor is not None:
            cmp_k = self.compressor(x, positions=positions)

        topk_indices = topk_scores = None
        if self.indexer is not None:
            topk_indices, topk_scores = self.indexer(
                x.detach(),
                qr.detach(),
                positions=positions,
                attention_metadata=attention_metadata,
                return_scores=(
                    self.training and self.inner_attention.uses_indexer_scores
                ),
            )

        o = self.inner_attention(
            q,
            kv,
            self.attn_sink.weight.squeeze(-1),
            cmp_k=cmp_k,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
            attention_metadata=attention_metadata,
            aux_loss_denominator=aux_loss_denominator,
            padding_mask_T=padding_mask,
        )

        o_nope, o_rope = torch.split(o, [self.head_dim - rd, rd], dim=-1)
        o_rope = self.rope(o_rope, positions=positions, inverse=True)
        o = torch.cat([o_nope, o_rope], dim=-1)

        with spmd.local():
            n_local_heads = o.shape[1]
            n_local_groups = self.n_groups // (self.n_heads // n_local_heads)
            o = o.view(num_tokens, n_local_groups, -1)
            _assert_spmd_attention_type(o, tp=spmd.S(1))
            wo_a = self.wo_a.weight.view(n_local_groups, self.o_lora_rank, -1)
            if spmd.is_type_checking():
                spmd.assert_type(
                    wo_a,
                    {"dp": spmd.R, "cp": spmd.R, "tp": spmd.S(0)},
                )
        o = torch.einsum("tgd,grd->tgr", o, wo_a)
        with spmd.local():
            o = o.reshape(num_tokens, -1)
            _assert_spmd_attention_type(o, tp=spmd.S(1))
        return self.wo_b(o)
