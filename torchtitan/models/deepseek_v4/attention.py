# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, replace

import spmd_types as spmd
import torch
import torch_remat as remat
from torch.export import register_dataclass

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

from .compressor import compressed_cu_seqlens, Compressor, Indexer, SparseIndexerLoss


@dataclass(frozen=True, slots=True, kw_only=True)
class DeepSeekV4CompressedSparseAttentionMetadata(VarlenAttentionMetadata):
    """Varlen offsets plus the geometry shared by one sparse-attention variant."""

    cu_seq_compressed: torch.Tensor
    group_start: torch.Tensor
    group_first: torch.Tensor
    compressed_causal_indices: torch.Tensor | None = None


register_dataclass(DeepSeekV4CompressedSparseAttentionMetadata)


def _assert_spmd_attention_type(tensor, *, tp):
    if spmd.is_type_checking():
        spmd.assert_type(
            tensor,
            {"dp": spmd.S(0), "cp": spmd.S(1), "tp": tp},
        )


class DSV4InnerAttention(InnerAttention):
    """DeepSeek-V4 sparse attention core on Attention Gym's ``gather_attn``.

    Each query attends to its causal sliding window over the uncompressed KV
    (``swa_k``), to the compressed KV positions (``cmp_k``) listed for it in
    ``cmp_topk``, and to a learned per-head attention sink. Subclasses differ
    only in ``cmp_topk``: none (SWA), every causal position (HCA), or the
    indexer's top-k (CSA). K and V are the same single-head latent.

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
        ) -> DeepSeekV4CompressedSparseAttentionMetadata:
            """Default for the sliding-window variant: no compressed groups."""
            varlen = create_varlen_metadata_for_document(
                positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )
            empty = torch.empty(0, dtype=torch.int32, device=positions.device)
            return DeepSeekV4CompressedSparseAttentionMetadata(
                cu_seq_q=varlen.cu_seq_q,
                cu_seq_k=varlen.cu_seq_k,
                max_q=varlen.max_q,
                max_k=varlen.max_k,
                window_size=varlen.window_size,
                cu_seq_compressed=torch.zeros_like(varlen.cu_seq_q),
                group_start=empty,
                group_first=torch.empty(0, dtype=torch.bool, device=positions.device),
            )

        def _compressed_geometry(
            self,
            metadata: "DeepSeekV4CompressedSparseAttentionMetadata",
            positions: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            """Compressed pool offsets and group indices, computed once."""
            ratio = self.compress_ratio
            cu = metadata.cu_seq_q
            cu_seq_compressed = compressed_cu_seqlens(cu, ratio)
            slots = torch.arange(
                positions.size(0) // ratio, dtype=torch.int32, device=positions.device
            )
            group_doc = torch.searchsorted(
                cu_seq_compressed[1:], slots, right=True, out_int32=True
            )
            group = slots - cu_seq_compressed[group_doc]
            start = torch.where(
                slots < cu_seq_compressed[-1], cu[group_doc] + group * ratio, 0
            )
            return {
                "cu_seq_compressed": cu_seq_compressed,
                "group_start": start,
                "group_first": group == 0,
            }

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.window_size = config.window_size
        self.compress_ratio = config.compress_ratio
        self.softmax_scale = config.softmax_scale
        self.aux_loss = config.aux_loss.build() if config.aux_loss is not None else None

    def _gather_attn(
        self,
        q_THD,
        swa_k_TD,
        cmp_k_SD,
        cmp_topk_TK,
        attn_sink,
        attention_metadata: DeepSeekV4CompressedSparseAttentionMetadata,
        *,
        topk_scores_TK=None,
        aux_loss_denominator=None,
        padding_mask_T=None,
    ) -> torch.Tensor:
        """``cmp_topk_TK`` holds indices into ``cmp_k_SD``; -1 marks unused slots.

        With packed documents the window stays inside each document and the
        indices are local to the query's document in the packed ``cmp_k_SD``.
        """
        cu_seqlens = attention_metadata.cu_seq_q
        cu_seqlens_k = attention_metadata.cu_seq_compressed
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
                    attention_metadata,
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


class SlidingWindowAttention(DSV4InnerAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(DSV4InnerAttention.Config):
        pass

    def forward(
        self,
        q,
        swa_k,
        attn_sink,
        *,
        attention_metadata: DeepSeekV4CompressedSparseAttentionMetadata,
    ) -> torch.Tensor:
        seqlen, _, head_dim = q.size()
        no_cmp_k = torch.empty(0, head_dim, dtype=swa_k.dtype, device=q.device)
        no_topk = torch.empty(seqlen, 0, dtype=torch.int32, device=q.device)
        return self._gather_attn(
            q, swa_k, no_cmp_k, no_topk, attn_sink, attention_metadata
        )


class HeavilyCompressedAttention(DSV4InnerAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(DSV4InnerAttention.Config):
        def build_attention_metadata(
            self, positions: torch.Tensor, **kwargs
        ) -> DeepSeekV4CompressedSparseAttentionMetadata:
            metadata = DSV4InnerAttention.Config.build_attention_metadata(
                self, positions, **kwargs
            )

            def compressed_causal_indices(max_k: int) -> torch.Tensor:
                ratio = self.compress_ratio
                width = min(positions.size(0) // ratio, max_k // ratio)
                limit = (positions + 1) // ratio
                candidates = torch.arange(
                    width, dtype=torch.int32, device=positions.device
                )
                return torch.where(candidates < limit[:, None], candidates, -1)

            return replace(
                metadata,
                **self._compressed_geometry(metadata, positions),
                compressed_causal_indices=compressed_causal_indices(metadata.max_k),
            )

    def forward(
        self,
        q,
        swa_k,
        cmp_k,
        attn_sink,
        *,
        attention_metadata: DeepSeekV4CompressedSparseAttentionMetadata,
    ) -> torch.Tensor:
        return self._gather_attn(
            q,
            swa_k,
            cmp_k,
            attention_metadata.compressed_causal_indices,
            attn_sink,
            attention_metadata,
        )


class CompressedSparseAttention(DSV4InnerAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(DSV4InnerAttention.Config):
        def build_attention_metadata(
            self, positions: torch.Tensor, **kwargs
        ) -> DeepSeekV4CompressedSparseAttentionMetadata:
            metadata = DSV4InnerAttention.Config.build_attention_metadata(
                self, positions, **kwargs
            )
            return replace(metadata, **self._compressed_geometry(metadata, positions))

    def forward(
        self,
        q,
        swa_k,
        cmp_k,
        topk_indices,
        topk_scores: torch.Tensor,
        attn_sink,
        *,
        attention_metadata: DeepSeekV4CompressedSparseAttentionMetadata,
        aux_loss_denominator=None,
        padding_mask_T=None,
    ) -> torch.Tensor:
        return self._gather_attn(
            q,
            swa_k,
            cmp_k,
            topk_indices,
            attn_sink,
            attention_metadata,
            topk_scores_TK=topk_scores,
            aux_loss_denominator=aux_loss_denominator,
            padding_mask_T=padding_mask_T,
        )


class DeepSeekV4Attention(Attention):
    """DeepSeek V4 attention wrapper around sparse inner attention.

    The module projects Q/KV, applies pre- and post-phase RoPE, prepares
    optional compressed/indexer tensors, and delegates sparse attention to
    ``DSV4InnerAttention``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Attention.Config):
        dim: int
        inner_attention: DSV4InnerAttention.Config  # pyrefly: ignore [bad-override]
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

        # Compressor/indexer are conditional, so keep them here too.
        compressor: Compressor.Config | None = None
        compressor_128: Compressor.Config | None = None
        indexer: Indexer.Config | None = None

        def flops(self, positions: torch.Tensor) -> int:
            from torchtitan.models.common.flops import quadratic_attention_flops

            inner_attention = self.inner_attention
            # A causal query at document position p attends to p + 1 keys,
            # at most window_size of them uncompressed.
            num_keys = positions + 1
            attention_op_flops = quadratic_attention_flops(
                num_heads=self.n_heads,
                qk_head_dim=self.head_dim,
                v_head_dim=self.head_dim,
                num_keys=num_keys.clamp(max=inner_attention.window_size),
            )

            if self.compress_ratio > 1:
                num_compressed_keys = num_keys // self.compress_ratio
                if self.compress_ratio == 4:
                    # The indexer scores every compressed key without grad
                    # (forward only); attention reads only the top-k.
                    index_dim = self.index_n_heads * self.index_head_dim
                    attention_op_flops += 2 * index_dim * int(num_compressed_keys.sum())
                    assert self.indexer is not None
                    num_compressed_keys = num_compressed_keys.clamp(
                        max=self.indexer.inner_indexer.topk
                    )
                    # The indexer loss recomputes the top-k scores with grad.
                    # The forward repeats the scoring pass, so only the
                    # backward is new work.
                    if inner_attention.aux_loss is not None:
                        attention_op_flops += (
                            4 * index_dim * int(num_compressed_keys.sum())
                        )
                attention_op_flops += quadratic_attention_flops(
                    num_heads=self.n_heads,
                    qk_head_dim=self.head_dim,
                    v_head_dim=self.head_dim,
                    num_keys=num_compressed_keys,
                )
            return attention_op_flops

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

        if cfg.compressor is not None:
            self.compressor = cfg.compressor.build()
        if cfg.indexer is not None:
            self.indexer = cfg.indexer.build()
        if cfg.compressor_128 is not None:
            self.compressor_128 = cfg.compressor_128.build()

        self.inner_attention = cfg.inner_attention.build()

    def forward(
        self,
        x,
        attention_metadata: DeepSeekV4CompressedSparseAttentionMetadata,
        positions: torch.Tensor,
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

        attn_sink_param = self.attn_sink.weight.squeeze(-1)
        if self.compress_ratio == 4:
            return_scores = (
                self.inner_attention.training
                and self.inner_attention.aux_loss is not None
            )
            with torch.set_grad_enabled(torch.is_grad_enabled() and return_scores):
                topk_indices, topk_scores = self.indexer(
                    x.detach(),
                    qr.detach(),
                    positions=positions,
                    attention_metadata=attention_metadata,
                    return_scores=return_scores,
                )
            cmp_k = self.compressor(
                x, positions=positions, attention_metadata=attention_metadata
            )
            o = self.inner_attention(
                q,
                kv,
                cmp_k,
                topk_indices,
                topk_scores,
                attn_sink_param,
                attention_metadata=attention_metadata,
                aux_loss_denominator=aux_loss_denominator,
                padding_mask_T=padding_mask,
            )
        elif self.compress_ratio > 1:
            cmp_k = self.compressor_128(
                x, positions=positions, attention_metadata=attention_metadata
            )
            o = self.inner_attention(
                q,
                kv,
                cmp_k,
                attn_sink_param,
                attention_metadata=attention_metadata,
            )
        else:
            o = self.inner_attention(
                q,
                kv,
                attn_sink_param,
                attention_metadata=attention_metadata,
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
