# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import spmd_types as spmd
import torch
from attn_gym.sparse.gather_attn import gather_attn

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_dense_sp_enabled, spmd_mesh_group
from torchtitan.models.common.attention import BaseAttention, InnerAttention
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE

from .compressor import Compressor, Indexer


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

    TODO: the indexer auxiliary loss is intentionally dropped for now; it will
    be re-added as a carrier-injected aux loss (see the NPU fork) once the
    general aux-loss mechanism lands.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        window_size: int
        compress_ratio: int
        softmax_scale: float
        index_topk: int

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.window_size = config.window_size
        self.compress_ratio = config.compress_ratio
        self.softmax_scale = config.softmax_scale
        self.index_topk = config.index_topk

    def _gather_attn(
        self, q_THD, swa_k_TD, cmp_k_SD, cmp_topk_TK, attn_sink, attention_masks
    ) -> torch.Tensor:
        """``cmp_topk_TK`` holds indices into ``cmp_k_SD``; -1 marks unused slots."""
        if attention_masks is not None:
            raise ValueError(
                f"{type(self).__name__} does not accept attention_masks; "
                "the attended positions are built internally."
            )
        with spmd.no_typecheck():
            # gather_attn takes [B, H, T, D]; the KV latent has one head.
            out_1HTD = gather_attn(
                q_THD.transpose(0, 1).unsqueeze(0),
                swa_k_TD[None, None],
                cmp_k_SD[None, None],
                cmp_topk_TK.unsqueeze(0),
                attention_sink=attn_sink,
                sliding_window_size=self.window_size,
                scale=self.softmax_scale,
                impl="fused" if q_THD.device.type == "cuda" else "reference",
            )
            out_THD = out_1HTD.squeeze(0).transpose(0, 1)
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
        attention_masks=None,
    ) -> torch.Tensor:
        seqlen, _, head_dim = q.size()
        no_cmp_k = torch.empty(0, head_dim, dtype=swa_k.dtype, device=q.device)
        no_topk = torch.empty(seqlen, 0, dtype=torch.int32, device=q.device)
        return self._gather_attn(
            q, swa_k, no_cmp_k, no_topk, attn_sink, attention_masks
        )


class HeavilyCompressedAttention(DSV4InnerAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(DSV4InnerAttention.Config):
        pass

    def forward(
        self,
        q,
        swa_k,
        cmp_k,
        attn_sink,
        *,
        attention_masks=None,
    ) -> torch.Tensor:
        # Every compressed block that ends at or before the query is attended:
        # query t sees positions s < (t + 1) // compress_ratio.
        seqlen, n_cmp = q.size(0), cmp_k.size(0)
        cmp_s = torch.arange(n_cmp, dtype=torch.int32, device=q.device)
        limit_t = torch.arange(1, seqlen + 1, device=q.device) // self.compress_ratio
        cmp_topk = torch.where(cmp_s < limit_t.unsqueeze(1), cmp_s, -1)
        return self._gather_attn(q, swa_k, cmp_k, cmp_topk, attn_sink, attention_masks)


class CompressedSparseAttention(DSV4InnerAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(DSV4InnerAttention.Config):
        pass

    def forward(
        self,
        q,
        swa_k,
        cmp_k,
        idx_q,
        idx_k,
        idx_w,
        attn_sink,
        *,
        attention_masks=None,
    ) -> torch.Tensor:
        with spmd.no_typecheck():
            cmp_topk = Indexer.select(
                idx_q,
                idx_k,
                idx_w,
                seqlen=q.size(0),
                ratio=self.compress_ratio,
                topk=self.index_topk,
            )
        return self._gather_attn(q, swa_k, cmp_k, cmp_topk, attn_sink, attention_masks)


class Attention(BaseAttention):
    """DeepSeek V4 attention wrapper around sparse inner attention.

    The module projects Q/KV, applies pre- and post-phase RoPE, prepares
    optional compressed/indexer tensors, and delegates sparse attention to
    ``DSV4InnerAttention``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        dim: int
        n_heads: int
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

    def forward(self, x, attention_masks=None, positions=None):
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

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr)
        with spmd.local():
            q = q.view(num_tokens, -1, self.head_dim)
            _assert_spmd_attention_type(q, tp=spmd.S(1))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.norm_eps)
        q_nope, q_rope = torch.split(q, [self.head_dim - rd, rd], dim=-1)

        kv = self.kv_norm(self.wkv(x))
        kv_nope, kv_rope = torch.split(kv, [self.head_dim - rd, rd], dim=-1)

        q_rope, kv_rope = self.rope(q_rope, kv_rope.unsqueeze(1), positions)
        q = torch.cat([q_nope, q_rope], dim=-1)
        kv = torch.cat([kv_nope, kv_rope.squeeze(1)], dim=-1)

        cmp_k = idx_q = idx_k = idx_w = None
        if self.compress_ratio > 1 and hasattr(self, "indexer"):
            idx_q, idx_k, idx_w = self.indexer(
                x.detach(), qr.detach(), positions=positions
            )
        if self.compress_ratio == 4:
            cmp_k = self.compressor(x, positions=positions)
        elif self.compress_ratio > 1:
            cmp_k = self.compressor_128(x, positions=positions)

        attn_sink_param = self.attn_sink.weight.squeeze(-1)
        if self.compress_ratio == 4:
            o = self.inner_attention(
                q,
                kv,
                cmp_k,
                idx_q,
                idx_k,
                idx_w,
                attn_sink_param,
                attention_masks=attention_masks,
            )
        elif self.compress_ratio > 1:
            o = self.inner_attention(
                q,
                kv,
                cmp_k,
                attn_sink_param,
                attention_masks=attention_masks,
            )
        else:
            o = self.inner_attention(
                q,
                kv,
                attn_sink_param,
                attention_masks=attention_masks,
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
