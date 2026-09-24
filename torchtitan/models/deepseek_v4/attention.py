# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import spmd_types as spmd
import torch
from attn_gym.sparse.gather_attn import gather_attn
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_dense_sp_enabled, spmd_mesh_group
from torchtitan.models.common.attention import (
    BaseAttention,
    FlexInnerAttention,
    InnerAttention,
)
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


class DSV4FlexInnerAttention(FlexInnerAttention):
    """DeepSeek sparse attention core for DeepSeek-V4 (sliding window and HCA).

    The core attends over the concatenated KV sequence ``[0, L + n_cmp + 1)``,
    where the first ``L`` positions are the uncompressed sliding-window KV
    (``swa_k``), the next ``n_cmp`` positions are the compressed KV
    (``cmp_k``), and the last position is a learned attention sink token:

    - sliding window: fixed pattern over ``swa_k``, expressed as a
      ``mask_mod`` predicate (no indices);
    - compressed blocks: for HCA (``compress_ratio=128``) all causal blocks
      are attendable, also a fixed ``mask_mod`` pattern;
    - attention sink: always attendable via ``score_mod``.

    CSA (``compress_ratio=4``) runs on ``DSV4InnerAttention`` instead; see
    ``CompressedSparseAttention``.

    The ``mask_mod`` is evaluated at token granularity inside flex_attention;
    the per-query-block KV block listing (``BlockMask.from_kv_blocks``) only
    restricts which blocks the kernel loads.

    Overrides can replace ``_build_block_mask`` (e.g. NPU varlen kernels) or
    the whole ``forward`` (e.g. fused SMLA/CSA kernels, which consume the raw
    ``q / swa_k / cmp_k / idx_q / idx_k / idx_w`` tensors). Under context
    parallelism, all-gathering ``idx_k`` and ``cmp_k`` at this module boundary
    enables global sparse selection.

    TODO: the indexer auxiliary loss is intentionally dropped for now; it will
    be re-added as a carrier-injected aux loss (see the NPU fork) once the
    general aux-loss mechanism lands.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(FlexInnerAttention.Config):
        window_size: int
        compress_ratio: int
        softmax_scale: float
        index_topk: int

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.window_size = config.window_size
        self.compress_ratio = config.compress_ratio
        self.softmax_scale = config.softmax_scale
        self.index_topk = config.index_topk
        self.block_size = config.block_size

    def get_window_topk_idxs(
        self,
        *,
        bsz: int,
        seqlen: int,
        device,
    ) -> torch.Tensor:
        """Return sliding-window KV indices in the concatenated KV space.

        Args:
            bsz: Batch size.
            seqlen: Query sequence length and uncompressed KV length.
            device: Device used for the generated index tensor.

        Returns:
            Tensor of shape ``[B, L, W]``. Valid entries are uncompressed KV
            positions in ``[0, L)`` and padded entries are ``-1``.
        """
        window = min(seqlen, self.window_size)
        q_idx = torch.arange(seqlen, device=device).unsqueeze(1)
        idxs = (q_idx - window + 1).clamp_min(0) + torch.arange(window, device=device)
        idxs = torch.where(idxs <= q_idx, idxs, -1)
        return idxs.unsqueeze(0).expand(bsz, -1, -1)

    def get_compress_topk_idxs(
        self,
        *,
        bsz: int,
        seqlen: int,
        n_cmp: int,
        device,
    ) -> torch.Tensor:
        """Return causal compressed KV indices in the concatenated KV space.

        Args:
            bsz: Batch size.
            seqlen: Query sequence length and uncompressed KV length.
            n_cmp: Number of compressed KV tokens.
            device: Device used for the generated index tensor.

        Returns:
            Tensor of shape ``[B, L, n_cmp]``. Valid entries are compressed KV
            positions offset by ``seqlen`` and padded entries are ``-1``.
        """
        if n_cmp == 0:
            return torch.empty((bsz, seqlen, 0), dtype=torch.int64, device=device)

        cmp_idx = torch.arange(n_cmp, device=device).repeat(seqlen, 1)
        causal_limit = torch.arange(1, seqlen + 1, device=device).unsqueeze(1)
        causal_limit = causal_limit // self.compress_ratio
        cmp_idx = torch.where(cmp_idx < causal_limit, seqlen + cmp_idx, -1)
        return cmp_idx.unsqueeze(0).expand(bsz, -1, -1)

    def _build_block_mask(
        self,
        bsz: int,
        seqlen: int,
        kv_len: int,
        selected_indices: torch.Tensor,
        device,
    ) -> BlockMask:
        """Build a FlexInnerAttention block mask from selected KV indices.

        Args:
            bsz: Batch size.
            seqlen: Query sequence length.
            kv_len: Length of the concatenated KV sequence.
            selected_indices: Tensor of shape ``[B, L, K]`` containing final KV
                positions in ``[0, kv_len)``; ``-1`` entries are ignored.
            device: Device used for mask tensors.

        Returns:
            ``BlockMask`` whose block list and token-level predicate encode
            exactly the selected KV positions.
        """
        bs = self.block_size
        bq, bk = bs if isinstance(bs, tuple) else (bs, bs)
        assert (
            seqlen % bq == 0
        ), f"seqlen ({seqlen}) must be divisible by Q block size ({bq})"
        n_kv_blocks = (kv_len + bk - 1) // bk
        n_q_blocks = seqlen // bq

        valid = selected_indices >= 0
        safe_indices = selected_indices.clamp(0, kv_len - 1)

        selected_blocks = (safe_indices // bk).reshape(
            bsz, n_q_blocks, bq * selected_indices.size(-1)
        )
        block_values = valid.reshape(selected_blocks.shape).to(torch.int32)
        bm = torch.zeros(
            bsz, 1, n_q_blocks, n_kv_blocks, dtype=torch.int32, device=device
        )
        bm[:, 0].scatter_add_(-1, selected_blocks, block_values)
        bm = (bm > 0).to(torch.int32)
        kv_num_blocks = bm.sum(dim=-1).to(torch.int32)
        kv_indices = torch.argsort(bm, dim=-1, descending=True, stable=True).to(
            torch.int32
        )

        selected_count = torch.zeros(
            bsz, seqlen, kv_len, dtype=torch.int32, device=device
        )
        selected_count.scatter_add_(2, safe_indices, valid.to(torch.int32))
        selected_mask = selected_count > 0

        def dsa_mask_mod(b, h, q_idx, kv_idx):
            return selected_mask[b, q_idx, kv_idx]

        return BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            BLOCK_SIZE=(bq, bk),
            mask_mod=dsa_mask_mod,
            seq_lengths=(seqlen, kv_len),
        )

    def _forward_impl(
        self,
        q,
        swa_k,
        attn_sink,
        *,
        cmp_k=None,
        attention_masks=None,
    ) -> torch.Tensor:
        """Run DSV4 sparse attention over a folded token stream."""
        if attention_masks is not None:
            raise ValueError(
                "DSV4FlexInnerAttention does not accept attention_masks; "
                "the DSA block mask is built internally."
            )
        if attn_sink is None:
            raise ValueError("DSV4FlexInnerAttention requires attn_sink")

        seqlen, _, head_dim = q.size()
        n_cmp = 0 if cmp_k is None else cmp_k.size(0)
        sink_idx = seqlen + n_cmp

        kv = swa_k.unsqueeze(1)
        if cmp_k is not None:
            kv = torch.cat([kv, cmp_k.unsqueeze(1)], dim=0)
        sink_kv = kv.new_zeros((1, 1, head_dim))
        kv = torch.cat([kv, sink_kv], dim=0)
        kv = kv.expand(-1, q.size(1), -1)

        with spmd.no_typecheck():
            selected_indices = [
                self.get_window_topk_idxs(bsz=1, seqlen=seqlen, device=q.device)
            ]
            if self.compress_ratio > 1:
                selected_indices.append(
                    self.get_compress_topk_idxs(
                        bsz=1, seqlen=seqlen, n_cmp=n_cmp, device=q.device
                    )
                )
            sink_indices = torch.full(
                (1, seqlen, 1), sink_idx, dtype=torch.int64, device=q.device
            )
            selected_indices.append(sink_indices)
            selected_indices = torch.cat(selected_indices, dim=-1)

            block_mask = self._build_block_mask(
                1, seqlen, kv.size(0), selected_indices, q.device
            )

            def v4_sink_score_mod(score, b, h, q_idx, kv_idx):
                return torch.where(kv_idx == sink_idx, attn_sink[h], score)

            return super().forward(
                q,
                kv,
                kv,
                attention_masks=block_mask,
                score_mod=v4_sink_score_mod,
                scale=self.softmax_scale,
            )


class SlidingWindowAttention(DSV4FlexInnerAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(DSV4FlexInnerAttention.Config):
        pass

    def forward(  # pyrefly: ignore[bad-param-name-override]
        self,
        q,
        swa_k,
        attn_sink,
        *,
        attention_masks=None,
    ) -> torch.Tensor:
        return self._forward_impl(
            q,
            swa_k,
            attn_sink,
            attention_masks=attention_masks,
        )


class HeavilyCompressedAttention(DSV4FlexInnerAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(DSV4FlexInnerAttention.Config):
        pass

    def forward(  # pyrefly: ignore[bad-param-name-override]
        self,
        q,
        swa_k,
        cmp_k,
        attn_sink,
        *,
        attention_masks=None,
    ) -> torch.Tensor:
        return self._forward_impl(
            q,
            swa_k,
            attn_sink,
            cmp_k=cmp_k,
            attention_masks=attention_masks,
        )


class DSV4InnerAttention(InnerAttention):
    """DeepSeek-V4 sparse attention core on Attention Gym's ``gather_attn``.

    Each query attends to its causal sliding window over the uncompressed KV
    (``swa_k``), to the compressed KV positions (``cmp_k``) listed for it in
    ``cmp_topk``, and to a learned per-head attention sink. K and V are the
    same single-head latent.
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
            # Mask compressed blocks that end after the query; -1 marks unused slots.
            limit_T1 = (
                torch.arange(1, q.size(0) + 1, device=q.device) // self.compress_ratio
            ).unsqueeze(1)
            cmp_topk = torch.where(cmp_topk < limit_T1, cmp_topk, -1)
        return self._gather_attn(q, swa_k, cmp_k, cmp_topk, attn_sink, attention_masks)


class Attention(BaseAttention):
    """DeepSeek V4 attention wrapper around sparse inner attention.

    The module projects Q/KV, applies pre- and post-phase RoPE, prepares
    optional compressed/indexer tensors, and delegates sparse attention to
    ``DSV4FlexInnerAttention``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        dim: int
        n_heads: int
        inner_attention: (  # pyrefly: ignore [bad-override]
            DSV4FlexInnerAttention.Config | DSV4InnerAttention.Config
        )
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
