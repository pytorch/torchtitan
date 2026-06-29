# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.common.attention import (
    BaseAttention,
    create_attention_mask,
    FlexAttention,
)
from torchtitan.models.common.nn_modules import Linear, RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import Module

from .compressor import Compressor, Indexer


class DSAFlexAttention(FlexAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(FlexAttention.Config):
        window_size: int
        compress_ratio: int
        softmax_scale: float

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.window_size = config.window_size
        self.compress_ratio = config.compress_ratio
        self.softmax_scale = config.softmax_scale
        self.block_size = config.block_size

    def _create_window_mask(
        self,
        *,
        bsz: int,
        seqlen: int,
        kv_len: int,
        device,
    ):
        sink_idx = kv_len

        def window_mask_mod(b, h, q_idx, kv_idx):
            in_window = (
                (kv_idx < seqlen)
                & (kv_idx <= q_idx)
                & (q_idx - kv_idx < self.window_size)
            )
            is_sink = kv_idx == sink_idx
            return in_window | is_sink

        return create_attention_mask(
            window_mask_mod,
            bsz,
            None,
            seqlen,
            kv_len + 1,
            device=device,
            BLOCK_SIZE=self.block_size,
            separate_full_blocks=False,
        )

    def _create_indexer_block_causal_mask(
        self,
        *,
        compress_topk_idxs: torch.Tensor,
        bsz: int,
        seqlen: int,
        kv_len: int,
        device,
    ):
        compress_len = seqlen // self.compress_ratio
        sink_idx = kv_len
        topk = compress_topk_idxs.size(-1)

        def v4_sparse_mask_mod(b, h, q_idx, kv_idx):
            in_window = (
                (kv_idx < seqlen)
                & (kv_idx <= q_idx)
                & (q_idx - kv_idx < self.window_size)
            )
            selected_compress = compress_topk_idxs[b, q_idx, 0] == kv_idx
            for idx in range(1, topk):
                selected_compress = selected_compress | (
                    compress_topk_idxs[b, q_idx, idx] == kv_idx
                )
            in_compressed_topk = (
                (kv_idx >= seqlen)
                & (kv_idx < seqlen + compress_len)
                & selected_compress
            )
            is_sink = kv_idx == sink_idx
            return in_window | in_compressed_topk | is_sink

        return create_attention_mask(
            v4_sparse_mask_mod,
            bsz,
            None,
            seqlen,
            kv_len + 1,
            device=device,
            BLOCK_SIZE=self.block_size,
            separate_full_blocks=False,
        )

    def _create_compress_causal_mask(
        self,
        *,
        bsz: int,
        seqlen: int,
        kv_len: int,
        device,
    ):
        compress_len = seqlen // self.compress_ratio
        sink_idx = kv_len

        def compress_causal_mask_mod(b, h, q_idx, kv_idx):
            in_window = (
                (kv_idx < seqlen)
                & (kv_idx <= q_idx)
                & (q_idx - kv_idx < self.window_size)
            )
            compress_idx = kv_idx - seqlen
            in_compressed_causal = (
                (kv_idx >= seqlen)
                & (kv_idx < seqlen + compress_len)
                & (compress_idx < (q_idx + 1) // self.compress_ratio)
            )
            is_sink = kv_idx == sink_idx
            return in_window | in_compressed_causal | is_sink

        return create_attention_mask(
            compress_causal_mask_mod,
            bsz,
            None,
            seqlen,
            kv_len + 1,
            device=device,
            BLOCK_SIZE=self.block_size,
            separate_full_blocks=False,
        )

    def forward(
        self,
        query_states,
        kv_states,
        attn_sink,
        kv_compress,
        compress_topk_idxs,
    ):
        if self.compress_ratio > 1 and kv_compress is None:
            raise ValueError(
                "DSAFlexAttention requires kv_compress when compress_ratio > 1"
            )

        bsz, seqlen, _, head_dim = query_states.size()
        if self.compress_ratio > 1:
            kv_states = torch.cat([kv_states, kv_compress], dim=1)
        kv_len = kv_states.size(1)

        sink_kv = kv_states.new_zeros((bsz, 1, head_dim))
        kv_states = torch.cat([kv_states, sink_kv], dim=1)
        key_value_states = kv_states.unsqueeze(2)

        if self.compress_ratio == 4:
            if compress_topk_idxs is None:
                raise ValueError(
                    "DSAFlexAttention block causal mask requires compress_topk_idxs"
                )
            block_mask = self._create_indexer_block_causal_mask(
                compress_topk_idxs=compress_topk_idxs,
                bsz=bsz,
                seqlen=seqlen,
                kv_len=kv_len,
                device=query_states.device,
            )
        elif self.compress_ratio > 1:
            block_mask = self._create_compress_causal_mask(
                bsz=bsz,
                seqlen=seqlen,
                kv_len=kv_len,
                device=query_states.device,
            )
        else:
            block_mask = self._create_window_mask(
                bsz=bsz,
                seqlen=seqlen,
                kv_len=kv_len,
                device=query_states.device,
            )
        sink_idx = kv_len

        def v4_sink_score_mod(score, b, h, q_idx, kv_idx):
            return torch.where(kv_idx == sink_idx, attn_sink[h], score)

        return super().forward(
            query_states,
            key_value_states,
            key_value_states,
            attention_masks=block_mask,
            score_mod=v4_sink_score_mod,
            scale=self.softmax_scale,
            enable_gqa=True,
        )


class GetAttnScores(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__()

    def forward(self, query, key, attention_masks, num_attn_heads, attn_scale):
        if num_attn_heads > 1:
            key = key.repeat_interleave(num_attn_heads, dim=1)
        attn = (query @ key.transpose(-1, -2)) * attn_scale
        if attention_masks is not None:
            attn.masked_fill_(attention_masks, float("-inf"))
        attn = F.softmax(attn.float(), dim=-1)
        attn = attn.sum(dim=1)
        return attn


class Attention(BaseAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        dim: int
        n_heads: int
        inner_attention: Module.Config
        rope: RoPE.Config
        head_dim: int = 512
        rope_head_dim: int = 64
        q_lora_rank: int = 1024
        o_lora_rank: int = 1024
        n_groups: int = 8
        compress_ratio: int = 1
        window_size: int = 128
        norm_eps: float = 1e-6
        index_n_heads: int = 64
        index_head_dim: int = 128
        index_topk: int = 512
        n_layers: int = 4
        layer_id: int = 0
        mask_type: str = "causal"

        # Sub-module configs — declared as fields so the sharding system can
        # set sharding_config on them before build().
        wq_a: Linear.Config | None = None
        q_norm: RMSNorm.Config | None = None
        wq_b: Linear.Config | None = None
        wkv: Linear.Config | None = None
        kv_norm: RMSNorm.Config | None = None
        wo_a: Linear.Config | None = None
        wo_b: Linear.Config | None = None
        attn_sink: Linear.Config | None = None

        # Compressor/indexer are conditional, so keep them here too.
        compressor: Compressor.Config | None = None
        compressor_128: Compressor.Config | None = None
        indexer: Indexer.Config | None = None
        sparse_attn: DSAFlexAttention.Config | None = None

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
        self.window_size = cfg.window_size
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

        self.sparse_attn = cfg.sparse_attn.build()
        self._dsa_loss_tracker = None

    def set_dsa_loss_tracker(self, tracker):
        self._dsa_loss_tracker = tracker

    def forward(self, x, attention_masks=None, positions=None):
        bsz, seqlen, _ = x.size()
        rd = self.rope_head_dim

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.norm_eps)
        q_nope, q_rope = torch.split(q, [self.head_dim - rd, rd], dim=-1)

        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        kv_nope, kv_rope = torch.split(kv, [self.head_dim - rd, rd], dim=-1)

        q_rope, kv_rope = self.rope(q_rope, kv_rope.unsqueeze(2), positions)
        q = torch.cat([q_nope, q_rope], dim=-1)
        kv = torch.cat([kv_nope, kv_rope.squeeze(2)], dim=-1)

        kv_compress = compress_topk_idxs = None

        if self.compress_ratio > 1 and hasattr(self, "indexer"):
            base = torch.arange(seqlen, device=x.device).unsqueeze(1)
            compress_causal_limit = (base + 1) // self.compress_ratio
            compress_causal_mask = (
                torch.arange(
                    seqlen // self.compress_ratio, device=x.device
                ).unsqueeze(0)
                >= compress_causal_limit
            )
            compress_topk_idxs, _ = self.indexer(
                x.detach(), qr.detach(),
                compress_causal_mask, compress_causal_limit,
                positions=positions,
                offset=kv.size(1),
            )

        if self.compress_ratio == 4:
            kv_compress = self.compressor(x, positions=positions)
        elif self.compress_ratio > 1:
            kv_compress = self.compressor_128(x, positions=positions)

        attn_sink_param = self.attn_sink.weight.squeeze(-1)
        if kv_compress is None:
            kv_compress = kv.new_empty((bsz, 0, self.head_dim))
        if compress_topk_idxs is None:
            compress_topk_idxs = torch.empty(
                (bsz, seqlen, 0), dtype=torch.int64, device=x.device
            )
        o = self.sparse_attn(
            q, kv, attn_sink_param, kv_compress, compress_topk_idxs,
        )

        o_nope, o_rope = torch.split(o, [self.head_dim - rd, rd], dim=-1)
        o_rope = self.rope(o_rope, o_rope, positions)[0]
        o = torch.cat([o_nope, o_rope], dim=-1)

        n_local_groups = self.n_groups // (self.n_heads // o.shape[2])
        o = o.view(bsz, seqlen, n_local_groups, -1)
        # wo_a is a Linear module; access its weight directly for the grouped
        # einsum (not a standard Linear forward).
        wo_a = self.wo_a.weight.view(n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return self.wo_b(o.reshape(bsz, seqlen, -1))
