# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.common.attention import BaseAttention
from torchtitan.models.common.nn_modules import Linear, RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import Module

from .compressor import Compressor, Indexer


class GetWindowTopkIdxs(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__()

    def forward(self, window_size: int, bsz: int, seqlen: int, device):
        base = torch.arange(seqlen, device=device).unsqueeze(1)
        window_topk = (base - window_size + 1).clamp(0) + torch.arange(
            min(seqlen, window_size), device=device
        )
        window_topk = torch.where(window_topk > base, -1, window_topk)
        return window_topk.unsqueeze(0).expand(bsz, -1, -1)


class GetCompressTopkIdxs(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        ratio: int = 1

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.ratio = config.ratio

    def forward(self, bsz: int, seqlen: int, offset: int, device):
        matrix = torch.arange(seqlen // self.ratio, device=device).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // self.ratio
        compress_topk = torch.where(mask, -1, matrix + offset)
        return compress_topk.unsqueeze(0).expand(bsz, -1, -1)


class LiCompute(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        ratio: int
        index_topk: int

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.ratio = config.ratio
        self.index_topk = config.index_topk

    def forward(self, q_indexer, k_indexer, weights, *, seqlen, offset):
        index_score = torch.einsum("bshd,btd->bsht", q_indexer, k_indexer)
        index_score = index_score.relu_() * weights.unsqueeze(-1)
        index_score = index_score.sum(dim=2)
        device = index_score.device
        base = torch.arange(seqlen, device=device).unsqueeze(1)
        mask = (
            torch.arange(seqlen // self.ratio, device=device).unsqueeze(0)
            >= (base + 1) // self.ratio
        )
        index_score += torch.where(mask, torch.finfo(q_indexer.dtype).min, 0)
        index_score, topk_idxs = index_score.topk(
            min(self.index_topk, seqlen // self.ratio), dim=-1
        )
        mask = topk_idxs >= (base + 1) // self.ratio
        compress_topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        return compress_topk_idxs, index_score


class SparseAttention_1(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        window_size: int
        compress_ratio: int
        softmax_scale: float

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.window_size = config.window_size
        self.compress_ratio = config.compress_ratio
        self.softmax_scale = config.softmax_scale
        self.get_window_topk_idxs = GetWindowTopkIdxs.Config().build()
        self.get_compress_topk_idxs = GetCompressTopkIdxs.Config(
            ratio=config.compress_ratio
        ).build()

    def forward(self, query_states, kv_states, attn_sink, *, kv_compress=None,
                compress_topk_idxs=None):
        bsz, seqlen, _, _ = query_states.size()

        topk_idxs = self.get_window_topk_idxs(self.window_size, bsz, seqlen, query_states.device)
        if self.compress_ratio > 1:
            offset = kv_states.size(1)
            if compress_topk_idxs is None:
                compress_topk_idxs = self.get_compress_topk_idxs(
                    bsz, seqlen, offset, query_states.device
                )
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs.to(topk_idxs.device)], dim=-1)
        topk_idxs = topk_idxs.int()

        if self.compress_ratio > 1 and kv_compress is not None:
            kv_states = torch.cat([kv_states, kv_compress], dim=1)

        query_states = query_states.transpose(1, 2)
        kv_states = kv_states.unsqueeze(1)
        attn_weights = torch.matmul(query_states, kv_states.transpose(2, 3)) * self.softmax_scale

        topk_idxs.masked_fill_(topk_idxs < 0, kv_states.shape[2])
        index_mask = torch.full(
            (query_states.shape[0], 1, query_states.shape[2], kv_states.shape[2] + 1),
            fill_value=torch.finfo(torch.bfloat16).min,
            dtype=torch.bfloat16, device=query_states.device,
        ).scatter_(-1, topk_idxs.unsqueeze(1), 0)

        attn_weights = attn_weights + index_mask[..., :-1]
        sinks = attn_sink.reshape(1, -1, 1, 1).expand(
            query_states.shape[0], -1, query_states.shape[-2], -1
        )
        combined_logits = torch.cat([attn_weights, sinks], dim=-1)
        combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        probs = F.softmax(combined_logits.float(), dim=-1).to(combined_logits.dtype)
        scores = probs[..., :-1]
        attn_output = torch.matmul(scores, kv_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output


class SparseAttention_4(SparseAttention_1):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        window_size: int
        compress_ratio: int
        softmax_scale: float
    def __init__(self, config: Config) -> None:
        super().__init__(config)
    def forward(self, query_states, kv_states, attn_sink, kv_compress=None,
                compress_topk_idxs=None):
        output = super().forward(query_states, kv_states, attn_sink,
            kv_compress=kv_compress,
            compress_topk_idxs=compress_topk_idxs
        )
        return output


class SparseAttention_128(SparseAttention_1):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        window_size: int
        compress_ratio: int
        softmax_scale: float
    def __init__(self, config: Config) -> None:
        super().__init__(config)
    def forward(self, query_states, kv_states, attn_sink, kv_compress=None,
                compress_topk_idxs=None):
        attn_output = super().forward(query_states, kv_states, attn_sink,
            kv_compress=kv_compress,
            compress_topk_idxs=compress_topk_idxs
        )
        return attn_output



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
        sparse_attn: SparseAttention_1.Config | None = None
        li_compute: LiCompute.Config | None = None

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
        
        if cfg.li_compute is not None:
            self.li_compute = cfg.li_compute.build()

        self._dsa_loss_tracker = None

    def set_dsa_loss_tracker(self, tracker):
        self._dsa_loss_tracker = tracker

    def _pre_phase(self, x, positions):
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

        kv_compress = q_indexer = k_indexer = weights = None

        if self.compress_ratio > 1 and hasattr(self, "indexer"):
            q_indexer, k_indexer, weights = self.indexer(
                x.detach(), qr.detach(), positions=positions,
            )

        if self.compress_ratio == 4:
            kv_compress = self.compressor(x, positions=positions)
        elif self.compress_ratio > 1:
            kv_compress = self.compressor_128(x, positions=positions)

        return q, kv, kv_compress, q_indexer, k_indexer, weights

    def _inner_phase(self, q, kv, kv_compress, q_indexer, k_indexer, weights,
                     seqlen, attention_masks):
        offset = kv.size(1)
        compress_topk_idxs = index_score = None
        has_li = (
            self.compress_ratio > 1
            and hasattr(self, "li_compute")
            and q_indexer is not None
        )
        if has_li:
            compress_topk_idxs, index_score = self.li_compute(
                q_indexer, k_indexer, weights, seqlen=seqlen, offset=offset,
            )
        attn_sink_param = self.attn_sink.weight.squeeze(-1)
        if self.compress_ratio == 1:
            o = self.sparse_attn(
                q, kv, attn_sink_param,
            )
        elif self.compress_ratio == 4:
            o = self.sparse_attn(
                q, kv, attn_sink_param, kv_compress, compress_topk_idxs,
            )
        else:
            o = self.sparse_attn(
                q, kv, attn_sink_param, kv_compress, 
            )
        return o

    def _post_phase(self, o, bsz, seqlen, positions):
        rd = self.rope_head_dim
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

    def forward(self, x, attention_masks=None, positions=None):
        bsz, seqlen, _ = x.size()
        q, kv, kv_compress, q_indexer, k_indexer, weights = self._pre_phase(
            x, positions=positions,
        )

        o = self._inner_phase(
            q, kv, kv_compress, q_indexer, k_indexer, weights,
            seqlen, attention_masks,
        )

        return self._post_phase(o, bsz, seqlen, positions=positions)
