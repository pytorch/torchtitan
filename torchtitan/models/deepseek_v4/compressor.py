# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import cache

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.common.nn_modules import Linear, RMSNorm
from torchtitan.models.common.rope import RoPE, SingleComplexRoPE
from torchtitan.protocols.module import Module


@cache
def _hadamard(dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if dim & (dim - 1) != 0:
        raise ValueError("Hadamard dim must be a power of two")
    h = torch.ones((1, 1), dtype=dtype, device=device)
    while h.shape[0] < dim:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


class Compressor(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        rope: RoPE.Config
        single_rope: SingleComplexRoPE.Config
        head_dim: int = 512
        rope_head_dim: int = 64
        compress_ratio: int = 4
        rotate: bool = False
        norm_eps: float = 1e-6

        wkv: Linear.Config | None = None
        wgate: Linear.Config | None = None
        norm: RMSNorm.Config | None = None
        ape: Linear.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.nope_head_dim = cfg.head_dim - cfg.rope_head_dim
        self.compress_ratio = cfg.compress_ratio
        self.overlap = cfg.compress_ratio == 4
        self.rotate = cfg.rotate
        self.rope = cfg.rope.build()
        self.single_rope = cfg.single_rope.build()

        self.wkv = cfg.wkv.build()
        self.wgate = cfg.wgate.build()
        self.norm = cfg.norm.build()
        # ape is stored as a Linear holding a (ape_rows, ape_cols) weight;
        # the forward path accesses .weight directly.
        self.ape = cfg.ape.build()

    def _overlap_transform(self, tensor, value=0):
        ratio, d = self.compress_ratio, self.head_dim
        prev = torch.cat(
            [
                torch.full_like(tensor[:, :1, :, :d], value),
                tensor[:, :-1, :, :d],
            ],
            dim=1,
        )
        curr = tensor[:, :, :, d:]
        return torch.cat([prev, curr], dim=2)

    def forward(self, x, positions=None):
        _, seqlen, _ = x.size()
        rd = self.rope_head_dim
        ratio = self.compress_ratio
        dtype = x.dtype
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            kv = self.wkv(x)
            score = self.wgate(x)
        if seqlen % ratio != 0:
            raise ValueError(
                f"seqlen ({seqlen}) must be divisible by compress_ratio ({ratio})"
            )
        if positions is not None:
            comp_positions = positions[:, ::ratio]
        else:
            comp_positions = torch.arange(
                0, seqlen, ratio, device=x.device
            ).unsqueeze(0)
        kv = kv.unflatten(1, (-1, ratio))
        score = score.unflatten(1, (-1, ratio)) + self.ape.weight
        if self.overlap:
            kv = self._overlap_transform(kv, 0)
            score = self._overlap_transform(score, float("-inf"))
        kv = (kv * score.softmax(dim=2)).sum(dim=2)
        kv = self.norm(kv.to(dtype))
        kv_nope, kv_rope = torch.split(kv, [self.head_dim - rd, rd], dim=-1)
        kv_rope = self.single_rope(kv_rope.unsqueeze(2), comp_positions)
        kv = torch.cat([kv_nope, kv_rope.squeeze(2)], dim=-1)
        return kv


class Indexer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        rope: RoPE.Config
        single_rope: SingleComplexRoPE.Config
        num_index_heads: int = 64
        index_head_dim: int = 128
        index_topk: int = 512
        rope_head_dim: int = 64
        q_lora_rank: int = 1024
        compress_ratio: int = 4
        norm_eps: float = 1e-6

        wq_b: Linear.Config | None = None
        weights_proj: Linear.Config | None = None
        compressor: "Compressor.Config | None" = None

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.dim = cfg.dim
        self.num_index_heads = cfg.num_index_heads
        self.head_dim = cfg.index_head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.index_topk = cfg.index_topk
        self.softmax_scale = cfg.index_head_dim**-0.5
        self.compress_ratio = cfg.compress_ratio
        self.rope = cfg.rope.build()
        self.single_rope = cfg.single_rope.build()

        self.wq_b = cfg.wq_b.build()
        self.weights_proj = cfg.weights_proj.build()
        self.compressor = cfg.compressor.build()

    @staticmethod
    def _rotate_activation(x):
        dim = x.size(-1)
        hadamard_mat = _hadamard(dim, dtype=x.dtype, device=x.device)
        return F.linear(x, hadamard_mat) * (dim**-0.5)

    def forward(
        self,
        x,
        qr,
        *,
        positions=None,
        offset: int,
    ):
        bsz, seqlen, _ = x.size()
        rd = self.rope_head_dim
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.num_index_heads, self.head_dim)
        q_nope, q_rope = torch.split(q, [self.head_dim - rd, rd], dim=-1)
        q_rope = self.single_rope(q_rope, positions)
        q = torch.cat([q_nope, q_rope], dim=-1)
        q = self._rotate_activation(q)
        k = self.compressor(x, positions=positions)
        k = self._rotate_activation(k)
        weights = self.weights_proj(x) * (self.softmax_scale * self.num_index_heads**-0.5)
        index_score = torch.einsum("bshd,btd->bsht", q, k)
        index_score = index_score.relu_() * weights.unsqueeze(-1)
        index_score = index_score.sum(dim=2)

        ratio = self.compress_ratio
        compress_causal_limit = (
            torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
        )
        compress_causal_mask = (
            torch.arange(seqlen // ratio, device=x.device).repeat(seqlen, 1)
            >= compress_causal_limit
        )
        index_score = index_score + torch.where(
            compress_causal_mask, torch.finfo(q.dtype).min, 0
        )
        index_score, topk_idxs = index_score.topk(
            min(self.index_topk, seqlen // ratio), dim=-1
        )
        mask = topk_idxs >= compress_causal_limit
        compress_topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        return compress_topk_idxs, index_score
