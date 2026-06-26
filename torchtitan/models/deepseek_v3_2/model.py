# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Replicate
from torch.nn.attention.flex_attention import (
    and_masks,
    BlockMask,
    create_block_mask,
    create_mask,
)

from torchtitan.models.common.attention import AttentionMasksType, FlexAttention
from torchtitan.models.common.nn_modules import LayerNorm, Linear
from torchtitan.models.common.rope import RoPE
from torchtitan.models.deepseek_v3.model import (
    Attention as _V3Attention,
    DeepSeekV3Model,
)
from torchtitan.protocols.module import Module

__all__ = [
    "Attention",
    "DeepSeekV32Model",
    "DSAFlexAttention",
    "Indexer",
]


@functools.cache
def _hadamard(dim: int) -> torch.Tensor:
    """Sylvester Hadamard matrix for power-of-two dim (torch-native)."""
    assert dim & (dim - 1) == 0, "Hadamard dim must be a power of two"
    H = torch.ones(1, 1)
    while H.shape[0] < dim:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H


class Indexer(Module):
    """Lightning Indexer for DeepSeek Sparse Attention (DSA).

    Produces index q/k/weights via linear projections + RoPE + Hadamard.
    The BlockMask-aware scoring + top-k lives in the static ``select``
    method.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        q_lora_rank: int
        index_n_heads: int
        index_head_dim: int
        rope_head_dim: int
        index_topk: int
        wq_b: Linear.Config
        wk: Linear.Config
        k_norm: LayerNorm.Config
        weights_proj: Linear.Config
        rope: RoPE.Config

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.rope_head_dim
        self.index_topk = config.index_topk

        self.wq_b = config.wq_b.build()
        self.wk = config.wk.build()
        self.k_norm = config.k_norm.build()
        self.weights_proj = config.weights_proj.build()
        self.rope = config.rope.build()

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor | None = None,
    ):
        bsz, seqlen, _ = x.size()

        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        q_pe, _ = self.rope(q_pe, q_pe, positions)
        idx_q = Indexer._hadamard_rotate(torch.cat([q_pe, q_nope], dim=-1))

        k = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        k_pe, _ = self.rope(k_pe.unsqueeze(2), k_pe.unsqueeze(2), positions)
        idx_k = Indexer._hadamard_rotate(torch.cat([k_pe.squeeze(2), k_nope], dim=-1))

        idx_w = self.weights_proj(x) * (self.n_heads**-0.5)
        idx_w = idx_w * (self.head_dim**-0.5)

        return idx_q, idx_w, idx_k

    @staticmethod
    def select(
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_w: torch.Tensor,
        block_mask: BlockMask,
        index_topk: int,
    ) -> torch.Tensor:
        """BlockMask-aware: ReLU-scores, masked with ``bm.mask_mod``
        (causal + document), top-k -> token-exact ``selected`` bool."""
        B, Lq, _, _ = idx_q.shape
        Lkv = idx_k.shape[1]

        scores = torch.relu(
            torch.einsum("blhd,bsd->blhs", idx_q.float(), idx_k.float())
        )
        index_scores = (scores * idx_w.unsqueeze(-1).float()).sum(dim=2)

        valid = create_mask(
            block_mask.mask_mod, B, 1, Lq, Lkv, device=idx_q.device
        )[:, 0]
        index_scores = index_scores.masked_fill(~valid, float("-inf"))

        k = min(index_topk, Lkv)
        topk_idx = index_scores.topk(k, dim=-1).indices
        selected = torch.zeros(B, Lq, Lkv, dtype=torch.bool, device=idx_q.device)
        return selected.scatter_(-1, topk_idx, True)

    @staticmethod
    def _hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
        d = x.size(-1)
        H = _hadamard(d).to(device=x.device, dtype=x.dtype)
        if isinstance(x, DTensor):
            H = DTensor.from_local(
                H, x.device_mesh, [Replicate()] * x.device_mesh.ndim, run_check=False
            )
        return F.linear(x.reshape(-1, d), H, None).reshape(x.shape) * (d ** -0.5)


class DSAFlexAttention(FlexAttention):
    """Fused sparse MLA inner attention.

    Inherits ``FlexAttention`` for its document-packing mask pipeline.
    The first three positional args are ``q, k, v``;
    ``idx_q, idx_k, idx_w`` follow.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(FlexAttention.Config):
        index_topk: int

    def __init__(self, config: Config):
        super().__init__(config)
        self.index_topk = config.index_topk

    @staticmethod
    def build_dsa_block_mask(selected: torch.Tensor, bm: BlockMask) -> BlockMask:
        """Compose causal+document ``bm`` with DSA token selection into a
        combined BlockMask with block-skip sparsity.

        ``bm.mask_mod`` stays intact (handles document packing);
        only the DSA token test is AND'd on top. The combined mask is fed to
        ``create_block_mask(_compile=True)`` for fast per-layer construction.
        """

        def dsa_token_mask(b, h, q_idx, kv_idx):
            return selected[b, q_idx, kv_idx]

        combined = and_masks(bm.mask_mod, dsa_token_mask)
        return create_block_mask(
            combined,
            B=None,
            H=None,
            Q_LEN=selected.shape[1],
            KV_LEN=selected.shape[2],
            device=selected.device,
            BLOCK_SIZE=bm.BLOCK_SIZE,
            _compile=True,
        )

    def forward(
        self,
        q_BLNH: torch.Tensor,
        k_BLNH: torch.Tensor,
        v_BLNH: torch.Tensor,
        idx_q_BLNH: torch.Tensor,
        idx_k_BLH: torch.Tensor,
        idx_w_BLN: torch.Tensor,
        *,
        attention_masks: BlockMask,
        scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        selected = Indexer.select(
            idx_q_BLNH, idx_k_BLH, idx_w_BLN,
            attention_masks, self.index_topk,
        )
        dsa_bm = DSAFlexAttention.build_dsa_block_mask(selected, attention_masks)
        return super().forward(
            q_BLNH, k_BLNH, v_BLNH,
            attention_masks=dsa_bm,
            scale=scale,
        )


class Attention(_V3Attention):
    """DeepSeek V3.2 MLA attention with Lightning Indexer.

    Computes main MLA q/k/v + indexer projections (outer),
    then delegates to ``DSAFlexAttention`` (inner) for
    BlockMask-aware sparse attention.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(_V3Attention.Config):
        indexer: Indexer.Config | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        if config.indexer is not None:
            self.indexer = config.indexer.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType,
        positions: torch.Tensor | None = None,
    ):
        bsz, seqlen, _ = x.size()

        q = self.wq_a(x)
        qr = self.q_norm(q)
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, -1, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        kv = self.wkv_a(x)
        kv_comp, k_pe = torch.split(
            kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        q_pe, k_pe = self.rope(q_pe, k_pe.unsqueeze(2), positions)
        q = torch.cat([q_nope, q_pe], dim=-1)

        kv = self.wkv_b(self.kv_norm(kv_comp))
        kv = kv.view(bsz, seqlen, -1, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)

        idx_q, idx_w, idx_k = self.indexer(
            x.detach(), qr.detach(), positions=positions
        )

        output = self.inner_attention(
            q, k, v,
            idx_q, idx_k, idx_w,
            attention_masks=attention_masks,
            scale=self.softmax_scale,
        )
        output = output.contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class DeepSeekV32Model(DeepSeekV3Model):
    """DeepSeek-V3.2 Transformer model with MLA, sparse attention, and MoE."""

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV3Model.Config):
        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            super().update_from_config(config=config, **kwargs)

            from torchtitan.models.deepseek_v3_2.sharding import (
                set_deepseek_v3_2_sharding_config,
            )

            set_deepseek_v3_2_sharding_config(self)
