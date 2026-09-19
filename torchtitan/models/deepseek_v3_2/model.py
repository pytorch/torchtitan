# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from dataclasses import dataclass, field

import spmd_types as spmd
import torch
import torch.nn.functional as F

from torch.nn.attention.flex_attention import BlockMask, create_mask

from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.attention import (
    AttentionMasksType,
    FlexAttention,
)
from torchtitan.models.common.attention import create_attention_mask
from torchtitan.models.common.aux_loss import LoggedAuxLoss
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.nn_modules import BatchedLinear, LayerNorm, Linear
from torchtitan.models.common.rope import RoPE
from torchtitan.models.deepseek_v3.model import (
    Attention as V3Attention,
    DeepSeekV3Model,
)
from torchtitan.protocols.module import Module


@functools.cache
def _hadamard(dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    assert dim & (dim - 1) == 0, "Hadamard dim must be a power of two"
    H = torch.ones((1, 1), dtype=dtype, device=device)
    while H.shape[0] < dim:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H


class Indexer(Module):
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
        with spmd.local():
            q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
            if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
                spmd.assert_type(
                    q,
                    {"dp": spmd.S(0), "cp": spmd.S(1), "tp": spmd.S(2)},
                )
        q_pe, q_nope = torch.split(
            q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        k = self.k_norm(self.wk(x))
        k_pe, k_nope = torch.split(
            k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        q_pe, k_pe = self.rope(q_pe, k_pe.unsqueeze(2), positions)
        idx_q = Indexer._hadamard_rotate(torch.cat([q_pe, q_nope], dim=-1))
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, Lq, _, _ = idx_q.shape
        Lkv = idx_k.shape[1]

        scores = torch.relu(
            torch.einsum("blhd,bsd->blhs", idx_q.float(), idx_k.float())
        )
        index_scores = (scores * idx_w.unsqueeze(-1).float()).sum(dim=2)

        valid = create_mask(
            block_mask.mask_mod, B, 1, Lq, Lkv, device=idx_q.device
        ).squeeze(1)
        index_scores = index_scores.masked_fill(~valid, float("-inf"))

        k = min(index_topk, Lkv)
        topk_scores, topk_indices = index_scores.topk(k, dim=-1)
        return topk_indices.where(topk_scores.isfinite(), -1), index_scores

    @staticmethod
    def _hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
        d = x.size(-1)
        H = _hadamard(d, device=x.device, dtype=x.dtype)
        return F.linear(x.reshape(-1, d), H).reshape(x.shape) * (d ** -0.5)


class SparseIndexerAuxLoss(LoggedAuxLoss):
    """Indexer alignment loss for the sparse training stage.

    In MQA absorb mode (k has 1 KV head, q has H query heads), k is
    broadcast to H heads before computing the target distribution.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(LoggedAuxLoss.Config):
        coeff: float = 1.0
        tag: str = "dsa_indexer_loss"
        reduce_mesh: str = "loss"

    def __init__(self, config: Config):
        super().__init__(config)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        scale: float | None,
        selected: torch.Tensor,
        index_scores: torch.Tensor,
        *,
        carrier: torch.Tensor,
    ) -> torch.Tensor:
        logits = torch.einsum(
            "blhd,bsd->blhs", q.float(), k.float().squeeze(2)
        )
        if scale is not None:
            logits = logits * scale

        logits = logits.masked_fill(
            ~selected.unsqueeze(2), float("-inf")
        )
        p = F.softmax(logits, dim=-1).mean(dim=2)

        scores = index_scores.masked_fill(
            ~selected, float("-inf")
        )
        q_pred = F.softmax(scores, dim=-1)

        eps = 1e-10
        kl_loss = (
            (p * ((p + eps).log() - (q_pred + eps).log()))
            .sum(dim=-1)
            .mean()
        )
        return self.inject(carrier, kl_loss)


class SparseInnerAttention(FlexAttention):
    """Sparse attention core for DSA, with separated nope/rope support."""

    @dataclass(kw_only=True, slots=True)
    class Config(FlexAttention.Config):
        index_topk: int
        indexer_loss: SparseIndexerAuxLoss.Config = field(
            default_factory=SparseIndexerAuxLoss.Config
        )

    def __init__(self, config: Config):
        super().__init__(config)
        self.index_topk = config.index_topk
        self.indexer_loss = config.indexer_loss.build()

    def forward(
        self,
        q_nope: torch.Tensor,
        k_nope: torch.Tensor,
        q_rope: torch.Tensor,
        k_rope: torch.Tensor,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_w: torch.Tensor,
        *,
        attention_masks: BlockMask,
        scale: float,
    ) -> torch.Tensor:
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)
        v = k_nope

        topk_indices, index_scores = Indexer.select(
            idx_q, idx_k, idx_w,
            attention_masks, self.index_topk,
        )

        def _build_selected(
            topk_indices: torch.Tensor,
        ) -> tuple[torch.Tensor, BlockMask]:
            B, Lq = q.shape[:2]
            Lkv = k.shape[1]

            mask = torch.zeros(
                B, Lq, Lkv, dtype=torch.bool, device=topk_indices.device
            ).scatter_add_(
                -1, topk_indices.clamp(min=0), topk_indices != -1
            )

            block_mask = create_attention_mask(
                lambda b, h, q_idx, k_idx: mask[b, q_idx, k_idx],
                B=B, H=None,
                Q_LEN=Lq, KV_LEN=Lkv,
                device=topk_indices.device,
                BLOCK_SIZE=attention_masks.BLOCK_SIZE,
            )
            return mask, block_mask

        selected, selected_bm = _build_selected(topk_indices)

        output = super().forward(
            q, k, v,
            attention_masks=selected_bm,
            scale=scale,
            enable_gqa=True,
        )
        if self.training:
            output = self.indexer_loss(
                q.detach(), k.detach(),
                scale, selected, index_scores, carrier=output,
            )
        return output


class Attention(V3Attention):
    """Multi-head latent attention in MQA absorb mode for DeepSeek-V3.2."""

    @dataclass(kw_only=True, slots=True)
    class Config(V3Attention.Config):
        w_uk: BatchedLinear.Config
        w_uv: BatchedLinear.Config
        indexer: Indexer.Config

    def __init__(self, config: Config):
        super().__init__(config)
        del self.wkv_b

        self.w_uk = config.w_uk.build()
        self.w_uv = config.w_uv.build()
        self.indexer = config.indexer.build()

        self.register_state_dict_post_hook(self._merge_wkv_b_on_save)
        self.register_load_state_dict_pre_hook(self._split_wkv_b_on_load)

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType,
        positions: torch.Tensor | None = None,
    ):
        bsz, seqlen, _ = x.size()

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr)
        with spmd.local():
            q = q.view(bsz, seqlen, -1, self.qk_head_dim)
            if get_spmd_backend() == "spmd_types":
                spmd.assert_type(
                    q,
                    {"dp": spmd.S(0), "cp": spmd.S(1), "tp": spmd.S(2)},
                )

        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        kv = self.wkv_a(x)
        kv_nope, k_pe = torch.split(
            kv.unsqueeze(2), [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv_nope = self.kv_norm(kv_nope)

        q_pe, k_pe = self.rope(q_pe, k_pe, positions)
        q_nope = self.w_uk(q_nope)

        idx_q, idx_w, idx_k = self.indexer(
            x.detach(), qr.detach(), positions=positions
        )

        output = self.inner_attention(
            q_nope, kv_nope,
            q_pe, k_pe,
            idx_q, idx_k, idx_w,
            attention_masks=attention_masks,
            scale=self.softmax_scale,
        )

        output = self.w_uv(output)
        output = output.contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

    @staticmethod
    def _split_wkv_b_on_load(module, state_dict, prefix, *args):
        wkv_key = f"{prefix}wkv_b.weight"
        wkv_b = state_dict.pop(wkv_key)
        wkv_b_3d = wkv_b.view(module.n_heads, -1, module.kv_lora_rank)
        state_dict[f"{prefix}w_uk.weight"] = (
            wkv_b_3d[:, :module.qk_nope_head_dim, :].transpose(-2, -1).contiguous()
        )
        state_dict[f"{prefix}w_uv.weight"] = (
            wkv_b_3d[:, module.qk_nope_head_dim:, :].contiguous()
        )

    @staticmethod
    def _merge_wkv_b_on_save(module, state_dict, prefix, local_metadata):
        w_uk = state_dict.pop(f"{prefix}w_uk.weight")
        w_uv = state_dict.pop(f"{prefix}w_uv.weight")
        wkv_b = torch.cat(
            [w_uk.transpose(-2, -1), w_uv], dim=1
        ).reshape(-1, module.kv_lora_rank)
        state_dict[f"{prefix}wkv_b.weight"] = wkv_b.contiguous()


class DeepSeekV32Model(DeepSeekV3Model):
    """DeepSeek-V3.2 model — identical to V3 but with V3.2 sharding config."""

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV3Model.Config):
        def update_from_config(self, *, config, **kwargs):
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            from torchtitan.models.deepseek_v3_2.sharding import (
                set_deepseek_v3_2_sharding_config,
            )

            set_deepseek_v3_2_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )
