# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.models.common import trunc_normal_
from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    FlexAttentionWrapper,
    ScaledDotProductAttentionWrapper,
)
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.rope import apply_rotary_emb_single_complex
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability


class Attention(BaseAttention):
    """
    Multi-head latent attention (MLA) module.

    This is DeepSeek V3-specific and NOT shared with other models.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int
        q_lora_rank: int = 0
        kv_lora_rank: int = 512
        qk_nope_head_dim: int = 128
        qk_rope_head_dim: int = 64
        v_head_dim: int = 128
        norm_eps: float = 1e-5
        attn_backend: str = "sdpa"
        attn_mask_type: str = "causal"
        mscale: float = 1.0
        rope_factor: float = 1.0
        rope_max_seq_len: int = 4096
        rope_original_seq_len: int = 4096

    def __init__(self, config: Config, *, dim: int):
        super().__init__()
        self.dim = dim
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=False)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
            self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=config.norm_eps)
            self.wq_b = nn.Linear(
                self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False
            )
        self.wkv_a = nn.Linear(
            self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=config.norm_eps)
        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)
        self.softmax_scale = self.qk_head_dim**-0.5

        if config.rope_max_seq_len > config.rope_original_seq_len:
            mscale = 0.1 * config.mscale * math.log(config.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.attn_backend = config.attn_backend
        self.inner_attention: nn.Module
        match self.attn_backend:
            case "flex":
                self.inner_attention = FlexAttentionWrapper()
            case "sdpa":
                self.inner_attention = ScaledDotProductAttentionWrapper()
            case "varlen":
                raise ValueError("Varlen attention is not supported with Deepseek V3.")
            case _:
                raise ValueError(f"Unknown attention backend: {self.attn_backend}")

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        bsz, seqlen, _ = x.size()

        # Query projection
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_a(x)
            q = self.wq_b(self.q_norm(q))
        q = q.view(bsz, seqlen, -1, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_pe = apply_rotary_emb_single_complex(q_pe, freqs_cis, positions)
        q = torch.cat([q_nope, q_pe], dim=-1)

        # Key-value projection
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pe = apply_rotary_emb_single_complex(k_pe.unsqueeze(2), freqs_cis, positions)

        kv = self.wkv_b(self.kv_norm(kv))
        kv = kv.view(bsz, seqlen, -1, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        match self.attn_backend:
            case "flex":
                assert isinstance(attention_masks, BlockMask)
                output = self.inner_attention(
                    q, k, v, block_mask=attention_masks, scale=self.softmax_scale
                )
            case _:
                assert attention_masks is None
                output = self.inner_attention(q, k, v, scale=self.softmax_scale)

        output = output.transpose(1, 2).contiguous()
        output = output.view(bsz, seqlen, -1)
        return self.wo(output)

    def init_weights(self, **kwargs) -> None:
        init_std = kwargs.get("init_std")
        assert init_std is not None
        linear_list = [
            self.wkv_a,
            self.wkv_b,
        ]
        if self.q_lora_rank > 0:
            linear_list.extend([self.wq_a, self.wq_b])
        else:
            linear_list.append(self.wq)

        for linear in linear_list:
            trunc_normal_(linear.weight, mean=0.0, std=0.02)
        trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

        self.kv_norm.reset_parameters()
        if self.q_lora_rank > 0:
            self.q_norm.reset_parameters()


class DeepSeekV3TransformerBlock(TransformerBlock):
    """
    DeepSeek V3 Transformer block with attention and feed-forward layers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        n_dense_layers: int = 1

    def __init__(self, config: Config, *, layer_id: int, dim: int, n_layers: int):
        super().__init__()
        self.attention = config.attention.build(dim=dim)
        self.attention_norm = nn.RMSNorm(dim, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=config.norm_eps)

        self.moe_enabled = layer_id >= config.n_dense_layers
        if self.moe_enabled:
            assert config.moe is not None
            self.moe = config.moe.build(dim=dim)
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build(dim=dim)

        self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        self.layer_id = layer_id

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        x = x + self.attention(
            self.attention_norm(x), freqs_cis, attention_masks, positions
        )
        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x

    def init_weights(self, **kwargs):
        buffer_device = kwargs.get("buffer_device")
        assert buffer_device is not None
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(init_std=self.weight_init_std)
        if self.moe_enabled:
            cast(MoE, self.moe).init_weights(
                init_std=self.weight_init_std, buffer_device=buffer_device
            )
        else:
            self.feed_forward.init_weights(self.weight_init_std)


class DeepSeekV3Model(Decoder):
    """
    DeepSeek-V3 Transformer model with attention and feed-forward layers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 2048
        n_layers: int = 27
        vocab_size: int = 102400
        layer: TransformerBlock.Config

        def update_from_config(
            self,
            *,
            trainer_config,
            **kwargs,
        ) -> None:
            training = trainer_config.training
            parallelism = trainer_config.parallelism
            debug = trainer_config.debug
            seq_len = training.seq_len
            if seq_len > self.rope.max_seq_len:
                logger.warning(
                    f"Sequence length {seq_len} exceeds original maximum {self.rope.max_seq_len}."
                )
            # Sync rope max_seq_len
            import dataclasses as _dc

            self.rope = _dc.replace(self.rope, max_seq_len=seq_len)

            # Sync rope fields to attention
            assert isinstance(self.layer.attention, Attention.Config)
            self.layer.attention = _dc.replace(
                self.layer.attention,
                rope_max_seq_len=seq_len,
                rope_factor=self.rope.rope_factor,
                rope_original_seq_len=self.rope.original_seq_len,
            )

            assert self.layer.moe is not None
            if self.layer.moe.use_grouped_mm and not has_cuda_capability(9, 0):
                logger.warning(
                    "Failed to use grouped mm, which is only supported on SM90 or later",
                )
                self.layer.moe.use_grouped_mm = False

            if (
                parallelism.context_parallel_degree > 1
                and self.layer.attention.attn_backend != "sdpa"
            ):
                raise NotImplementedError(
                    f"Context Parallel only supports SDPA attention. "
                    f"Got attn_backend='{self.layer.attention.attn_backend}'. "
                    f"FlexAttention and varlen attention are not supported with CP."
                )

            self.layer.moe._debug_force_load_balance = debug.moe_force_load_balance

            # Configure expert parallel communication backend from config
            if parallelism.expert_parallel_comm_backend == "deepep":
                from torchtitan.models.common.moe.moe_deepep import DeepEPMoE

                self.layer.moe = DeepEPMoE.Config(**_dc.asdict(self.layer.moe))

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            assert isinstance(self.layer.attention, Attention.Config)
            return get_moe_model_nparams_and_flops(
                self,
                model,
                self.layer.attention.n_heads,
                self.layer.attention.qk_nope_head_dim
                + self.layer.attention.qk_rope_head_dim
                + self.layer.attention.v_head_dim,
                seq_len,
            )
