# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import cast

import torch
from torch import nn
from torch.nn.attention.flex_attention import and_masks, BlockMask
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.attention import (
    create_attention_mask,
    FlexAttentionWrapper,
    get_causal_mask_mod,
    get_document_mask_mod,
    ScaledDotProductAttentionWrapper,
)
from torchtitan.models.common import FeedForward, RoPE
from torchtitan.models.common.moe import build_moe, MoE
from torchtitan.models.common.rope import apply_rotary_emb_single_complex
from torchtitan.models.utils import get_moe_model_nparams_and_flops, trunc_normal_
from torchtitan.protocols.model import AttentionMasksType, BaseModel
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability


class Attention(nn.Module):
    """
    Multi-head latent attention (MLA) module.

    This is DeepSeek V3-specific and NOT shared with other models.
    """

    def __init__(self, config: "DeepSeekV3Model.Config"):
        super().__init__()
        self.dim = config.dim
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

        if config.max_seq_len > config.rope_config.original_seq_len:
            mscale = (
                0.1 * config.mscale * math.log(config.rope_config.rope_factor) + 1.0
            )
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.attn_type = config.attn_type
        self.inner_attention: nn.Module
        match self.attn_type:
            case "flex":
                self.inner_attention = FlexAttentionWrapper()
            case "sdpa":
                self.inner_attention = ScaledDotProductAttentionWrapper()
            case "varlen":
                raise ValueError("Varlen attention is not supported with Deepseek V3.")
            case _:
                raise ValueError(f"Unknown attention type: {self.attn_type}")

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            attention_masks (AttentionMasksType | None): Masks used when calculating attention scores.
            positions (torch.Tensor | None): Position indices used to access/shuffle RoPE cache. Defaults to None.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()

        # Query projection
        if self.q_lora_rank == 0:
            q = self.wq(x)  # (bsz, seqlen, n_heads * qk_head_dim)
        else:
            q = self.wq_a(x)
            q = self.wq_b(self.q_norm(q))
        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of q and kv as TP may have sharded them after
        # the above linear ops.
        q = q.view(bsz, seqlen, -1, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_pe = apply_rotary_emb_single_complex(q_pe, freqs_cis, positions)
        q = torch.cat([q_nope, q_pe], dim=-1)  # (bsz, seqlen, n_heads, qk_head_dim)

        # Key-value projection
        kv = self.wkv_a(x)  # (bsz, seqlen, kv_lora_rank + qk_rope_head_dim)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pe = apply_rotary_emb_single_complex(
            k_pe.unsqueeze(2), freqs_cis, positions
        )  # (bsz, seqlen, 1, qk_rope_head_dim)

        kv = self.wkv_b(
            self.kv_norm(kv)
        )  # (bsz, seqlen, n_heads * (qk_nope_head_dim + v_head_dim))
        kv = kv.view(bsz, seqlen, -1, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat(
            [k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1
        )  # (bsz, seqlen, n_heads, qk_head_dim)

        q = q.transpose(1, 2)  # (bsz, n_heads, seqlen, qk_head_dim)
        k = k.transpose(1, 2)  # (bsz, n_heads, seqlen, qk_head_dim)
        v = v.transpose(1, 2)  # (bsz, n_heads, seqlen, v_head_dim)

        match self.attn_type:
            case "flex":
                assert isinstance(attention_masks, BlockMask)
                output = self.inner_attention(
                    q, k, v, block_mask=attention_masks, scale=self.softmax_scale
                )
            case _:
                assert attention_masks is None
                output = self.inner_attention(q, k, v, scale=self.softmax_scale)

        # Reshape and project output
        output = output.transpose(
            1, 2
        ).contiguous()  # (bsz, seqlen, n_heads, v_head_dim)
        output = output.view(bsz, seqlen, -1)  # (bsz, seqlen, n_heads * v_head_dim)
        return self.wo(output)  # (bsz, seqlen, dim)

    def init_weights(self, init_std: float):
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


class TransformerBlock(nn.Module):
    """
    Transformer block with attention and feed-forward layers.
    """

    def __init__(self, layer_id: int, config: "DeepSeekV3Model.Config"):

        super().__init__()
        self.attention = Attention(config)
        self.attention_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)

        self.moe_enabled = layer_id >= config.n_dense_layers
        if self.moe_enabled:
            self.moe = build_moe(
                config=config.moe_config,
                dim=config.dim,
                moe_impl=config.moe_impl,
            )
        else:
            self.feed_forward = config.ff_config.build(dim=config.dim)

        self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        self.layer_id = layer_id

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            attention_masks (AttentionMasksType | None): Masks used when calculating attention scores.
            positions (torch.Tensor | None): Position indices used to access/shuffle RoPE cache. Defaults to None.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        x = x + self.attention(
            self.attention_norm(x), freqs_cis, attention_masks, positions
        )
        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            cast(MoE, self.moe).init_weights(self.weight_init_std, buffer_device)
        else:
            self.feed_forward.init_weights(self.weight_init_std)


class DeepSeekV3Model(BaseModel):
    """
    DeepSeek-V3 Transformer model with attention and feed-forward layers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        dim: int = 2048
        n_layers: int = 27
        n_heads: int = 16
        vocab_size: int = 102400
        norm_eps: float = 1e-5
        max_seq_len: int = 4096 * 4
        attn_mask_type: str = "causal"
        n_dense_layers: int = 1

        # MoE
        moe_config: MoE.Config = field(default_factory=MoE.Config)

        # Expert parallel communication backend (set dynamically in update_from_config)
        moe_impl: str = "standard"

        # Multi-Head Latent Attention (MLA)
        q_lora_rank: int = 0
        kv_lora_rank: int = 512
        qk_nope_head_dim: int = 128
        qk_rope_head_dim: int = 64
        v_head_dim: int = 128
        attn_type: str = "sdpa"
        mscale: float = 1.0

        # Sub-component configs
        ff_config: FeedForward.Config = field(
            default_factory=lambda: FeedForward.Config(hidden_dim=10944)
        )
        rope_config: RoPE.Config = field(
            default_factory=lambda: RoPE.Config(
                dim=64,
                max_seq_len=4096 * 4,
                theta=10000.0,
                format="complex",
                scaling="yarn",
                rope_factor=40.0,
                beta_fast=32.0,
                beta_slow=1.0,
                original_seq_len=4096,
            )
        )

        def update_from_config(
            self,
            *,
            job_config,
            **kwargs,
        ) -> None:
            training = job_config.training
            parallelism = job_config.parallelism
            debug = job_config.debug
            seq_len = training.seq_len
            if seq_len > self.max_seq_len:
                logger.warning(
                    f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
                )
            self.max_seq_len = seq_len

            # Sync rope_config max_seq_len
            import dataclasses as _dc

            self.rope_config = _dc.replace(self.rope_config, max_seq_len=seq_len)

            if self.moe_config.use_grouped_mm and not has_cuda_capability(9, 0):
                logger.warning(
                    "Failed to use grouped mm, which is only supported on SM90 or later",
                )
                self.moe_config.use_grouped_mm = False

            if parallelism.context_parallel_degree > 1 and self.attn_type != "sdpa":
                raise NotImplementedError(
                    f"Context Parallel only supports SDPA attention. "
                    f"Got attn_type='{self.attn_type}'. "
                    f"FlexAttention and varlen attention are not supported with CP."
                )

            self.moe_config._debug_force_load_balance = debug.moe_force_load_balance

            # Configure expert parallel communication backend from config
            self.moe_impl = parallelism.expert_parallel_comm_backend

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            return get_moe_model_nparams_and_flops(
                self,
                model,
                self.n_heads,
                self.qk_nope_head_dim + self.qk_rope_head_dim + self.v_head_dim,
                seq_len,
            )

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

        self.rope = config.rope_config.build()
        self.register_buffer("freqs_cis", self.rope.cache, persistent=False)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(config.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, config)

        self.norm = nn.RMSNorm(config.dim)
        self.output = nn.Linear(
            config.dim,
            config.vocab_size,
            dtype=torch.get_default_dtype(),
            bias=False,
        )

    def init_weights(
        self, *, buffer_device: torch.device | None = None, **kwargs
    ) -> None:
        buffer_device = buffer_device or self.freqs_cis.device
        self.rope.init_weights(buffer_device=buffer_device)
        self.freqs_cis = self.rope.cache
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                cast(TransformerBlock, layer).init_weights(buffer_device=buffer_device)
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.config.dim**-0.5
        cutoff_factor = 3
        if self.output is not None:
            trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        mask_mods = [get_causal_mask_mod()]
        match self.config.attn_mask_type:
            case "causal":
                B = 1
            case "block_causal":
                B = input_batch.shape[0]
                assert tokenizer.eos_id is not None
                mask_mods.append(get_document_mask_mod(input_batch, tokenizer.eos_id))
            case _:
                raise ValueError(
                    f"Unknown attention mask type: {self.config.attn_mask_type}"
                )
        return create_attention_mask(
            and_masks(*mask_mods), B, None, input_batch.shape[1], input_batch.shape[1]
        )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices if pipeline parallelism is not enabled.
                If pipeline parallelism is enabled, this will be the input token indices
                for the ranks on the first pipeline stage. This will be the activation of the
                previous pipeline stage if the current rank is not on the first stage.
            attention_masks (AttentionMasksType | None): Masks used when calculating attention scores.
            positions (torch.Tensor | None): Position indices used to access/shuffle RoPE cache. Defaults to None.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """

        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis, attention_masks, positions)
        h = self.norm(h) if self.norm is not None else h
        output = self.output(h) if self.output is not None else h
        return output
