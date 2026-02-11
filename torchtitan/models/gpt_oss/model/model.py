# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn.attention.flex_attention import and_masks, BlockMask
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.attention import (
    create_attention_mask,
    FlexAttentionWrapper,
    get_causal_mask_mod,
    get_document_mask_mod,
    get_sliding_window_mask_mod,
)
from torchtitan.models.common import RoPE
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.rope import apply_rotary_emb_cos_sin
from torchtitan.models.utils import get_moe_model_nparams_and_flops, trunc_normal_
from torchtitan.protocols.model import AttentionMasksType, BaseModel
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

from .moe import GptOssMoE


class Attention(nn.Module):
    """
    Multi-head attention (MLA) module with sink attention.
    """

    def __init__(self, config: "GptOssModel.Config"):
        super().__init__()
        self.head_dim = config.head_dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.enable_gqa = self.n_heads > self.n_kv_heads

        self.n_rep = self.n_heads // self.n_kv_heads

        # Standard attention softmax scale (1/sqrt(head_dim))
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)

        self.wq = nn.Linear(
            config.dim,
            config.n_heads * config.head_dim,
            bias=True,
        )
        self.wk = nn.Linear(
            config.dim,
            config.n_kv_heads * config.head_dim,
            bias=True,
        )
        self.wv = nn.Linear(
            config.dim,
            config.n_kv_heads * config.head_dim,
            bias=True,
        )
        self.wo = nn.Linear(
            config.n_heads * config.head_dim,
            config.dim,
            bias=True,
        )
        self.sinks = nn.Parameter(torch.empty(config.n_heads))
        self.inner_attention = FlexAttentionWrapper()

    def init_weights(self, init_std: float):
        linear_list = [
            self.wq,
            self.wk,
            self.wv,
        ]

        trunc_normal_(self.sinks, mean=0.0, std=init_std)
        for linear in linear_list:
            trunc_normal_(linear.weight, mean=0.0, std=init_std)
            trunc_normal_(linear.bias, mean=0.0, std=init_std)
        trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
        trunc_normal_(self.wo.bias, mean=0.0, std=init_std)

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
    ):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            rope_cache (torch.Tensor): Precomputed cosine and sine frequencies for rope embedding.
            attention_masks: Attention mask (BlockMask).

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()
        hidden_shape = (bsz, seqlen, -1, self.head_dim)

        q = self.wq(x).view(hidden_shape)
        k = self.wk(x).view(hidden_shape)
        v = self.wv(x).view(hidden_shape)

        q, k = apply_rotary_emb_cos_sin(q, k, rope_cache)

        xq = q.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = k.transpose(1, 2)  # (bs, n_kv_heads, seqlen, head_dim)
        xv = v.transpose(1, 2)  # (bs, n_kv_heads, seqlen, head_dim)

        assert isinstance(attention_masks, BlockMask), attention_masks
        output, lse = self.inner_attention(
            xq,
            xk,
            xv,
            block_mask=attention_masks,
            scale=self.softmax_scale,
            return_lse=True,
            enable_gqa=self.enable_gqa,
        )

        # Apply attention sink rescaling: rescale by sigma(lse - w[h])
        # This is mathematically equivalent to concatenating learnable sink weights
        sink_scale = torch.sigmoid(lse - self.sinks.view(1, -1, 1)).unsqueeze(-1)
        output = output * sink_scale.to(output.dtype)

        output = output.transpose(1, 2).contiguous()  # (B, H, T, D) -> (B, T, H, D)

        # Reshape and project output
        output = output.reshape(
            bsz, seqlen, -1
        ).contiguous()  # (bsz, seqlen, n_heads * v_head_dim)
        output = self.wo(output)  # (bsz, seqlen, dim)
        return output


class TransformerBlock(nn.Module):
    """
    Transformer block with attention and feed-forward layers.
    """

    def __init__(self, layer_id: int, config: "GptOssModel.Config"):

        super().__init__()
        self.use_sliding_attention = layer_id % 2 == 0
        self.attention = Attention(config)
        self.attention_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)

        self.moe = GptOssMoE(
            config.moe_config, dim=config.dim, swiglu_limit=config.swiglu_limit
        )
        self.moe_enabled = True  # for composability with load balancing

        self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        self.layer_id = layer_id

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType,
    ):
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            rope_cache (torch.Tensor): Precomputed cosine and sine frequencies.
            attention_masks (AttentionMasksType): a dict of BlockMasks.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        # Extract the appropriate mask for this layer
        if self.use_sliding_attention:
            # pyrefly: ignore [missing-attribute]
            layer_mask = attention_masks.get("sliding_window_mask", None)
        else:
            # pyrefly: ignore [missing-attribute]
            layer_mask = attention_masks.get("basic_mask", None)
        assert layer_mask is not None

        x = x + self.attention(self.attention_norm(x), rope_cache, layer_mask)
        x = x + self.moe(self.ffn_norm(x))
        return x

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.moe.init_weights(self.weight_init_std, buffer_device)


class GptOssModel(BaseModel):
    """
    GPT-OSS Transformer model with attention and feed-forward layers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        dim: int = 2880
        n_layers: int = 24
        n_heads: int = 64
        n_kv_heads: int = 8
        vocab_size: int = 201088
        norm_eps: float = 1e-5
        max_seq_len: int = 131072
        attn_mask_type: str = "causal"
        attn_type: str = "flex"  # NOTE: gpt-oss only supports FlexAttention
        head_dim: int = 64
        sliding_window_size: int = 128

        # MoE
        moe_config: MoE.Config = field(default_factory=MoE.Config)
        swiglu_limit: float = 7.0

        # Sub-component configs
        rope_config: RoPE.Config = field(
            default_factory=lambda: RoPE.Config(
                dim=64,
                max_seq_len=131072,
                theta=150000.0,
                format="cos_sin",
                scaling="yarn",
                rope_factor=32,
                beta_slow=32.0,
                beta_fast=1.0,
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

            if parallelism.context_parallel_degree > 1:
                raise NotImplementedError(
                    "CP support for gpt-oss model is still in progress."
                )

        # pyrefly: ignore [bad-override]
        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, float]:
            return get_moe_model_nparams_and_flops(
                self, model, self.n_heads, 2 * self.head_dim, seq_len
            )

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

        self.rope = config.rope_config.build()
        self.register_buffer("rope_cache", self.rope.cache, persistent=False)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(config.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, config).to(
                torch.bfloat16
            )

        self.norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(
            config.dim,
            config.vocab_size,
            dtype=torch.get_default_dtype(),
            bias=False,
        )

    def init_weights(
        self, *, buffer_device: torch.device | None = None, **kwargs
    ) -> None:
        buffer_device = buffer_device or self.rope_cache.device
        self.rope.init_weights(buffer_device=buffer_device)
        self.rope_cache = self.rope.cache
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                # pyrefly: ignore [not-callable]
                layer.init_weights(buffer_device=buffer_device)
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

        basic_mask_mods = []
        sliding_window_mask_mods = [
            get_sliding_window_mask_mod(self.config.sliding_window_size)
        ]
        match self.config.attn_mask_type:
            case "causal":
                B = 1
                basic_mask_mods.append(get_causal_mask_mod())
            case "block_causal":
                B = input_batch.shape[0]
                assert tokenizer.eos_id is not None
                basic_mask_mods.append(
                    get_document_mask_mod(input_batch, tokenizer.eos_id)
                )
            case _:
                raise ValueError(
                    f"Unknown attention mask type: {self.config.attn_mask_type}"
                )

        # create basic attention mask: causal or block_causal
        basic_mask = create_attention_mask(
            and_masks(*basic_mask_mods),
            B,
            None,
            input_batch.shape[1],
            input_batch.shape[1],
        )

        # create sliding window mask, has to be created on top of basic attention mask
        sliding_window_mask = create_attention_mask(
            and_masks(*basic_mask_mods, *sliding_window_mask_mods),
            B,
            None,
            input_batch.shape[1],
            input_batch.shape[1],
        )

        return {"basic_mask": basic_mask, "sliding_window_mask": sliding_window_mask}

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType,
    ):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            attention_masks (AttentionMasksType): a dict of BlockMasks.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        h = self.tok_embeddings(tokens)

        for layer in self.layers.values():
            h = layer(h, self.rope_cache, attention_masks)
        h = self.norm(h)
        output = self.output(h)
        return output
