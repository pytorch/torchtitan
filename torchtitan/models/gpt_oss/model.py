# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.attention.flex_attention import and_masks, BlockMask

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.common import trunc_normal_
from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    create_attention_mask,
    FlexAttentionWrapper,
    get_causal_mask_mod,
    get_document_mask_mod,
    get_sliding_window_mask_mod,
)
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.rope import apply_rotary_emb_cos_sin
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability


class Attention(BaseAttention):
    """
    Multi-head attention (MLA) module with sink attention.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int = 64
        n_kv_heads: int = 8
        head_dim: int = 64
        attn_backend: str = "flex"  # NOTE: gpt-oss only supports FlexAttention
        attn_mask_type: str = "causal"
        sliding_window_size: int = 128

    def __init__(self, config: Config, *, dim: int):
        super().__init__()
        self.head_dim = config.head_dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.enable_gqa = self.n_heads > self.n_kv_heads

        self.n_rep = self.n_heads // self.n_kv_heads

        # Standard attention softmax scale (1/sqrt(head_dim))
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)

        self.wq = nn.Linear(
            dim,
            config.n_heads * config.head_dim,
            bias=True,
        )
        self.wk = nn.Linear(
            dim,
            config.n_kv_heads * config.head_dim,
            bias=True,
        )
        self.wv = nn.Linear(
            dim,
            config.n_kv_heads * config.head_dim,
            bias=True,
        )
        self.wo = nn.Linear(
            config.n_heads * config.head_dim,
            dim,
            bias=True,
        )
        self.sinks = nn.Parameter(torch.empty(config.n_heads))
        assert config.attn_backend == "flex", "gpt-oss only supports FlexAttention"
        self.inner_attention = FlexAttentionWrapper()

    def init_weights(self, **kwargs):
        init_std = kwargs.get("init_std")
        assert init_std is not None
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
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies for rope embedding.
            attention_masks: Attention mask (BlockMask).
            positions: Optional position indices (unused, for API compatibility).

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()
        hidden_shape = (bsz, seqlen, -1, self.head_dim)

        q = self.wq(x).view(hidden_shape)
        k = self.wk(x).view(hidden_shape)
        v = self.wv(x).view(hidden_shape)

        q, k = apply_rotary_emb_cos_sin(q, k, freqs_cis, positions)

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


class GptOssTransformerBlock(TransformerBlock):
    """
    GptOss Transformer block with sliding window attention support.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        pass

    def __init__(self, config: Config, *, layer_id: int, dim: int, n_layers: int):
        super().__init__()
        self.use_sliding_attention = layer_id % 2 == 0
        self.attention = config.attention.build(dim=dim)
        self.attention_norm = nn.RMSNorm(dim, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=config.norm_eps)

        assert config.moe is not None
        self.moe = config.moe.build(dim=dim)
        self.moe_enabled = True  # for composability with load balancing

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
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            attention_masks (AttentionMasksType): a dict of BlockMasks.
            positions: Optional position indices.

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

        x = x + self.attention(self.attention_norm(x), freqs_cis, layer_mask, positions)
        x = x + self.moe(self.ffn_norm(x))
        return x

    def init_weights(self, **kwargs):
        buffer_device = kwargs.get("buffer_device")
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(init_std=self.weight_init_std)
        self.moe.init_weights(
            init_std=self.weight_init_std, buffer_device=buffer_device
        )


class GptOssModel(Decoder):
    """
    GPT-OSS Transformer model with attention and feed-forward layers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 2880
        n_layers: int = 24
        vocab_size: int = 201088

        # Sub-component configs
        layer: TransformerBlock.Config

        def update_from_config(
            self,
            *,
            trainer_config,
            **kwargs,
        ) -> None:
            training = trainer_config.training
            parallelism = trainer_config.parallelism
            seq_len = training.seq_len
            if seq_len > self.rope.max_seq_len:
                logger.warning(
                    f"Sequence length {seq_len} exceeds original maximum {self.rope.max_seq_len}."
                )

            # Sync rope max_seq_len
            self.rope = dataclasses.replace(self.rope, max_seq_len=seq_len)

            assert self.layer.moe is not None
            if self.layer.moe.use_grouped_mm and not has_cuda_capability(9, 0):
                logger.warning(
                    "Failed to use grouped mm, which is only supported on SM90 or later",
                )
                self.layer.moe.use_grouped_mm = False

            if parallelism.context_parallel_degree > 1:
                raise NotImplementedError(
                    "CP support for gpt-oss model is still in progress."
                )

        # pyrefly: ignore [bad-override]
        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, float]:
            assert isinstance(self.layer.attention, Attention.Config)
            return get_moe_model_nparams_and_flops(
                self,
                model,
                self.layer.attention.n_heads,
                2 * self.layer.attention.head_dim,
                seq_len,
            )

    def __init__(self, config: Config):
        super().__init__(config)
        # GptOss uses dtype=torch.get_default_dtype() for output linear
        self.output = nn.Linear(
            config.dim,
            config.vocab_size,
            dtype=torch.get_default_dtype(),
            bias=False,
        )

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:

        basic_mask_mods = []
        assert isinstance(self.config.layer.attention, Attention.Config)
        sliding_window_mask_mods = [
            get_sliding_window_mask_mod(self.config.layer.attention.sliding_window_size)
        ]
        match self.config.layer.attention.attn_mask_type:
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
                    f"Unknown attention mask type: {self.config.layer.attention.attn_mask_type}"
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
