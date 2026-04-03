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
from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    create_attention_mask,
    FlexAttention,
    get_causal_mask_mod,
    get_document_mask_mod,
    get_sliding_window_mask_mod,
    LocalMapInnerAttention,
)
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.linear import Linear
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
        dim: int
        wq: Linear.Config  # query projection
        wkv: Linear.Config  # shared config for key + value (build() copies)
        wo: Linear.Config  # output projection
        inner_attention: LocalMapInnerAttention.Config = dataclasses.field(
            default_factory=FlexAttention.Config
        )
        mask_type: str = "causal"
        sliding_window_size: int = 128

    def __init__(self, config: Config):
        super().__init__()
        self.head_dim = config.head_dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.enable_gqa = self.n_heads > self.n_kv_heads

        self.n_rep = self.n_heads // self.n_kv_heads

        # Standard attention softmax scale (1/sqrt(head_dim))
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)

        self.wq = config.wq.build()
        self.wk = config.wkv.build()  # build() copies — independent module
        self.wv = config.wkv.build()  # build() copies — independent module
        self.wo = config.wo.build()
        self.sinks = nn.Parameter(torch.empty(config.n_heads))
        assert isinstance(
            config.inner_attention, FlexAttention.Config
        ), "gpt-oss only supports FlexAttention"
        self.inner_attention = config.inner_attention.build()

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

        assert isinstance(attention_masks, BlockMask), attention_masks
        # FlexAttention handles transpose internally; returns (bs, seq, heads, dim)
        # and lse as (bs, seq, heads)
        output, lse = self.inner_attention(
            q,
            k,
            v,
            attention_masks=attention_masks,
            scale=self.softmax_scale,
            return_lse=True,
            enable_gqa=self.enable_gqa,
        )

        # Apply attention sink rescaling: rescale by sigma(lse - w[h])
        # output: (bs, seq, heads, dim), lse: (bs, seq, heads)
        sink_scale = torch.sigmoid(lse - self.sinks.view(1, 1, -1)).unsqueeze(-1)
        output = output * sink_scale.to(output.dtype)

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
        use_sliding_attention: bool = False

    def __init__(self, config: Config):
        super().__init__()
        self.use_sliding_attention = config.use_sliding_attention
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

        assert config.moe is not None
        self.moe = config.moe.build()
        self.moe_enabled = True  # for composability with load balancing

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


class GptOssModel(Decoder):
    """
    GPT-OSS Transformer model with attention and feed-forward layers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 2880
        vocab_size: int = 201088

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

            for layer_cfg in self.layers:
                if layer_cfg.moe is not None:
                    if (
                        layer_cfg.moe.experts.use_grouped_mm
                        and not has_cuda_capability(9, 0)
                    ):
                        logger.warning(
                            "Failed to use grouped mm, which is only supported on SM90 or later",
                        )
                        layer_cfg.moe.experts.use_grouped_mm = False

            tp = parallelism.tensor_parallel_degree
            if tp > 1:
                n_heads = self.layers[0].attention.n_heads
                # pyrefly: ignore [missing-attribute]
                n_kv_heads = self.layers[0].attention.n_kv_heads
                if n_heads % tp != 0:
                    raise ValueError(
                        f"tensor_parallel_degree ({tp}) must divide n_heads ({n_heads})."
                    )
                if n_kv_heads % tp != 0:
                    raise ValueError(
                        f"tensor_parallel_degree ({tp}) must divide n_kv_heads ({n_kv_heads})."
                    )

        # pyrefly: ignore [bad-override]
        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, float]:
            assert isinstance(self.layers[0].attention, Attention.Config)
            return get_moe_model_nparams_and_flops(
                self,
                model,
                self.layers[0].attention.n_heads,
                2 * self.layers[0].attention.head_dim,
                seq_len,
            )

    def __init__(self, config: Config):
        super().__init__(config)

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        basic_mask_mods = []
        attn_cfg = self.config.layers[0].attention
        assert isinstance(attn_cfg, Attention.Config)
        sliding_window_mask_mods = [
            get_sliding_window_mask_mod(attn_cfg.sliding_window_size)
        ]
        match attn_cfg.mask_type:
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
                raise ValueError(f"Unknown attention mask type: {attn_cfg.mask_type}")

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
