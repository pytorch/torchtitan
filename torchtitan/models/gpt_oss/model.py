# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import math
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn.attention.flex_attention import and_masks, BlockMask

from torchtitan.components.tokenizer import BaseTokenizer
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
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rope import apply_rotary_emb_cos_sin
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.state_initializer import StateInitializer
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability


class GptOssAttentionStateInitializer(StateInitializer):
    @dataclass(kw_only=True, slots=True)
    class Config(StateInitializer.Config):
        init_std: float = 0.02

    def __init__(self, config: Config):
        self.init_std = config.init_std

    def init_states(self, module, *, buffer_device=None) -> None:
        assert isinstance(module, Attention)
        assert isinstance(module.sinks, nn.Parameter)
        nn.init.trunc_normal_(module.sinks, mean=0.0, std=self.init_std)
        for linear in (module.wq, module.wk, module.wv, module.wo):
            linear.init_states()


class Attention(BaseAttention):
    """
    Multi-head attention (MLA) module with sink attention.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        wq: Linear.Config
        wk: Linear.Config
        wv: Linear.Config
        wo: Linear.Config
        n_heads: int = 64
        n_kv_heads: int = 8
        head_dim: int = 64
        attn_backend: str = "flex"  # NOTE: gpt-oss only supports FlexAttention
        attn_mask_type: str = "causal"
        sliding_window_size: int = 128
        state_initializer: StateInitializer.Config = field(
            default_factory=GptOssAttentionStateInitializer.Config
        )

    def __init__(self, config: Config, *, dim: int):
        super().__init__(config)
        self.head_dim = config.head_dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.enable_gqa = self.n_heads > self.n_kv_heads

        self.n_rep = self.n_heads // self.n_kv_heads

        # Standard attention softmax scale (1/sqrt(head_dim))
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)

        # Read init_std from our state_initializer config for all projections
        assert isinstance(
            config.state_initializer, GptOssAttentionStateInitializer.Config
        )
        init_std = config.state_initializer.init_std
        self.wq = config.wq.replace_state_init_field(init_std=init_std).build(
            in_features=dim, out_features=config.n_heads * config.head_dim
        )
        self.wk = config.wk.replace_state_init_field(init_std=init_std).build(
            in_features=dim, out_features=config.n_kv_heads * config.head_dim
        )
        self.wv = config.wv.replace_state_init_field(init_std=init_std).build(
            in_features=dim, out_features=config.n_kv_heads * config.head_dim
        )
        self.wo = config.wo.replace_state_init_field(init_std=init_std).build(
            in_features=config.n_heads * config.head_dim, out_features=dim
        )
        self.sinks = nn.Parameter(torch.empty(config.n_heads))
        assert config.attn_backend == "flex", "gpt-oss only supports FlexAttention"
        self.inner_attention = FlexAttentionWrapper()

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


class GptOssTransformerBlockStateInitializer(StateInitializer):
    @dataclass(kw_only=True, slots=True)
    class Config(StateInitializer.Config):
        pass

    def init_states(self, module, *, buffer_device=None) -> None:
        for norm in (module.attention_norm, module.ffn_norm):
            norm.init_states()
        module.attention.init_states()
        module.moe.init_states(buffer_device=buffer_device)


class GptOssTransformerBlock(TransformerBlock):
    """
    GptOss Transformer block with sliding window attention support.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        state_initializer: StateInitializer.Config = field(
            default_factory=GptOssTransformerBlockStateInitializer.Config
        )

    def __init__(self, config: Config, *, layer_id: int, dim: int, n_layers: int):
        super().__init__(config)

        weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5

        self.use_sliding_attention = layer_id % 2 == 0

        # Replace init_std on output projections before building
        attn_cfg = config.attention.replace_state_init_field(init_std=weight_init_std)
        self.attention = attn_cfg.build(dim=dim)
        self.attention_norm = config.attention_norm.build(normalized_shape=dim)
        self.ffn_norm = config.ffn_norm.build(normalized_shape=dim)

        assert config.moe is not None
        moe_cfg = config.moe.replace_state_init_field(init_std=weight_init_std)
        self.moe = moe_cfg.build(dim=dim)
        self.moe_enabled = True  # for composability with load balancing

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
