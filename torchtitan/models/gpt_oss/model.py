# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor
from torch.nn.attention.flex_attention import and_masks, BlockMask

from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    BaseQKVLinear,
    create_attention_mask,
    create_varlen_metadata_for_document,
    FlexAttention,
    get_causal_mask_mod,
    get_document_mask_mod,
    get_sliding_window_mask_mod,
    VarlenAttention,
)
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.nn_modules import Linear
from torchtitan.models.common.rope import RoPE
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.module import Module


class ScaledBiasRowwiseLinear(Linear):
    """
    Rowwise linear whose local bias contribution is scaled by TP degree.
    TODO(pianpwk): this should work in decomposition in spmd_types, or as Partial
    init in DTensor. Today the local SPMD typecheck errors on the TP-axis
    input:V, weight:V, bias:P case; decomposing to input @ weight -> P, then P + P should pass.
    For DTensor, this errors because FSDP does not want to redistribute the incoming gradient
    from Replicate -> storage-time Partial.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.tp_degree = 1

    def parallelize(self, parallel_dims) -> None:
        self.tp_degree = parallel_dims.tp
        super().parallelize(parallel_dims)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = (
            self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        )
        bias = self.bias.to_local() if isinstance(self.bias, DTensor) else self.bias
        bias = bias / self.tp_degree
        return F.linear(input, weight, bias)


def apply_attention_sink_rescale(
    out: torch.Tensor, lse: torch.Tensor, sinks: torch.Tensor
) -> torch.Tensor:
    """Rescale attention output by the learned per-head sink term."""
    sinks = sinks.view(*([1] * (lse.ndim - 1)), -1)
    sink_scale = torch.sigmoid(lse - sinks).unsqueeze(-1)
    return out * sink_scale.to(out.dtype)

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
        qkv_linear: BaseQKVLinear.Config
        wo: Linear.Config  # output projection
        inner_attention: Module.Config = dataclasses.field(
            default_factory=VarlenAttention.Config
        )
        sliding_window_size: int | None = None
        """Per-layer causal sliding-window size"""
        rope: RoPE.Config

    def __init__(self, config: Config):
        super().__init__()
        self.head_dim = config.head_dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.enable_gqa = self.n_heads > self.n_kv_heads

        self.n_rep = self.n_heads // self.n_kv_heads

        # Standard attention softmax scale (1/sqrt(head_dim))
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv_linear = config.qkv_linear.build()
        self.wo = config.wo.build()
        self.sinks = nn.Parameter(torch.empty(config.n_heads))
        self.inner_attention = config.inner_attention.build()
        self.rope = config.rope.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType,
        positions: torch.Tensor | None = None,
    ):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            attention_masks: a ``BlockMask`` (flex) or ``VarlenMetadata`` (varlen).
            positions: Optional position indices (unused, for API compatibility).

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()

        q, k, v = self.qkv_linear(x)

        q, k = self.rope(q, k, positions)

        output = self.inner_attention(
            q,
            k,
            v,
            attention_masks=attention_masks,
            scale=self.softmax_scale,
            enable_gqa=self.enable_gqa,
            out_transform=self._apply_sinks,
        )

        # Reshape and project output
        output = output.reshape(
            bsz, seqlen, -1
        ).contiguous()  # (bsz, seqlen, n_heads * v_head_dim)
        output = self.wo(output)  # (bsz, seqlen, dim)
        return output

    def _apply_sinks(self, out: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
        """out_transform hook: rescale attention output by this layer's sinks."""
        sinks = self.sinks
        if isinstance(sinks, DTensor):
            sinks = sinks.to_local(grad_placements=sinks.placements)
        return apply_attention_sink_rescale(out, lse, sinks)


class GptOssTransformerBlock(TransformerBlock):
    """
    GptOss Transformer block with sliding window attention support.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()
        assert isinstance(config.attention, Attention.Config)
        self.attn_mask_key = (
            "sliding_window_mask"
            if config.attention.sliding_window_size is not None
            else "basic_mask"
        )
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

        assert config.moe is not None
        self.moe = config.moe.build()
        self.moe_enabled = True  # for composability with load balancing

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            attention_masks (AttentionMasksType): with flex, a dict of per-window
                ``BlockMask``s from which this layer picks its mask; with varlen,
                a single ``VarlenMetadata`` shared by all layers (the per-layer
                causal window is baked into each layer's
                ``VarlenAttention.window_size``).
            positions: Optional position indices.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """

        if isinstance(attention_masks, dict):  # flex
            attention_masks = attention_masks[self.attn_mask_key]

        x = x + self.attention(self.attention_norm(x), attention_masks, positions)
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
            config,
            **kwargs,
        ) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            from torchtitan.models.gpt_oss.sharding import set_gpt_oss_sharding_config

            set_gpt_oss_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
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
        positions: torch.Tensor,
    ) -> AttentionMasksType:
        attn_cfg = self.config.layers[0].attention
        assert isinstance(attn_cfg, Attention.Config)
        inner_attn = attn_cfg.inner_attention

        if isinstance(inner_attn, VarlenAttention.Config):
            return create_varlen_metadata_for_document(positions)
        elif isinstance(inner_attn, FlexAttention.Config):
            seq_len = positions.shape[1]
            B = positions.shape[0]
            basic_mask_mods = [
                get_causal_mask_mod(),
                get_document_mask_mod(positions),
            ]

            # Full-attention (causal + document) mask, used by layers without a
            # sliding window.
            masks: dict[str, BlockMask] = {
                "basic_mask": create_attention_mask(
                    and_masks(*basic_mask_mods),
                    B,
                    None,
                    seq_len,
                    seq_len,
                )
            }

            # Sliding-window mask, built only if some layer requests a window.
            window = None
            for layer in self.config.layers:
                if (
                    isinstance(layer.attention, Attention.Config)
                    and layer.attention.sliding_window_size is not None
                ):
                    window = layer.attention.sliding_window_size
                    break
            if window is not None:
                masks["sliding_window_mask"] = create_attention_mask(
                    and_masks(*basic_mask_mods, get_sliding_window_mask_mod(window)),
                    B,
                    None,
                    seq_len,
                    seq_len,
                )

            return masks
        else:
            raise TypeError(
                f"GPT-OSS supports FlexAttention and VarlenAttention inner attention, "
                f"got {type(inner_attn).__name__}"
            )
