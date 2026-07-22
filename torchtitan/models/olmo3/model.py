# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.models.common.attention import (
    AttentionMasksType,
    create_varlen_metadata_for_document,
    FlexAttention,
    get_causal_mask_mod,
    get_efficient_causal_mask_mod_for_packed_document,
    get_sliding_window_mask_mod,
    GQAttention,
    VarlenAttention,
)
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.utils import get_dense_model_nparams_and_flops


class Olmo3Attention(GQAttention):
    """GQA attention with Olmo2/Olmo3-style QK-norm.

    Unlike Qwen3's per-head QK-norm (RMSNorm over ``head_dim``, applied after
    splitting into per-head tensors), Olmo2/Olmo3 normalize the *full*
    concatenated query/key projection (width ``n_heads*head_dim``) before the
    head split. This is mathematically a reshape-then-RMSNorm-then-reshape
    around ``GQAttention``'s existing per-head split, so only ``forward``
    needs to change.

    Both ``q_norm`` and ``k_norm`` are built from the same ``qk_norm`` config
    (see ``GQAttention.__init__``), so this only supports ``n_heads ==
    n_kv_heads`` (true for Olmo3-7B's MHA; GQA would need distinct widths).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GQAttention.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        if self.q_norm is not None:
            assert self.n_heads == self.n_kv_heads, (
                "Olmo3Attention shares one qk_norm config for q_norm and "
                "k_norm (both normalized over n_heads*head_dim); GQA "
                "(n_heads != n_kv_heads) is not supported."
            )

    def forward(
        self,
        x_BLD: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, _ = x_BLD.shape
        xq_BLNH, xk_BLNH, xv_BLNH = self.qkv_linear(x_BLD)

        if self.q_norm is not None or self.k_norm is not None:
            assert self.q_norm is not None and self.k_norm is not None
            xq_BLNH = self.q_norm(xq_BLNH.reshape(B, L, -1)).view(
                B, L, self.n_heads, self.head_dim
            )
            xk_BLNH = self.k_norm(xk_BLNH.reshape(B, L, -1)).view(
                B, L, self.n_kv_heads, self.head_dim
            )

        xq_BLNH, xk_BLNH = self.rope(xq_BLNH, xk_BLNH, positions)

        out_BLNH = self.inner_attention(
            xq_BLNH,
            xk_BLNH,
            xv_BLNH,
            attention_masks=attention_masks,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
        ).contiguous()
        out_BLD = out_BLNH.view(B, L, -1)
        return self.wo(out_BLD)


class Olmo3TransformerBlock(TransformerBlock):
    """
    Olmo3 TransformerBlock Module.

    Unlike Llama3's pre-norm placement, Olmo3 (like Olmo2) applies
    ``attention_norm``/``ffn_norm`` to the *output* of the attention/feed-forward
    sublayers rather than their input: ``x + norm(sublayer(x))``.

    Args:
        layer_id (int): Identifier for the layer.
        dim (int): Model dimension.
        n_layers (int): Total number of layers.
        config (Olmo3TransformerBlock.Config): Block configuration.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()
        assert isinstance(config.attention, GQAttention.Config)
        self.attn_mask_key = (
            "sliding_attention"
            if config.attention.sliding_window_size is not None
            else "full_attention"
        )
        self.attention = config.attention.build()
        assert config.feed_forward is not None
        self.feed_forward = config.feed_forward.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        if isinstance(attention_masks, dict):  # flex
            attention_masks = attention_masks[self.attn_mask_key]

        h = x + self.attention_norm(self.attention(x, attention_masks, positions))
        out = h + self.ffn_norm(self.feed_forward(h))
        return out


class Olmo3Model(Decoder):
    """
    Olmo3Model Module.

    Hybrid sliding-window / full-attention decoder: most layers use a
    causal sliding-window attention, with a full-attention layer every
    ``full_attention_interval`` layers.

    Args:
        config (Olmo3Model.Config): Model configuration.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 4096
        vocab_size: int = 100278

        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            if parallelism.tensor_parallel_degree > 1:
                raise NotImplementedError(
                    "Olmo3 only supports FSDP. QK-norm is applied over the "
                    "full projection width (not per-head, see "
                    "Olmo3Attention), which is incompatible with "
                    "head-sharded tensor_parallel_degree > 1."
                )

            if parallelism.context_parallel_degree > 1 and isinstance(
                self.layers[0].attention.inner_attention, VarlenAttention.Config
            ):
                raise NotImplementedError(
                    "Context Parallel only supports SDPA and FlexAttention. "
                    "Varlen attention is not supported with CP."
                )

            from torchtitan.models.olmo3.sharding import set_olmo3_sharding_config

            set_olmo3_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
            )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            return get_dense_model_nparams_and_flops(
                model,
                n_layers=len(self.layers),
                n_heads=self.layers[0].attention.n_heads,
                head_dims=2 * (self.dim // self.layers[0].attention.n_heads),
                seq_len=seq_len,
                enable_weight_tying=self.enable_weight_tying,
            )

    def get_attention_masks(
        self,
        positions: torch.Tensor,
    ) -> AttentionMasksType:
        attn_cfg = self.config.layers[0].attention
        assert isinstance(attn_cfg, GQAttention.Config)
        inner_attn = attn_cfg.inner_attention

        if isinstance(inner_attn, VarlenAttention.Config):
            # Per-layer sliding window is baked into each layer's own
            # VarlenAttention.window_size at config-build time.
            return create_varlen_metadata_for_document(positions)
        elif isinstance(inner_attn, FlexAttention.Config):
            base_mask_mods = [
                get_causal_mask_mod(),
                get_efficient_causal_mask_mod_for_packed_document(positions),
            ]
            masks: dict[str, BlockMask] = {
                "full_attention": self._create_flex_attention_mask(
                    positions, attn_cfg, base_mask_mods
                )
            }

            window = None
            for layer in self.config.layers:
                if (
                    isinstance(layer.attention, GQAttention.Config)
                    and layer.attention.sliding_window_size is not None
                ):
                    window = layer.attention.sliding_window_size
                    break
            if window is not None:
                masks["sliding_attention"] = self._create_flex_attention_mask(
                    positions,
                    attn_cfg,
                    [*base_mask_mods, get_sliding_window_mask_mod(window)],
                )

            return masks
        else:
            raise TypeError(
                f"Olmo3 supports FlexAttention and VarlenAttention inner attention, "
                f"got {type(inner_attn).__name__}"
            )
