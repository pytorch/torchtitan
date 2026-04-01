# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
from torch.nn.attention.flex_attention import and_masks

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    create_attention_mask,
    create_varlen_metadata_for_document,
    FlexAttention,
    get_causal_mask_mod,
    get_document_mask_mod,
    VarlenAttention,
)
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe.moe import MoE
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import Module, ModuleDict

__all__ = ["Decoder", "TransformerBlock"]


# TODO: we can unify the TransformerBlock impl across all models when
# there is no special logic for each model, including
# ffn vs. moe naming and creation, rope vs. nope, etc.
class TransformerBlock(Module):
    """Base class for all language model transformer blocks.

    All language model TransformerBlocks share:
    - Attention module (from ``attention.build()``)
    - FFN or MoE (from ``feed_forward.build()`` / ``moe.build()``)
    - Two RMSNorms (``attention_norm``, ``ffn_norm``)
    - Forward: ``x + attn(norm(x), ...); x + ffn(norm(x))``

    Children implement ``__init__`` and ``forward``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        attention: BaseAttention.Config  # required, no default
        feed_forward: FeedForward.Config | None = None
        moe: MoE.Config | None = None
        attention_norm: RMSNorm.Config
        ffn_norm: RMSNorm.Config


class Decoder(BaseModel):
    """Base class for autoregressive decoder-only language models.

    Provides shared ``__init__``, ``forward``, ``init_states``, and
    ``get_attention_masks`` (flex/varlen dispatch) used by most models.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        dim: int
        n_layers: int
        vocab_size: int
        output: Linear.Config
        tok_embeddings: Embedding.Config
        norm: RMSNorm.Config
        # TODO: Right now RoPE config is not in each TransformerBlock / Attention,
        # so that rope cache, a.k.a. freqs_cis, is shared by all layers. However,
        # it causes redundantly passing backend (complex / cos_sin) to both RoPE
        # and Attention. Also RoPE itself as a standalone module requires PP special
        # handling, see below.
        rope: RoPE.Config
        layer: TransformerBlock.Config
        layers: list | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.tok_embeddings = config.tok_embeddings.build()
        self.rope = config.rope.build()
        self.register_buffer("freqs_cis", self.rope.cache, persistent=False)

        assert config.layers is not None, (
            "config.layers must be populated by expand_layer_configs() "
            "in the model registry before build()."
        )
        self.layers = ModuleDict()
        for i, layer_config in enumerate(config.layers):
            self.layers[str(i)] = layer_config.build()

        self.norm = config.norm.build()
        self.output = config.output.build()

    def init_states(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        # Compute buffer_device before recursion so children (RoPE) get
        # the correct device when buffer_device is not explicitly provided.
        if buffer_device is None:
            buffer_device = self.freqs_cis.device
        super().init_states(buffer_device=buffer_device)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        assert buffer_device is None or buffer_device.type != "meta", (
            f"buffer_device must not be meta, got {buffer_device}. "
            f"Buffers should be initialized on a real device after to_empty()."
        )
        if self.rope is not None:
            # RoPE's _init_self_buffers was already called by auto-recursion
            self.freqs_cis = self.rope.cache
        else:
            # PP case: rope module was pruned, rebuild to get freqs_cis
            rope = self.config.rope.build()
            rope._init_self_buffers(buffer_device=buffer_device)
            self.freqs_cis = rope.cache

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ):
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis, attention_masks, positions)

        h = self.norm(h) if self.norm is not None else h
        output = self.output(h) if self.output is not None else h
        return output

    def _get_flex_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        mask_mods = [get_causal_mask_mod()]

        match self.attn_config.mask_type:
            case "causal":
                B = 1
            case "block_causal":
                B = input_batch.shape[0]
                assert tokenizer.eos_id is not None
                mask_mods.append(get_document_mask_mod(input_batch, tokenizer.eos_id))
            case _:
                raise ValueError(
                    f"Unknown attention mask type: {self.attn_config.mask_type}"
                )

        assert isinstance(self.attn_config.inner_attention, FlexAttention.Config)
        return create_attention_mask(
            and_masks(*mask_mods),
            B,
            None,
            input_batch.shape[1],
            input_batch.shape[1],
            BLOCK_SIZE=self.attn_config.inner_attention.block_size,
        )

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        inner_attn = self.attn_config.inner_attention
        if isinstance(inner_attn, FlexAttention.Config):
            return self._get_flex_attention_masks(input_batch, tokenizer, extra_inputs)
        elif isinstance(inner_attn, VarlenAttention.Config):
            if self.attn_config.mask_type != "block_causal":
                raise ValueError(
                    f"varlen attention is only supported with block_causal "
                    f"attention mask type, got {self.attn_config.mask_type}"
                )
            assert tokenizer.eos_id is not None
            return create_varlen_metadata_for_document(input_batch, tokenizer.eos_id)
        else:
            raise TypeError(
                f"Only VarlenAttention and FlexAttention support attention masks, "
                f"got {type(inner_attn).__name__}"
            )

    @property
    def attn_config(self):
        """Convenience accessor for the attention config from layer."""
        return self.config.layer.attention
