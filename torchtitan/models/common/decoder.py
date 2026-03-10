# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch
from torch.nn.attention.flex_attention import and_masks

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    create_attention_mask,
    create_varlen_metadata_for_document,
    get_causal_mask_mod,
    get_document_mask_mod,
)
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe.moe import MoE
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import Module
from torchtitan.protocols.state_initializer import StateInitializer

__all__ = ["Decoder", "DecoderStateInitializer", "TransformerBlock"]


class DecoderStateInitializer(StateInitializer):
    @dataclass(kw_only=True, slots=True)
    class Config(StateInitializer.Config):
        pass

    def init_states(self, module, *, buffer_device=None) -> None:
        buffer_device = buffer_device or module.freqs_cis.device

        if module.rope is not None:
            module.rope.init_states(buffer_device=buffer_device)
            module.freqs_cis = module.rope.cache
        else:
            # PP case: rope module was pruned, rebuild to get freqs_cis
            rope = module.config.rope.build()
            rope.init_states(buffer_device=buffer_device)
            module.freqs_cis = rope.cache

        if module.tok_embeddings is not None:
            module.tok_embeddings.init_states()

        for layer in module.layers.values():
            # FSDP mess up the class, so we cannot use isinstance to check
            # anymore. But all the methods and attributes still preserve.
            if hasattr(layer, "init_states"):
                layer.init_states(buffer_device=buffer_device)

        if module.norm is not None:
            module.norm.init_states()

        if module.output is not None:
            module.output.init_states()


# TODO: we can unify the TransformerBlock impl across all models when
# there is no special logic for each model, including
# init_states, ffn vs. moe naming and creation, rope vs. nope, etc.
class TransformerBlock(Module):
    """Base class for all language model transformer blocks.

    All language model TransformerBlocks share:
    - Attention module (from ``attention.build(dim=dim)``)
    - FFN or MoE (from ``feed_forward.build()`` / ``moe.build()``)
    - Two RMSNorms (``attention_norm``, ``ffn_norm``)
    - ``weight_init_std`` computed from ``layer_id``
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
        layer: TransformerBlock.Config  # required, no default
        state_initializer: StateInitializer.Config = field(
            default_factory=DecoderStateInitializer.Config
        )

    def __init__(self, config: Config):
        super().__init__(config)
        self.config = config

        self.tok_embeddings = config.tok_embeddings.build(
            num_embeddings=config.vocab_size, embedding_dim=config.dim
        )

        self.rope = config.rope.build()
        self.register_buffer("freqs_cis", self.rope.cache, persistent=False)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(config.n_layers):
            self.layers[str(layer_id)] = config.layer.build(
                layer_id=layer_id, dim=config.dim, n_layers=config.n_layers
            )

        self.norm = config.norm.build(normalized_shape=config.dim)

        # Set init_std on the output projection before building
        final_out_std = config.dim**-0.5
        output_cfg = config.output.replace_state_init_field(init_std=final_out_std)
        self.output = output_cfg.build(
            in_features=config.dim, out_features=config.vocab_size
        )

    def init_states(self, *, buffer_device=None):
        self._state_initializer.init_states(self, buffer_device=buffer_device)

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

        match self.attn_config.attn_mask_type:
            case "causal":
                B = 1
            case "block_causal":
                B = input_batch.shape[0]
                assert tokenizer.eos_id is not None
                mask_mods.append(get_document_mask_mod(input_batch, tokenizer.eos_id))
            case _:
                raise ValueError(
                    f"Unknown attention mask type: {self.attn_config.attn_mask_type}"
                )

        return create_attention_mask(
            and_masks(*mask_mods), B, None, input_batch.shape[1], input_batch.shape[1]
        )

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        match self.attn_config.attn_backend:
            case "flex":
                return self._get_flex_attention_masks(
                    input_batch, tokenizer, extra_inputs
                )
            case "varlen":
                if self.attn_config.attn_mask_type != "block_causal":
                    raise ValueError(
                        f"varlen attention is only supported with block_causal "
                        f"attention mask type, got {self.attn_config.attn_mask_type}"
                    )
                assert tokenizer.eos_id is not None
                return create_varlen_metadata_for_document(
                    input_batch, tokenizer.eos_id
                )
            case _:
                raise TypeError("Only varlen and flex attn masks are supported")

    @property
    def attn_config(self):
        """Convenience accessor for the attention config from layer."""
        return self.config.layer.attention
