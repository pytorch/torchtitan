# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from dataclasses import dataclass, field

import spmd_types as spmd
import torch
import torch_remat as remat

from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.common.attention import (
    AttentionMetadataMap,
    FlexAttentionMetadata,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.moe import MoE
from .state_dict_adapter import Qwen3StateDictAdapter


class Qwen3TransformerBlock(TransformerBlock):
    """
    Qwen3 TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        dim (int): Model dimension.
        n_layers (int): Total number of layers.
        config (Qwen3TransformerBlock.Config): Block configuration.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()

        self.attention = config.attention.build()

        self.feed_forward = config.feed_forward.build()

        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_metadata: FlexAttentionMetadata | VarlenAttentionMetadata | None,
        positions: torch.Tensor | None = None,
        *,
        residual: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(ffn_out, residual)`` unsummed; the next block's ``attention_norm`` adds them.

        With the ``add_rmsnorm`` region on, that add runs inside the norm kernel.
        The first block gets ``residual=None``: its ``x`` is the residual stream.

        Example (the stream entering this block is h = residual + x):
            ffn_out, residual = block(x, ..., residual=residual)
            # residual: h + attn_out; the stream entering the next block: residual + ffn_out
        """
        if residual is None:
            residual = x
            attention_input = self.attention_norm(x)
        else:
            attention_input, residual = self.attention_norm(x, residual=residual)
        attn_out = self.attention(attention_input, attention_metadata, positions)
        # The residual add reads the attention output with bare ops.
        remat.recompute_needs_tensor(attn_out)
        x = residual + attn_out

        if isinstance(self.feed_forward, MoE):
            ffn_out = self.feed_forward(
                self.ffn_norm(x),
                padding_mask_T=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )
        else:
            ffn_out = self.feed_forward(self.ffn_norm(x))
        return ffn_out, x


class Qwen3Model(Decoder):
    state_dict_adapter_cls = Qwen3StateDictAdapter

    @classmethod
    def _register_optimizer_hooks(
        cls, optimizers, model_parts, parallelism_context
    ) -> None:
        from torchtitan.models.common.aux_loss import register_aux_loss_zero_hook
        from torchtitan.models.common.moe import register_moe_load_balancing_hook

        register_moe_load_balancing_hook(optimizers, model_parts, parallelism_context)
        register_aux_loss_zero_hook(optimizers, model_parts, parallelism_context)

    """
    Qwen3Model Module

    Args:
        config (Qwen3Model.Config): Model configuration.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 1024
        vocab_size: int = 151936
        local_compile: LocalCompileConfig = field(
            default_factory=lambda: LocalCompileConfig(
                regions=[
                    "loss",
                    "fused_binary_activation",
                    "rope",
                    "fp32_to_bf16_split",
                    "add_rmsnorm",
                ]
            )
        )

    def __init__(self, config: Config):
        super().__init__(config)

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_metadata: AttentionMetadataMap | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
    ):
        """``Decoder.forward``, but each block hands the next its output and the
        residual stream unsummed (see ``Qwen3TransformerBlock.forward``).

        Example, 2 layers:
            h = tok_embeddings(tokens)
            out, residual = layers["0"](h)                       # residual = h + attn_out
            out, residual = layers["1"](out, residual=residual)  # attention_norm adds them
            lm_head(norm(residual + out))
        """
        # Arguments in Decoder.forward's order: AutoParallel binds (tokens, positions)
        # positionally.
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        with spmd.no_typecheck():
            aux_loss_denominator = (
                None if aux_loss_denominators is None else aux_loss_denominators[0]
            )
        residual = None
        for layer in self.layers.values():
            layer_attention_metadata = (
                None
                if attention_metadata is None
                else attention_metadata.get(type(layer.attention.inner_attention))
            )
            h, residual = layer(
                h,
                layer_attention_metadata,
                positions,
                residual=residual,
                padding_mask=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )
        # The next pipeline stage takes one tensor, so every stage ends with the sum.
        # A stage without layers passes its input through (residual stays None).
        # TODO: fuse this add into the root norm, self.norm(h, residual=residual): one
        # kernel fewer per forward. Under sequence parallelism the root norm gathers its
        # output through out_dst_shardings, which Module._redistribute_outputs skips
        # for tuple outputs, so that gather must move out of the norm first.
        if residual is not None:
            h = residual + h
        h = self.norm(h) if self.norm is not None else h

        # _skip_lm_head: see Decoder.forward.
        if self._skip_lm_head:
            return h
        return self.lm_head(h) if self.lm_head is not None else h
