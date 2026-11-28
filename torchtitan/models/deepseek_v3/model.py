# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import cast

import spmd_types as spmd
import torch
import torch_remat as remat

from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.base.module import ModuleList
from torchtitan.models.common.attention import (
    AttentionMetadataMap,
    FlexAttentionMetadata,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.decoder import TransformerBlock
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.mtp import MTPDecoder

from .attention import DeepSeekMLAAttention
from .mtp import DeepSeekV3MTPBlock
from .state_dict_adapter import DeepSeekV3StateDictAdapter


class DeepSeekV3TransformerBlock(TransformerBlock):
    """
    DeepSeek V3 Transformer block with attention and feed-forward layers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        attention: DeepSeekMLAAttention.Config  # pyrefly: ignore [bad-override]

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

        self.feed_forward = config.feed_forward.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_metadata: FlexAttentionMetadata | VarlenAttentionMetadata | None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ):
        attn_out = self.attention(self.attention_norm(x), attention_metadata, positions)
        # The residual add reads the attention output with bare ops.
        remat.recompute_needs_tensor(attn_out)
        x = x + attn_out
        if isinstance(self.feed_forward, MoE):
            ffn_out = self.feed_forward(
                self.ffn_norm(x),
                padding_mask_T=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )
        else:
            ffn_out = self.feed_forward(self.ffn_norm(x))
        # Trailing add, always saved: it saves nothing for backward, so replay skips
        # it and its inputs need no persisting, matching checkpoint early stop.
        return remat.region(
            torch.add, self.remat_region_name("ffn_residual"), recompute=False
        )(x, ffn_out)


class DeepSeekV3Model(MTPDecoder):
    state_dict_adapter_cls = DeepSeekV3StateDictAdapter

    """
    DeepSeek-V3 Transformer model with attention and feed-forward layers.
    """

    mtp_layers: ModuleList[DeepSeekV3MTPBlock]  # pyrefly: ignore [bad-override]

    @dataclass(kw_only=True, slots=True)
    class Config(MTPDecoder.Config):
        dim: int = 2048
        vocab_size: int = 102400
        # pyrefly: ignore [bad-override]
        layers: list[DeepSeekV3TransformerBlock.Config]
        # pyrefly: ignore [bad-override]
        mtp_layers: list[DeepSeekV3MTPBlock.Config] = field(default_factory=list)
        local_compile: LocalCompileConfig = field(
            default_factory=lambda: LocalCompileConfig(
                regions=[
                    "loss",
                    "fused_binary_activation",
                    "fp32_to_bf16_split",
                    "moe_aux_loss",
                    "rope",
                    "mla_qk",
                ]
            )
        )

    def __init__(self, config: Config):
        super().__init__(config)

    def forward(
        self,
        tokens: torch.Tensor | tuple[torch.Tensor, ...],
        positions: torch.Tensor | None = None,
        attention_metadata: AttentionMetadataMap | None = None,
        mtp_input_valid_masks: tuple[torch.Tensor, ...] | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
    ):
        if not self.mtp_layers:
            return super().forward(
                cast(torch.Tensor, tokens),
                positions,
                attention_metadata,
                padding_mask=padding_mask,
                aux_loss_denominators=aux_loss_denominators,
            )
        if self.tok_embeddings is None:
            raise ValueError("MTP decoder forward requires token embeddings.")
        if not isinstance(tokens, tuple) or len(tokens) != len(self.mtp_layers) + 1:
            raise ValueError(
                "MTP decoder requires one main token tensor and one prepared "
                "token tensor per MTP layer."
            )
        if mtp_input_valid_masks is None or len(mtp_input_valid_masks) != len(
            self.mtp_layers
        ):
            raise ValueError("MTP decoder requires one validity mask per MTP layer.")

        main_tokens, *mtp_input_tokens = tokens

        # Keep this aligned with Decoder.forward(), but preserve the pre-norm
        # hidden state because MTP consumes the last decoder-layer output.
        h = self.tok_embeddings(main_tokens)
        if (
            aux_loss_denominators is not None
            and aux_loss_denominators.numel() != len(self.mtp_layers) + 1
        ):
            raise ValueError("Expected one aux-loss denominator per MTP objective.")
        with spmd.no_typecheck():
            main_aux_loss_denominator = (
                None if aux_loss_denominators is None else aux_loss_denominators[0]
            )
        for layer in self.layers.values():
            layer_attention_metadata = (
                None
                if attention_metadata is None
                else attention_metadata.get(type(layer.attention.inner_attention))
            )
            h = layer(
                h,
                layer_attention_metadata,
                positions,
                padding_mask=padding_mask,
                aux_loss_denominator=main_aux_loss_denominator,
            )

        prev_depth_hidden = h
        h = self._pre_lm_head_norm_and_gather(h)

        mtp_outputs = []
        for depth, (layer, depth_tokens, mtp_input_valid_mask) in enumerate(
            zip(
                self.mtp_layers,
                mtp_input_tokens,
                mtp_input_valid_masks,
                strict=True,
            ),
            1,
        ):
            mtp_input_embed = self.tok_embeddings(depth_tokens)
            layer_attention_metadata = (
                None
                if attention_metadata is None
                else attention_metadata.get(type(layer.attention.inner_attention))
            )
            with spmd.no_typecheck():
                aux_loss_denominator = (
                    None
                    if aux_loss_denominators is None
                    else aux_loss_denominators[depth]
                )
            prev_depth_hidden, mtp_output = layer(
                mtp_input_embed,
                prev_depth_hidden,
                mtp_input_valid_mask,
                layer_attention_metadata,
                positions,
                padding_mask=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )
            mtp_outputs.append(mtp_output)

        outputs = (h, *mtp_outputs)
        if self._skip_lm_head:
            predictions = outputs
        else:
            predictions = tuple(
                self.lm_head(item) if self.lm_head is not None else item
                for item in outputs
            )
        return predictions

    @classmethod
    def _register_optimizer_hooks(
        cls, optimizers, model_parts, parallelism_context
    ) -> None:
        from torchtitan.models.common.aux_loss import register_aux_loss_zero_hook
        from torchtitan.models.common.moe import register_moe_load_balancing_hook

        register_moe_load_balancing_hook(optimizers, model_parts, parallelism_context)
        register_aux_loss_zero_hook(optimizers, model_parts, parallelism_context)
