# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torch_remat as remat

from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.base.model import BaseModel
from torchtitan.models.common.attention import (
    FlexAttentionMetadata,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.decoder import TransformerBlock
from torchtitan.models.common.moe import MoE
from torchtitan.models.deepseek_v3.mtp import MTPDecoder

from .attention import DeepSeekMLAAttention
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

    @dataclass(kw_only=True, slots=True)
    class Config(MTPDecoder.Config):
        dim: int = 2048
        vocab_size: int = 102400
        # pyrefly: ignore [bad-override]
        layers: list[DeepSeekV3TransformerBlock.Config]
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

    def pipeline(self, **kwargs: Any) -> tuple[Any, list[BaseModel], bool, bool]:
        """Use the model-owned MTP partition when MTP is enabled."""
        if not self.num_mtp_layers:
            return super().pipeline(**kwargs)
        from .pipeline import pipeline_dsv3

        return pipeline_dsv3(self, **kwargs)

    @classmethod
    def _synchronize_shared_parameters(
        cls, model_parts: list[BaseModel], parallelism_context
    ) -> None:
        """Synchronize MTP embedding shards on their PP owners."""
        del parallelism_context
        from .pipeline import _synchronize_mtp_embeddings

        _synchronize_mtp_embeddings(cast(list[DeepSeekV3Model], model_parts))

    @classmethod
    def _register_optimizer_hooks(
        cls, optimizers, model_parts, parallelism_context
    ) -> None:
        from torchtitan.models.common.aux_loss import register_aux_loss_zero_hook
        from torchtitan.models.common.moe import register_moe_load_balancing_hook

        register_moe_load_balancing_hook(optimizers, model_parts, parallelism_context)
        register_aux_loss_zero_hook(optimizers, model_parts, parallelism_context)
