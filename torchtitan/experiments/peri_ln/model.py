# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Llama 3 with normalization around each attention and feed-forward branch."""

from copy import deepcopy
from dataclasses import dataclass

import torch
import torch_remat as remat

from torchtitan.config import derive
from torchtitan.models.common.attention import (
    FlexAttentionMetadata,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock

# Tensor shape legend: T = packed tokens, D = model dimension.


class PeriLNTransformerBlock(Llama3TransformerBlock):
    """Apply output RMSNorm before each residual addition (Peri-LN)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Llama3TransformerBlock.Config):
        attention_output_norm: RMSNorm.Config
        ffn_output_norm: RMSNorm.Config

    def __init__(self, config: Config):
        super().__init__(config)
        self.attention_output_norm = config.attention_output_norm.build()
        self.ffn_output_norm = config.ffn_output_norm.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_metadata: FlexAttentionMetadata | VarlenAttentionMetadata | None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del padding_mask, aux_loss_denominator
        attn_out_TD = self.attention(
            self.attention_norm(x_TD), attention_metadata, positions
        )
        # The output norm reads saved region outputs during recomputation.
        remat.recompute_needs_tensor(attn_out_TD)
        h_TD = x_TD + self.attention_output_norm(attn_out_TD)
        ffn_out_TD = self.feed_forward(self.ffn_norm(h_TD))
        remat.recompute_needs_tensor(ffn_out_TD)
        # Reuse Llama's saved trailing add, which needs no backward state.
        return remat.region(
            torch.add, self.remat_region_name("ffn_residual"), recompute=False
        )(h_TD, self.ffn_output_norm(ffn_out_TD))


class PeriLNModel(Llama3Model):
    """Reuse the Llama decoder with native Peri-LN checkpoints."""

    # Llama's HF adapter does not represent the additional norm parameters.
    state_dict_adapter_cls = None

    @dataclass(kw_only=True, slots=True)
    class Config(Llama3Model.Config):
        pass


def apply_peri_ln(config: Llama3Model.Config) -> PeriLNModel.Config:
    """Return an independent Peri-LN config, retaining Llama initialization."""
    config = deepcopy(config)
    layers = [
        derive(
            layer,
            PeriLNTransformerBlock.Config,
            attention_output_norm=deepcopy(layer.attention_norm),
            ffn_output_norm=deepcopy(layer.ffn_norm),
        )
        for layer in config.layers
    ]
    return derive(config, PeriLNModel.Config, layers=layers)
