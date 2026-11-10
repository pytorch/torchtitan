# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch_remat as remat

from torchtitan.models.common.attention import AttentionMetadata
from torchtitan.models.common.aux_loss import register_aux_loss_zero_hook
from torchtitan.models.kimi_k3.model import KimiK3Model, KimiK3TransformerBlock
from torchtitan.models.kimi_k3.moe import KimiLatentMoE

from .state_dict_adapter import KimiK31StateDictAdapter


__all__ = ["KimiK31Model", "KimiK31TransformerBlock"]


class KimiK31TransformerBlock(KimiK3TransformerBlock):
    """Kimi K3 block whose attention is Gated DSA."""

    @dataclass(kw_only=True, slots=True)
    class Config(KimiK3TransformerBlock.Config):
        pass

    def forward(
        self,
        x_TD: torch.Tensor,
        block_residual_TND: torch.Tensor,
        attention_metadata: AttentionMetadata | None = None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.first_layer_in_block:
            block_residual_TND = torch.cat(
                (block_residual_TND, x_TD.unsqueeze(1)), dim=1
            )
            partial_block_TD = None
        else:
            partial_block_TD = x_TD

        if self.attention_res_proj is None:
            h_TD = x_TD
        else:
            assert self.attention_res_norm is not None
            h_TD = self.attention_residual(
                partial_block_TD,
                block_residual_TND,
                self.attention_res_proj,
                self.attention_res_norm,
            )
        h_TD = self.attention_norm(h_TD)
        h_TD = self.attention(
            h_TD,
            attention_metadata,
            positions,
            aux_loss_denominator=aux_loss_denominator,
            padding_mask=padding_mask,
        )
        remat.recompute_needs_tensor(h_TD)
        prefix_sum_TD = h_TD if self.first_layer_in_block else x_TD + h_TD

        h_TD = self.attention_residual(
            prefix_sum_TD,
            block_residual_TND,
            self.ffn_res_proj,
            self.ffn_res_norm,
        )
        h_TD = self.ffn_norm(h_TD)
        if isinstance(self.feed_forward, KimiLatentMoE):
            h_TD = self.feed_forward(
                h_TD,
                padding_mask_T=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )
        else:
            h_TD = self.feed_forward(h_TD)
        out_TD = remat.region(
            torch.add, self.remat_region_name("ffn_residual"), recompute=False
        )(prefix_sum_TD, h_TD)
        return out_TD, block_residual_TND


class KimiK31Model(KimiK3Model):
    """Kimi K3 with Gated DSA in place of the full-attention MLA layers."""

    state_dict_adapter_cls = KimiK31StateDictAdapter

    @dataclass(kw_only=True, slots=True)
    class Config(KimiK3Model.Config):
        pass

    @classmethod
    def _register_optimizer_hooks(
        cls, optimizers, model_parts, parallelism_context
    ) -> None:
        super()._register_optimizer_hooks(optimizers, model_parts, parallelism_context)
        register_aux_loss_zero_hook(optimizers, model_parts, parallelism_context)
