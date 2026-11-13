# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLM-5 multi-token prediction (NextN) training components."""

from dataclasses import dataclass

import torch
import torch_remat as remat

from torchtitan.models.common.moe import MoE
from torchtitan.models.deepseek_v3.mtp import (
    MTPCrossEntropyLoss as MTPLoss,
    MTPTransformerBlock,
    roll_mtp_sequence,
)
from .dsa import DsaMetadata


class Glm5MTPTransformerBlock(MTPTransformerBlock):
    """NextN block with explicit remat dependencies for bare tensor consumers."""

    @dataclass(kw_only=True, slots=True)
    class Config(MTPTransformerBlock.Config):
        pass

    # The GLM attention consumes compact DSA metadata instead of BlockMask.
    def forward(  # pyrefly: ignore[bad-override]
        self,
        mtp_input_embed: torch.Tensor,
        prev_embed: torch.Tensor,
        mtp_input_valid_mask: torch.Tensor,
        attention_metadata: DsaMetadata,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ):
        mtp_padding_mask_T = ~mtp_input_valid_mask
        if padding_mask is not None:
            mtp_padding_mask_T = mtp_padding_mask_T | padding_mask
        local_valid_T = self._maybe_shard_mtp_valid_mask_across_tp(mtp_input_valid_mask)
        # MTP reads the preceding decoder/depth output with a bare mask multiply.
        remat.recompute_needs_tensor(prev_embed)
        prev_embed = prev_embed * local_valid_T.unsqueeze(-1).to(prev_embed.dtype)
        h_TD = self.eh_proj(
            torch.cat([self.enorm(mtp_input_embed), self.hnorm(prev_embed)], dim=-1)
        )
        remat.recompute_needs_tensor(h_TD)
        attention_TD = self.attention(
            self.attention_norm(h_TD),
            attention_metadata,
            positions,
            aux_loss_denominator=aux_loss_denominator,
        )
        remat.recompute_needs_tensor(attention_TD)
        h_TD = h_TD + attention_TD
        if isinstance(self.feed_forward, MoE):
            ffn_TD = self.feed_forward(
                self.ffn_norm(h_TD),
                padding_mask_T=mtp_padding_mask_T,
                aux_loss_denominator=aux_loss_denominator,
            )
        else:
            ffn_TD = self.feed_forward(self.ffn_norm(h_TD))
        remat.recompute_needs_tensor(ffn_TD)
        return self.mtp_norm(h_TD + ffn_TD)


__all__ = ["Glm5MTPTransformerBlock", "MTPLoss", "roll_mtp_sequence"]
