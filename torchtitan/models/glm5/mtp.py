# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""GLM-5 multi-token prediction (NextN) training components."""

from dataclasses import dataclass

import torch
import torch_remat as remat

from torchtitan.models.common.moe import MoE
from torchtitan.models.deepseek_v3.mtp import (
    MTPLoss,
    MTPTransformerBlock,
    roll_mtp_sequence,
)


class Glm5MTPTransformerBlock(MTPTransformerBlock):
    """One GLM-5 NextN depth using the checkpoint's full DSA decoder block.

    GLM-5.2 uses the same embedding/hidden fusion as DeepSeek-V3 MTP, but its
    inner block consumes the GLM DSA additive mask and may produce indexer
    selections.  Keeping this specialization here avoids weakening either
    model's attention interface.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MTPTransformerBlock.Config):
        pass

    def forward(
        self,
        mtp_input_embed: torch.Tensor,
        prev_embed: torch.Tensor,
        mtp_input_valid_mask: torch.Tensor,
        attention_masks: torch.Tensor,
        positions: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        local_valid_mask = self._maybe_shard_mtp_valid_mask_across_tp(
            mtp_input_valid_mask
        )
        prev_embed = prev_embed * local_valid_mask.unsqueeze(-1).to(prev_embed.dtype)
        fused = torch.cat(
            [self.enorm(mtp_input_embed), self.hnorm(prev_embed)], dim=-1
        )
        h = self.eh_proj(fused)

        attention_out = self.attention(
            self.attention_norm(h),
            attention_masks,
            positions,
            None,
        )
        remat.recompute_needs_tensor(attention_out)
        h = h + attention_out
        mtp_padding_mask = ~mtp_input_valid_mask
        if padding_mask is not None:
            mtp_padding_mask = mtp_padding_mask | padding_mask
        if isinstance(self.feed_forward, MoE):
            ffn_out = self.feed_forward(
                self.ffn_norm(h),
                padding_mask_T=mtp_padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )
        else:
            ffn_out = self.feed_forward(self.ffn_norm(h))
        h = remat.region(
            torch.add, self.remat_region_name("ffn_residual"), recompute=False
        )(h, ffn_out)
        remat.recompute_needs_tensor(h)
        return self.mtp_norm(h)


__all__ = [
    "Glm5MTPTransformerBlock",
    "MTPLoss",
    "roll_mtp_sequence",
]
