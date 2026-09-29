# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.deepseek_v3.mtp import MTPLoss, roll_mtp_sequence

from .model import DeepSeekV4TransformerBlock

if TYPE_CHECKING:
    from torchtitan.models.common.attention import AttentionMasksType

    from .mhc import HcHead


class MTPBlock(DeepSeekV4TransformerBlock):
    """One auxiliary multi-token prediction depth for DeepSeek V4."""

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV4TransformerBlock.Config):
        enorm: RMSNorm.Config
        hnorm: RMSNorm.Config
        e_proj: Linear.Config
        h_proj: Linear.Config
        mtp_norm: RMSNorm.Config
        hc_head: "HcHead.Config"

    def __init__(self, config: Config):
        super().__init__(config)
        self.enorm = config.enorm.build()
        self.hnorm = config.hnorm.build()
        self.e_proj = config.e_proj.build()
        self.h_proj = config.h_proj.build()
        self.mtp_norm = config.mtp_norm.build()
        self.hc_head = config.hc_head.build()

    def forward(  # pyrefly: ignore[bad-param-name-override]
        self,
        mtp_input_embed: torch.Tensor,
        prev_hc_hidden: torch.Tensor,
        mtp_input_ids_T: torch.Tensor,
        mtp_input_valid_mask: torch.Tensor,
        attention_masks: "AttentionMasksType | None",
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prev_hc_hidden.ndim != 3:
            raise ValueError(
                "DeepSeek V4 MTP expects an HC hidden state with shape "
                "[tokens, hc_mult, hidden], got "
                f"{tuple(prev_hc_hidden.shape)}."
            )

        valid_mask = mtp_input_valid_mask.view(-1, 1, 1).to(dtype=prev_hc_hidden.dtype)
        prev_hc_hidden = prev_hc_hidden * valid_mask

        hidden = self.e_proj(self.enorm(mtp_input_embed)).unsqueeze(1)
        hidden = hidden + self.h_proj(self.hnorm(prev_hc_hidden))
        next_hc_hidden = super().forward(
            hidden,
            mtp_input_ids_T,
            attention_masks,
            positions,
        )
        prediction_hidden = self.hc_head(next_hc_hidden)
        prediction_hidden = self.mtp_norm(prediction_hidden)
        return next_hc_hidden, prediction_hidden


__all__ = ["MTPBlock", "MTPLoss", "roll_mtp_sequence"]
